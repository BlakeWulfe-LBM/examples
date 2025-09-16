#!/usr/bin/env python3
"""
Simple script to test torch distributed setup.
Run with: torchrun --nproc_per_node=2 test_distributed.py
"""

import os
import sys
import time
import signal
import torch
import torch.distributed as dist
from datetime import timedelta

def timeout_handler(signum, frame):
    print(f"[Rank {os.environ.get('RANK', '?')}] Timeout! Process hanging detected.")
    sys.exit(1)

def test_distributed():
    # Set timeout (60 seconds for safer testing)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    
    try:
        # Get distributed info from environment
        rank = int(os.environ.get('RANK', -1))
        world_size = int(os.environ.get('WORLD_SIZE', -1))
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        
        print(f"[Rank {rank}] Starting distributed test...")
        print(f"[Rank {rank}] World size: {world_size}, Local rank: {local_rank}")
        
        # Determine backend
        use_cuda = torch.cuda.is_available() and os.environ.get('FORCE_CPU', '0') != '1'
        
        if use_cuda:
            # Set CUDA device BEFORE any CUDA operations
            torch.cuda.set_device(local_rank)
            print(f"[Rank {rank}] Using CUDA device {local_rank}")
            backend = 'nccl'
            
            # Set NCCL environment variables for better debugging
            os.environ['NCCL_DEBUG'] = 'INFO'
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
            # Try P2P disable which sometimes helps with multi-GPU issues
            os.environ['NCCL_P2P_DISABLE'] = '0'
        else:
            print(f"[Rank {rank}] Using CPU (Gloo backend)")
            backend = 'gloo'
        
        # Initialize process group with explicit timeout
        print(f"[Rank {rank}] Initializing process group with backend: {backend}...")
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=30)
        )
        
        # Verify initialization
        assert dist.is_initialized(), "Failed to initialize distributed"
        assert dist.get_rank() == rank, "Rank mismatch"
        assert dist.get_world_size() == world_size, "World size mismatch"
        
        print(f"[Rank {rank}] Process group initialized successfully!")
        print(f"[Rank {rank}] Backend: {dist.get_backend()}")
        
        # Test 1: Simple barrier first (less complex than all-reduce)
        print(f"[Rank {rank}] Testing barrier first...")
        dist.barrier()
        print(f"[Rank {rank}] Barrier test passed!")
        
        # Test 2: Broadcast (simpler than all-reduce)
        print(f"[Rank {rank}] Testing broadcast...")
        if backend == 'nccl':
            device = torch.device(f'cuda:{local_rank}')
            broadcast_tensor = torch.ones(1, device=device) * rank
        else:
            broadcast_tensor = torch.ones(1) * rank
        
        dist.broadcast(broadcast_tensor, src=0)
        print(f"[Rank {rank}] Broadcast complete! Value: {broadcast_tensor.item()}")
        
        # Test 3: All-reduce
        print(f"[Rank {rank}] Creating tensor for all-reduce...")
        if backend == 'nccl':
            device = torch.device(f'cuda:{local_rank}')
            tensor = torch.ones(1, device=device) * (rank + 1)
        else:
            tensor = torch.ones(1) * (rank + 1)
        
        print(f"[Rank {rank}] Initial tensor value: {tensor.item()}")
        print(f"[Rank {rank}] Performing all-reduce...")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Calculate expected sum: 1 + 2 + ... + world_size
        expected = sum(range(1, world_size + 1))
        actual = tensor.item()
        
        print(f"[Rank {rank}] All-reduce complete! Result: {actual} (expected: {expected})")
        assert abs(actual - expected) < 1e-6, f"All-reduce failed: expected {expected}, got {actual}"
        
        # Final barrier
        print(f"[Rank {rank}] Final barrier...")
        dist.barrier()
        print(f"[Rank {rank}] Final barrier passed!")
        
        # Cleanup
        print(f"[Rank {rank}] Cleaning up...")
        dist.destroy_process_group()
        print(f"[Rank {rank}] ✓ All tests passed!")
        
        # Cancel timeout
        signal.alarm(0)
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        signal.alarm(0)
        sys.exit(1)

if __name__ == "__main__":
    # Check if running under torchrun
    if 'RANK' not in os.environ:
        print("Error: This script must be run with torchrun")
        print("Example: torchrun --nproc_per_node=2 test_distributed.py")
        print("\nTo force CPU mode: FORCE_CPU=1 torchrun --nproc_per_node=2 test_distributed.py")
        sys.exit(1)
    
    # Print environment info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    test_distributed()