#!/usr/bin/env python3
"""
Diagnostic script for NCCL communication issues.
Run with: torchrun --nproc_per_node=2 diagnose_nccl.py
"""

import os
import sys
import subprocess
import torch
import torch.distributed as dist
from datetime import timedelta

def run_command(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "Command failed"

def diagnose_system():
    """Print system diagnostic information."""
    rank = int(os.environ.get('RANK', -1))
    
    if rank == 0:
        print("=" * 70)
        print("SYSTEM DIAGNOSTICS")
        print("=" * 70)
        
        # GPU Information
        print("\n📊 GPU Information:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    - Compute Capability: {props.major}.{props.minor}")
                print(f"    - Memory: {props.total_memory / 1e9:.2f} GB")
        
        # NVIDIA-SMI output
        print("\n📊 nvidia-smi topo -m (GPU Topology):")
        topo = run_command("nvidia-smi topo -m")
        print(topo)
        
        # Check IOMMU
        print("\n📊 IOMMU Status:")
        iommu = run_command("cat /proc/cmdline | grep -o 'iommu=[^ ]*' || echo 'IOMMU not explicitly set'")
        print(f"  {iommu}")
        
        # Check ACS
        print("\n📊 PCIe ACS Status:")
        acs = run_command("lspci -vvv 2>/dev/null | grep -i 'access control services' | head -5")
        if acs:
            print(f"  ACS found (may need to be disabled for P2P)")
        else:
            print(f"  No ACS info available")
        
        print("=" * 70)

def test_p2p_access():
    """Test GPU peer-to-peer access."""
    rank = int(os.environ.get('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    if rank == 0 and torch.cuda.is_available():
        print("\n🔍 Testing P2P Access Between GPUs:")
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            for j in range(device_count):
                if i != j:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    print(f"  GPU {i} -> GPU {j}: {'✅ Yes' if can_access else '❌ No'}")
        
        print("\n🔍 Checking CUDA_VISIBLE_DEVICES:")
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all GPUs visible)')
        print(f"  {cvd}")
        print()

def test_basic_nccl():
    """Test NCCL with different configurations."""
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    print(f"[Rank {rank}] Starting NCCL diagnostic tests...")
    
    if not torch.cuda.is_available():
        print(f"[Rank {rank}] ⚠️  CUDA not available, skipping NCCL tests")
        return
    
    # Set CUDA device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Try different NCCL configurations
    configs = [
        {
            "name": "P2P Disabled + Simple Ring Algorithm",
            "env": {
                "NCCL_P2P_DISABLE": "0",
                "NCCL_ALGO": "Ring",
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,ENV"
            }
        },
        {
            "name": "Default Configuration",
            "env": {
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,ENV"
            }
        },
        {
            "name": "Tree Algorithm Only",
            "env": {
                "NCCL_ALGO": "Tree",
                "NCCL_DEBUG": "INFO"
            }
        }
    ]
    
    for config_idx, config in enumerate(configs):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Testing Configuration {config_idx + 1}: {config['name']}")
            print(f"{'='*60}")
        
        # Apply environment variables
        for key, value in config['env'].items():
            os.environ[key] = value
            if rank == 0:
                print(f"  Set {key}={value}")
        
        try:
            # Initialize process group
            if dist.is_initialized():
                dist.destroy_process_group()
            
            print(f"[Rank {rank}] Initializing with {config['name']}...")
            dist.init_process_group(
                backend='nccl',
                timeout=timedelta(seconds=10)  # Shorter timeout for testing
            )
            
            # Test 1: Barrier
            print(f"[Rank {rank}] Testing barrier...")
            dist.barrier()
            print(f"[Rank {rank}] ✅ Barrier passed!")
            
            # Test 2: Broadcast
            print(f"[Rank {rank}] Testing broadcast...")
            tensor = torch.ones(1, device=device) * rank
            dist.broadcast(tensor, src=0)
            print(f"[Rank {rank}] ✅ Broadcast passed! Value: {tensor.item()}")
            
            # Test 3: Small AllReduce
            print(f"[Rank {rank}] Testing small all-reduce...")
            tensor = torch.ones(1, device=device) * (rank + 1)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = sum(range(1, world_size + 1))
            print(f"[Rank {rank}] ✅ AllReduce passed! Result: {tensor.item()} (expected: {expected})")
            
            # Clean up
            dist.barrier()
            dist.destroy_process_group()
            
            if rank == 0:
                print(f"\n✅ Configuration '{config['name']}' SUCCEEDED!")
            
            # If we found a working configuration, report it
            return True
            
        except Exception as e:
            if rank == 0:
                print(f"\n❌ Configuration '{config['name']}' FAILED:")
                print(f"   Error: {e}")
            
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except:
                    pass
            
            # Continue to next configuration
            continue
    
    return False

def suggest_fixes():
    """Suggest potential fixes based on diagnostic results."""
    rank = int(os.environ.get('RANK', -1))
    
    if rank == 0:
        print("\n" + "="*70)
        print("💡 SUGGESTED FIXES")
        print("="*70)
        print("""
1. **Disable P2P Communication** (Most Common Fix):
   ```bash
   export NCCL_P2P_DISABLE=1
   torchrun --nproc_per_node=2 test_distributed.py
   ```

2. **Disable IOMMU** (Requires Reboot):
   Add to kernel parameters: `iommu=off` or `intel_iommu=off`
   Edit /etc/default/grub and update GRUB_CMDLINE_LINUX

3. **Disable PCIe ACS** (If applicable):
   Add to kernel parameters: `pcie_acs_override=downstream,multifunction`

4. **Use Different NCCL Algorithm**:
   ```bash
   export NCCL_ALGO=Tree
   # or
   export NCCL_ALGO=Ring
   ```

5. **Set CUDA Device Order**:
   ```bash
   export CUDA_DEVICE_ORDER=PCI_BUS_ID
   ```

6. **For Docker/Containers**:
   Add: `--gpus all --ipc=host` flags

7. **Update NCCL** (if old version):
   The version can be checked with: `python -c "import torch; print(torch.cuda.nccl.version())"`

Try these in order, starting with #1 which usually works!
""")
        print("="*70)

if __name__ == "__main__":
    # Check if running under torchrun
    if 'RANK' not in os.environ:
        print("Error: This script must be run with torchrun")
        print("Example: torchrun --nproc_per_node=2 diagnose_nccl.py")
        sys.exit(1)
    
    rank = int(os.environ.get('RANK', -1))
    
    # System diagnostics (rank 0 only)
    if rank == 0:
        diagnose_system()
        test_p2p_access()
    
    # Wait for rank 0 to finish diagnostics
    if 'RANK' in os.environ:
        import time
        time.sleep(2)
    
    # Test NCCL configurations
    success = test_basic_nccl()
    
    # Suggest fixes if all tests failed
    if not success:
        suggest_fixes()