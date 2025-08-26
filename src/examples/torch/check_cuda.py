#!/usr/bin/env python3
"""
CUDA Availability Checker
Checks if CUDA is available and provides detailed GPU information
"""

import sys
import subprocess

def check_cuda():
    try:
        import torch
    except ImportError:
        print("❌ PyTorch is not installed!")
        print("Install it with: pip install torch")
        return
    
    print("=" * 60)
    print("CUDA AVAILABILITY CHECK")
    print("=" * 60)
    
    # Basic CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n✓ PyTorch version: {torch.__version__}")
    
    if cuda_available:
        print(f"✅ CUDA is AVAILABLE")
    else:
        print(f"❌ CUDA is NOT AVAILABLE")
        print("\nPossible reasons:")
        print("  - No NVIDIA GPU present")
        print("  - NVIDIA drivers not installed")
        print("  - PyTorch installed is CPU-only version")
        print("  - Driver/CUDA version mismatch")
        return

    nccl_version = torch.cuda.nccl.version()
    print(f"NCCL version: {nccl_version}")
    breakpoint()
    
    # Detailed CUDA information
    print(f"\n📊 CUDA Details:")
    print(f"  - CUDA version PyTorch was built with: {torch.version.cuda}")
    print(f"  - cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  - cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    # GPU information
    gpu_count = torch.cuda.device_count()
    print(f"\n🎮 GPU Information:")
    print(f"  - Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\n  GPU {i}:")
        print(f"    - Name: {torch.cuda.get_device_name(i)}")
        
        # Get compute capability
        capability = torch.cuda.get_device_capability(i)
        print(f"    - Compute Capability: {capability[0]}.{capability[1]}")
        
        # Memory information
        mem_total = torch.cuda.get_device_properties(i).total_memory
        mem_allocated = torch.cuda.memory_allocated(i)
        mem_reserved = torch.cuda.memory_reserved(i)
        
        print(f"    - Total Memory: {mem_total / 1024**3:.2f} GB")
        print(f"    - Allocated Memory: {mem_allocated / 1024**3:.2f} GB")
        print(f"    - Reserved Memory: {mem_reserved / 1024**3:.2f} GB")
    
    # Current device
    current_device = torch.cuda.current_device()
    print(f"\n🎯 Current GPU device: {current_device}")
    
    # Try to get NVIDIA-SMI information
    print("\n💻 System NVIDIA Driver Info:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True, check=True)
        driver_version = result.stdout.strip()
        print(f"  - NVIDIA Driver Version: {driver_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  - Could not retrieve driver version (nvidia-smi not available)")
    
    # Test CUDA with a simple operation
    print("\n🧪 Testing CUDA with tensor operation...")
    try:
        # Create a tensor on GPU
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = torch.tensor([4.0, 5.0, 6.0]).cuda()
        z = x + y
        print(f"  - Test successful! Result: {z.cpu().numpy()}")
    except Exception as e:
        print(f"  - Test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ CUDA check complete!")
    print("=" * 60)

def check_requirements():
    """Check if PyTorch is installed and provide installation instructions"""
    print("\n📦 Checking PyTorch installation...")
    
    try:
        import torch
        return True
    except ImportError:
        print("\n❌ PyTorch is not installed!")
        print("\nTo install PyTorch with CUDA support:")
        print("  1. Visit: https://pytorch.org/get-started/locally/")
        print("  2. Select your preferences (OS, Package, CUDA version)")
        print("  3. Run the generated command\n")
        print("Example for CUDA 12.6:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
        print("\nFor CPU-only version:")
        print("  pip install torch torchvision torchaudio")
        return False

if __name__ == "__main__":
    if check_requirements():
        check_cuda()
    else:
        sys.exit(1)