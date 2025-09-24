#!/usr/bin/env python3
"""
Minimal test to understand why GPU augmentation appears faster than no augmentation.
"""

import time
import torch
import torch.nn as nn
import numpy as np

def test_gpu_operations():
    """Test if GPU operations somehow optimize the pipeline."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create dummy data
    batch_size = 32
    num_batches = 100
    data = torch.randn(num_batches, batch_size, 3, 224, 224)
    
    # Simple model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10)
    ).to(device)
    model.eval()
    
    # Test 1: No augmentation
    print("\nTest 1: No augmentation")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for i in range(num_batches):
        batch = data[i].to(device)
        with torch.no_grad():
            output = model(batch)
        torch.cuda.synchronize()
    
    no_aug_time = time.perf_counter() - start
    print(f"Time: {no_aug_time:.3f}s")
    
    # Test 2: With GPU operation (simulating augmentation)
    print("\nTest 2: With GPU multiplication (simulating augmentation)")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for i in range(num_batches):
        batch = data[i].to(device)
        # Add a simple GPU operation
        batch = batch * 1.01  # Minimal "augmentation"
        with torch.no_grad():
            output = model(batch)
        torch.cuda.synchronize()
    
    with_aug_time = time.perf_counter() - start
    print(f"Time: {with_aug_time:.3f}s")
    
    # Test 3: Check if it's about memory allocation
    print("\nTest 3: Pre-allocated GPU tensors")
    gpu_data = data.to(device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for i in range(num_batches):
        batch = gpu_data[i]  # Already on GPU
        with torch.no_grad():
            output = model(batch)
        torch.cuda.synchronize()
    
    pre_allocated_time = time.perf_counter() - start
    print(f"Time: {pre_allocated_time:.3f}s")
    
    # Test 4: Pre-allocated with augmentation
    print("\nTest 4: Pre-allocated GPU tensors with augmentation")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for i in range(num_batches):
        batch = gpu_data[i]  # Already on GPU
        batch = batch * 1.01  # Minimal "augmentation"
        with torch.no_grad():
            output = model(batch)
        torch.cuda.synchronize()
    
    pre_allocated_aug_time = time.perf_counter() - start
    print(f"Time: {pre_allocated_aug_time:.3f}s")
    
    print("\n" + "="*50)
    print("RESULTS:")
    print(f"No aug (CPU->GPU):        {no_aug_time:.3f}s (baseline)")
    print(f"With aug (CPU->GPU):      {with_aug_time:.3f}s ({with_aug_time/no_aug_time:.2f}x)")
    print(f"Pre-allocated (no aug):   {pre_allocated_time:.3f}s ({pre_allocated_time/no_aug_time:.2f}x)")
    print(f"Pre-allocated (with aug): {pre_allocated_aug_time:.3f}s ({pre_allocated_aug_time/no_aug_time:.2f}x)")
    
    if with_aug_time < no_aug_time:
        print("\n🚨 BUG: Augmentation is faster than no augmentation!")
    else:
        print("\n✅ Results are logical")


def test_dataloader_behavior():
    """Test if DataLoader behavior changes with different processing."""
    
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleDataset(Dataset):
        def __init__(self, size=1000):
            self.size = size
            self.data = torch.randn(size, 3, 224, 224)
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Simulate some CPU work (image loading)
            time.sleep(0.001)  # 1ms to simulate I/O
            return self.data[idx], 0
    
    device = 'cuda'
    dataset = SimpleDataset(size=320)
    batch_size = 32
    num_batches = 10
    
    print("\n" + "="*50)
    print("DATALOADER TEST")
    print("="*50)
    
    for num_workers in [0, 2, 4]:
        print(f"\nTesting with {num_workers} workers:")
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        
        # Test 1: No GPU operations
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        count = 0
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            torch.cuda.synchronize()
            count += images.size(0)
        
        no_op_time = time.perf_counter() - start
        print(f"  No GPU ops: {count/no_op_time:.1f} samples/sec")
        
        # Test 2: With GPU operations
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        count = 0
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            images = images * 1.01  # GPU operation
            torch.cuda.synchronize()
            count += images.size(0)
        
        with_op_time = time.perf_counter() - start
        print(f"  With GPU ops: {count/with_op_time:.1f} samples/sec")
        
        if with_op_time < no_op_time:
            print(f"  🚨 GPU ops made it FASTER by {(no_op_time/with_op_time - 1)*100:.1f}%")


if __name__ == '__main__':
    print("Testing GPU operation timing behavior...")
    test_gpu_operations()
    test_dataloader_behavior()