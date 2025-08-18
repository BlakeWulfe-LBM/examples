#!/usr/bin/env python3
"""
Script to check whether torchvision transforms v2 run on GPU or CPU,
and compare batch vs sequential performance.
"""

import time
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F
import numpy as np
from typing import List, Tuple
import gc


def check_device_info():
    """Check and display device information."""
    print("=== Device Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print()


def create_test_images(batch_size: int = 8, height: int = 224, width: int = 224) -> torch.Tensor:
    """Create test images for benchmarking."""
    # Create random images on CPU first
    images = torch.randn(batch_size, 3, height, width)
    return images


def check_tensor_device(tensor: torch.Tensor, name: str) -> str:
    """Check which device a tensor is on."""
    device = tensor.device
    dtype = tensor.dtype
    return str(device)


def benchmark_transforms_sequential(images: torch.Tensor, transform_list: List, num_runs: int = 100) -> Tuple[float, List[str]]:
    """Benchmark transforms applied sequentially to individual images."""
    print(f"Benchmarking sequential transforms ({num_runs} runs)...")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images_gpu = images.to(device)
    
    # Warm up
    for _ in range(10):
        for i in range(images_gpu.shape[0]):
            img = images_gpu[i:i+1]
            for transform in transform_list:
                img = transform(img)
    
    # Benchmark
    start_time = time.time()
    device_trace = []
    
    for _ in range(num_runs):
        for i in range(images_gpu.shape[0]):
            img = images_gpu[i:i+1]
            for transform in transform_list:
                img = transform(img)
                device_trace.append(check_tensor_device(img, f"Transform {transform.__class__.__name__}"))
    
    end_time = time.time()
    sequential_time = end_time - start_time
    
    print(f"Sequential time: {sequential_time:.4f}s")
    return sequential_time, device_trace


def benchmark_transforms_batch(images: torch.Tensor, transform_list: List, num_runs: int = 100) -> Tuple[float, List[str]]:
    """Benchmark transforms applied in batch to all images at once."""
    print(f"Benchmarking batch transforms ({num_runs} runs)...")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images_gpu = images.to(device)
    
    # Warm up
    for _ in range(10):
        batch = images_gpu.clone()
        for transform in transform_list:
            batch = transform(batch)
    
    # Benchmark
    start_time = time.time()
    device_trace = []
    
    for _ in range(num_runs):
        batch = images_gpu.clone()
        for transform in transform_list:
            batch = transform(batch)
            device_trace.append(check_tensor_device(batch, f"Batch transform {transform.__class__.__name__}"))
    
    end_time = time.time()
    batch_time = end_time - start_time
    
    print(f"Batch time: {batch_time:.4f}s")
    return batch_time, device_trace


def main():
    """Main function to run all checks."""
    print("Torchvision Transforms v2 GPU Usage Checker")
    print("=" * 50)
    
    # Check device info
    check_device_info()
    
    # Create test images
    batch_size = 80
    images = create_test_images(batch_size=batch_size)
    print(f"Created {batch_size} test images of shape {images.shape}")
    print()
    
    # Define a realistic transform pipeline
    transform_pipeline = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(size=(224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    print("=== Performance Benchmarking ===")
    
    # Benchmark sequential vs batch
    sequential_time, sequential_devices = benchmark_transforms_sequential(images, transform_pipeline, num_runs=50)
    print()
    
    batch_time, batch_devices = benchmark_transforms_batch(images, transform_pipeline, num_runs=50)
    print()
    
    # Calculate speedup
    if sequential_time > 0:
        speedup = sequential_time / batch_time
        print(f"Batch processing is {speedup:.2f}x faster than sequential")
    else:
        print("Could not calculate speedup")
    
    # Memory usage check
    if torch.cuda.is_available():
        print(f"\nGPU memory after sequential: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Run batch again to check memory
        _ = benchmark_transforms_batch(images, transform_pipeline, num_runs=10)
        print(f"GPU memory after batch: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    print("\n=== Summary ===")
    print("Torchvision transforms v2:")
    if torch.cuda.is_available():
        print("- Can run on GPU when input tensors are on GPU")
        print("- Device placement follows the input tensor's device")
        print("- Batch processing is generally faster than sequential")
    else:
        print("- Running on CPU (CUDA not available)")
        print("- Batch processing may still be faster due to vectorization")


if __name__ == "__main__":
    main() 