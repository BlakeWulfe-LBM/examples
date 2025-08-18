#!/usr/bin/env python3
"""
Data Augmentation Profiling Script - Valid Implementation

THIS SCRIPT COMPARES TWO FUNDAMENTALLY DIFFERENT APPROACHES:

1. CPU PER-SAMPLE AUGMENTATION (Traditional):
   - Augmentations applied to INDIVIDUAL images in DataLoader workers
   - Each worker processes one image at a time through the transform pipeline
   - Happens during data loading, before batching
   - Example: transforms.RandomCrop applied to each 256x256 image separately

2. GPU BATCH AUGMENTATION (Modern):
   - Images are loaded and converted to tensors with minimal CPU processing
   - After batching, the ENTIRE BATCH is transferred to GPU
   - Augmentations applied to the whole batch at once on GPU

Key Differences Measured:
- CPU: Sequential processing of N images
- GPU: Parallel processing of N images as one operation
- CPU: No transfer overhead but limited parallelism
- GPU: Transfer overhead but massive parallelism

WITH FORWARD PASS SIMULATION:
When a model forward pass is simulated, it reveals how data loading bottlenecks
change based on compute time:
- Fast models: Data loading/augmentation can be the bottleneck
- Slow models: GPU may be idle waiting for compute, making CPU augmentation viable
- The optimal strategy depends on the relative speed of augmentation vs model

The script tests:
- CPU augmentation via torchvision.transforms (per-sample)
- GPU augmentation via torchvision.transforms.v2 (batch processing)
- Various configurations to find optimal batch sizes and worker counts
- Impact of model forward pass time on the optimal augmentation strategy
"""

import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import argparse
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings
from PIL import Image
import io

# Try to import optional GPU augmentation libraries
try:
    import torchvision.transforms.v2 as transforms_v2
    TRANSFORMS_V2_AVAILABLE = True
except ImportError:
    TRANSFORMS_V2_AVAILABLE = False
    warnings.warn("torchvision.transforms.v2 not available. Update torchvision to latest version.")


@dataclass
class ProfilingResult:
    """Container for profiling results."""
    total_time: float
    total_samples: int
    samples_per_second: float
    mean_batch_time: float
    std_batch_time: float
    augmentation_time: float
    data_loading_time: float
    transfer_time: float
    forward_pass_time: float
    device: str
    augmentation_type: str
    batch_size: int
    num_workers: int
    with_forward_pass: bool
    forward_pass_ms: float
    model_complexity: Optional[str]


class SyntheticImageDataset(Dataset):
    """Dataset that simulates real image loading with optional CPU transforms."""
    
    def __init__(
        self, 
        size: int = 1000, 
        image_size: Tuple[int, int, int] = (3, 256, 256),
        transform_fns: Optional[Any] = None,
        simulate_disk_io: bool = True
    ):
        self.size = size
        self.image_size = image_size
        self.transform_fns = transform_fns
        self.simulate_disk_io = simulate_disk_io
        
        # Pre-generate synthetic image data (simulating compressed images on disk)
        self.synthetic_images = []
        for _ in range(size):
            # Create random image data
            img_array = np.random.randint(0, 256, 
                                         (image_size[1], image_size[2], image_size[0]), 
                                         dtype=np.uint8)
            if simulate_disk_io:
                # Convert to JPEG bytes to simulate disk storage
                img = Image.fromarray(img_array, mode='RGB' if image_size[0] == 3 else 'L')
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                self.synthetic_images.append(buffer.getvalue())
            else:
                self.synthetic_images.append(img_array)
        
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if self.simulate_disk_io:
            # Simulate reading and decoding image from disk
            img_bytes = self.synthetic_images[idx]
            img = Image.open(io.BytesIO(img_bytes))
        else:
            # Direct numpy array to PIL
            img_array = self.synthetic_images[idx]
            img = Image.fromarray(img_array, mode='RGB' if self.image_size[0] == 3 else 'L')
        
        # Apply transforms if provided, otherwise just convert to tensor
        if self.transform_fns is not None:
            img = self.transform_fns(img)
        else:
            # Default: just convert to tensor
            img = transforms.ToTensor()(img)
        
        return img, self.labels[idx]


def create_cpu_transforms() -> transforms.Compose:
    """Create CPU-based augmentation pipeline using torchvision."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_v2_transforms_gpu() -> Any:
    """Create GPU-compatible batch transforms using torchvision.transforms.v2."""
    if not TRANSFORMS_V2_AVAILABLE:
        return None
    
    # These v2 transforms can operate on batched tensors on GPU
    return torch.nn.Sequential(
        transforms_v2.Resize((256, 256), antialias=True),
        transforms_v2.RandomCrop(224),
        transforms_v2.RandomHorizontalFlip(p=0.5),
        transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )


def create_dummy_model(device: str = 'cuda', complexity: str = 'medium') -> torch.nn.Module:
    """Create a dummy model for simulating forward pass.
    
    Args:
        device: Device to place model on
        complexity: Model complexity - 'light', 'medium', 'heavy'
    """
    if complexity == 'light':
        # Simple model - fast forward pass
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 10)
        )
    elif complexity == 'medium':
        # Medium complexity - moderate forward pass time
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 10)
        )
    elif complexity == 'heavy':
        # Complex model - slow forward pass
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 1000),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1000, 10)
        )
    else:  
        raise ValueError(f"Invalid model complexity: {complexity}")
    
    return model.to(device)


def simulate_forward_pass(images: torch.Tensor, model: Optional[torch.nn.Module] = None, 
                         target_time_ms: float = 50.0, device: str = 'cuda') -> torch.Tensor:
    """Simulate a forward pass with controlled execution time.
    
    Args:
        images: Input batch
        model: Model to use for forward pass
        target_time_ms: Target time for forward pass in milliseconds
        device: Device to run on
    """
    if model is None:
        return images.mean()  # Minimal computation
    
    # Do the actual forward pass
    output = model(images)
    
    # Add controlled delay to reach target time if needed
    if device == 'cuda' and target_time_ms > 0:
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Busy wait with GPU computation to simulate longer processing
        while (time.perf_counter() - start) * 1000 < target_time_ms:
            dummy = torch.mm(images.view(images.size(0), -1)[:, :100], 
                            torch.randn(100, 100, device=device))
            torch.cuda.synchronize()
    
    return output


def profile_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: str,
    augmentation_type: str,
    gpu_transforms: Optional[torch.nn.Module] = None,
    num_batches: Optional[int] = None,
    warmup_batches: int = 5,
    should_simulate_forward_pass: bool = False,
    forward_pass_time_ms: float = 50.0,
    model_complexity: str = 'medium'
) -> ProfilingResult:
    """Profile a specific data loading and augmentation configuration.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers
        device: 'cpu' or 'cuda'
        augmentation_type: Name of augmentation strategy
        gpu_transforms: Optional GPU batch transforms
        num_batches: Number of batches to process (None = all)
        warmup_batches: Number of warmup batches
        should_simulate_forward_pass: Whether to simulate model forward pass
        forward_pass_time_ms: Simulated forward pass time in milliseconds
        model_complexity: Complexity of dummy model ('light', 'medium', 'heavy')
    """
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=(num_workers > 0)
    )
    
    # Create dummy model if simulating forward pass
    model = None
    if should_simulate_forward_pass:
        model = create_dummy_model(device, model_complexity)
        model.eval()
    
    # Warmup
    warmup_iter = iter(loader)
    for _ in range(min(warmup_batches, len(loader))):
        try:
            images, labels = next(warmup_iter)
            if device == 'cuda':
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                if gpu_transforms is not None:
                    with torch.no_grad():
                        images = gpu_transforms(images)
                if should_simulate_forward_pass:
                    _ = simulate_forward_pass(images, model, forward_pass_time_ms, device)
        except StopIteration:
            break
    
    # Clear GPU cache after warmup
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Profiling
    batch_times = []
    augmentation_times = []
    transfer_times = []
    data_loading_times = []
    forward_pass_times = []
    total_samples = 0
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    total_start_time = time.perf_counter()
    
    for batch_idx, (images, labels) in enumerate(loader):
        if num_batches is not None and batch_idx >= num_batches:
            break
        
        batch_start = time.perf_counter()
        
        # Data loading time (already measured by DataLoader)
        data_load_time = time.perf_counter() - batch_start
        data_loading_times.append(data_load_time)
        
        # Transfer to GPU if needed
        transfer_start = time.perf_counter()
        if device == 'cuda':
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            torch.cuda.synchronize()
        transfer_time = time.perf_counter() - transfer_start
        transfer_times.append(transfer_time)
        
        # Apply GPU augmentations if provided
        aug_start = time.perf_counter()
        if gpu_transforms is not None and device == 'cuda':
            with torch.no_grad():
                images = gpu_transforms(images)
            torch.cuda.synchronize()
        aug_time = time.perf_counter() - aug_start
        augmentation_times.append(aug_time)

        # Simulate forward pass if requested
        forward_start = time.perf_counter()
        if should_simulate_forward_pass:
            _ = simulate_forward_pass(images, model, forward_pass_time_ms, device)
            if device == 'cuda':
                torch.cuda.synchronize()
        else:
            # Minimal processing to keep data moving
            with torch.no_grad():
                _ = images.mean()
        forward_time = time.perf_counter() - forward_start
        forward_pass_times.append(forward_time)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        batch_end = time.perf_counter()
        batch_times.append(batch_end - batch_start)
        total_samples += images.size(0)
    
    total_end_time = time.perf_counter()
    total_time = total_end_time - total_start_time
    
    return ProfilingResult(
        total_time=total_time,
        total_samples=total_samples,
        samples_per_second=total_samples / total_time if total_time > 0 else 0,
        mean_batch_time=np.mean(batch_times),
        std_batch_time=np.std(batch_times),
        augmentation_time=np.mean(augmentation_times),
        data_loading_time=np.mean(data_loading_times),
        transfer_time=np.mean(transfer_times),
        forward_pass_time=np.mean(forward_pass_times),
        device=device,
        augmentation_type=augmentation_type,
        batch_size=batch_size,
        num_workers=num_workers,
        with_forward_pass=should_simulate_forward_pass,
        forward_pass_ms=forward_pass_time_ms if should_simulate_forward_pass else 0,
        model_complexity=model_complexity if should_simulate_forward_pass else None
    )


def run_comprehensive_profiling(
    batch_sizes: List[int],
    num_workers_list: List[int],
    dataset_size: int = 1000,
    num_runs: int = 3,
    num_batches: Optional[int] = None,
    simulate_disk_io: bool = True,
    should_simulate_forward_pass: bool = False,
    forward_pass_time_ms: float = 50.0,
    model_complexity: str = 'medium'
) -> Dict[str, Any]:
    """Run comprehensive profiling of different augmentation strategies.
    
    Args:
        batch_sizes: List of batch sizes to test
        num_workers_list: List of worker counts to test
        dataset_size: Number of samples in dataset
        num_runs: Number of runs per configuration
        num_batches: Number of batches to process (None = all)
        simulate_disk_io: Whether to simulate disk I/O
        should_simulate_forward_pass: Whether to simulate model forward pass
        forward_pass_time_ms: Simulated forward pass time in milliseconds
        model_complexity: Complexity of dummy model
    """
    
    results = {
        'configurations': [],
        'results': [],
        'summary': {},
        'forward_pass_enabled': should_simulate_forward_pass,
        'forward_pass_ms': forward_pass_time_ms if should_simulate_forward_pass else 0,
        'model_complexity': model_complexity if should_simulate_forward_pass else None
    }
    
    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"GPU Available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available, skipping GPU tests")
    
    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            config = {
                'batch_size': batch_size,
                'num_workers': num_workers
            }
            results['configurations'].append(config)
            
            print(f"\n{'='*60}")
            print(f"Testing: batch_size={batch_size}, num_workers={num_workers}")
            if should_simulate_forward_pass:
                print(f"         forward_pass={forward_pass_time_ms}ms, complexity={model_complexity}")
            print(f"{'='*60}")
            
            # Test configurations
            test_configs = []
            
            # Add GPU configurations if available
            if cuda_available:
                # BASELINE: No augmentation, just transfer to GPU
                test_configs.append({
                    'name': 'no_aug',
                    'dataset': SyntheticImageDataset(
                        size=dataset_size,
                        transform_fns=transforms.ToTensor(),  # Just convert to tensor
                        simulate_disk_io=simulate_disk_io
                    ),
                    'device': 'cuda',
                    'gpu_transforms': None
                })

                # CPU augmentation, then transfer to GPU
                test_configs.append({
                    'name': 'cpu_aug',
                    'dataset': SyntheticImageDataset(
                        size=dataset_size,
                        transform_fns=create_cpu_transforms(),
                        simulate_disk_io=simulate_disk_io
                    ),
                    'device': 'cuda',
                    'gpu_transforms': None
                })

                # GPU BATCH AUGMENTATION with torchvision.v2:
                # - CPU: Load images, convert to tensor (per-sample)  
                # - GPU: Apply v2 transforms to entire batch at once
                if TRANSFORMS_V2_AVAILABLE:
                    test_configs.append({
                        'name': 'GPU_v2_aug_batch',
                        'dataset': SyntheticImageDataset(
                            size=dataset_size,
                            transform_fns=transforms.ToTensor(),  # Only convert to tensor on CPU
                            simulate_disk_io=simulate_disk_io
                        ),
                        'device': 'cuda',
                        'gpu_transforms': create_v2_transforms_gpu()  # Apply on GPU in batches
                    })
            
            # Run tests for each configuration
            for test_config in test_configs:
                print(f"\nTesting: {test_config['name']}")
                
                run_results = []
                for run in range(num_runs):
                    result = profile_dataloader(
                        dataset=test_config['dataset'],
                        batch_size=batch_size,
                        num_workers=num_workers,
                        device=test_config['device'],
                        augmentation_type=test_config['name'],
                        gpu_transforms=test_config['gpu_transforms'],
                        num_batches=num_batches,
                        should_simulate_forward_pass=should_simulate_forward_pass,
                        forward_pass_time_ms=forward_pass_time_ms,
                        model_complexity=model_complexity
                    )
                    run_results.append(result)
                    print(f"  Run {run+1}: {result.samples_per_second:.1f} samples/sec")
                
                                # Calculate statistics
                avg_result = ProfilingResult(
                    total_time=np.mean([r.total_time for r in run_results]),
                    total_samples=run_results[0].total_samples,
                    samples_per_second=np.mean([r.samples_per_second for r in run_results]),
                    mean_batch_time=np.mean([r.mean_batch_time for r in run_results]),
                    std_batch_time=np.mean([r.std_batch_time for r in run_results]),
                    augmentation_time=np.mean([r.augmentation_time for r in run_results]),
                    data_loading_time=np.mean([r.data_loading_time for r in run_results]),
                    transfer_time=np.mean([r.transfer_time for r in run_results]),
                    forward_pass_time=np.mean([r.forward_pass_time for r in run_results]),
                    device=test_config['device'],
                    augmentation_type=test_config['name'],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    with_forward_pass=should_simulate_forward_pass,
                    forward_pass_ms=forward_pass_time_ms if should_simulate_forward_pass else 0,
                    model_complexity=model_complexity if should_simulate_forward_pass else None
                )
                
                results['results'].append({
                    'config': config,
                    'test': test_config['name'],
                    'avg_result': asdict(avg_result),
                    'all_runs': [asdict(r) for r in run_results]
                })
    
    # Generate summary statistics
    results['summary'] = generate_summary(results)
    
    return results


def generate_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics from profiling results."""
    summary = {
        'best_configurations': {},
        'speedup_analysis': {},
        'bottleneck_analysis': {},
        'forward_pass_impact': {}
    }
    
    # Find best configuration for each augmentation type
    aug_types = set(r['test'] for r in results['results'])
    for aug_type in aug_types:
        type_results = [r for r in results['results'] if r['test'] == aug_type]
        best = max(type_results, key=lambda x: x['avg_result']['samples_per_second'])
        summary['best_configurations'][aug_type] = {
            'config': best['config'],
            'samples_per_second': best['avg_result']['samples_per_second']
        }
    
    # Calculate speedups
    baseline = [r for r in results['results'] if r['test'] == 'CPU_no_aug']
    if baseline:
        baseline_speed = np.mean([r['avg_result']['samples_per_second'] for r in baseline])
        for aug_type in aug_types:
            type_results = [r for r in results['results'] if r['test'] == aug_type]
            avg_speed = np.mean([r['avg_result']['samples_per_second'] for r in type_results])
            summary['speedup_analysis'][aug_type] = avg_speed / baseline_speed
    
    # Analyze bottlenecks
    for result in results['results']:
        key = f"{result['test']}_{result['config']['batch_size']}_{result['config']['num_workers']}"
        avg = result['avg_result']
        total_time = avg['mean_batch_time']
        
        if total_time > 0:
            summary['bottleneck_analysis'][key] = {
                'data_loading_pct': (avg['data_loading_time'] / total_time) * 100,
                'transfer_pct': (avg['transfer_time'] / total_time) * 100,
                'augmentation_pct': (avg['augmentation_time'] / total_time) * 100,
                'forward_pass_pct': (avg['forward_pass_time'] / total_time) * 100
            }
    
    # Analyze forward pass impact if enabled
    if results.get('forward_pass_enabled', False):
        for aug_type in aug_types:
            type_results = [r for r in results['results'] if r['test'] == aug_type]
            if type_results:
                avg_result = type_results[0]['avg_result']
                total_time = avg_result['mean_batch_time']
                if total_time > 0:
                    summary['forward_pass_impact'][aug_type] = {
                        'forward_pass_percentage': (avg_result['forward_pass_time'] / total_time) * 100,
                        'augmentation_percentage': (avg_result['augmentation_time'] / total_time) * 100,
                        'data_pipeline_percentage': ((avg_result['data_loading_time'] + 
                                                     avg_result['transfer_time'] + 
                                                     avg_result['augmentation_time']) / total_time) * 100
                    }
    
    return summary


def plot_results(results: Dict[str, Any], save_path: Optional[str] = None):
    """Create comprehensive visualization of profiling results."""
    
    # Prepare data
    configs = results['configurations']
    config_labels = [f"BS{c['batch_size']}_W{c['num_workers']}" for c in configs]
    
    # Group results by augmentation type
    aug_types = sorted(set(r['test'] for r in results['results']))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Throughput comparison
    ax1 = plt.subplot(2, 3, 1)
    for aug_type in aug_types:
        throughputs = []
        for config in configs:
            matching = [r for r in results['results'] 
                       if r['test'] == aug_type and r['config'] == config]
            if matching:
                throughputs.append(matching[0]['avg_result']['samples_per_second'])
            else:
                throughputs.append(0)
        ax1.plot(config_labels, throughputs, marker='o', label=aug_type)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Samples/Second')
    title = 'Throughput Comparison'
    if results.get('forward_pass_enabled', False):
        title += f"\n(with {results.get('forward_pass_ms', 0)}ms forward pass)"
    ax1.set_title(title)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Best configuration for each type
    ax2 = plt.subplot(2, 3, 2)
    if 'best_configurations' in results['summary']:
        best_configs = results['summary']['best_configurations']
        names = list(best_configs.keys())
        speeds = [best_configs[n]['samples_per_second'] for n in names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        bars = ax2.bar(range(len(names)), speeds, color=colors)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Samples/Second')
        ax2.set_title('Best Performance by Augmentation Type')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, speed in zip(bars, speeds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speed:.0f}', ha='center', va='bottom')
    
    # 3. Speedup analysis
    ax3 = plt.subplot(2, 3, 3)
    if 'speedup_analysis' in results['summary']:
        speedups = results['summary']['speedup_analysis']
        names = list(speedups.keys())
        values = list(speedups.values())
        colors = ['red' if v < 1 else 'green' for v in values]
        bars = ax3.bar(range(len(names)), values, color=colors, alpha=0.7)
        ax3.axhline(y=1, color='black', linestyle='--', label='Baseline')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right')
        ax3.set_ylabel('Speedup vs No Augmentation')
        ax3.set_title('Relative Performance')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}x', ha='center', va='bottom' if val > 0 else 'top')
    
    # 4. Batch size impact
    ax4 = plt.subplot(2, 3, 4)
    batch_sizes = sorted(set(c['batch_size'] for c in configs))
    for aug_type in aug_types:
        throughputs = []
        for bs in batch_sizes:
            matching = [r for r in results['results'] 
                       if r['test'] == aug_type and r['config']['batch_size'] == bs]
            if matching:
                avg_throughput = np.mean([r['avg_result']['samples_per_second'] for r in matching])
                throughputs.append(avg_throughput)
            else:
                throughputs.append(0)
        ax4.plot(batch_sizes, throughputs, marker='s', label=aug_type)
    
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Samples/Second')
    ax4.set_title('Impact of Batch Size')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Worker count impact
    ax5 = plt.subplot(2, 3, 5)
    worker_counts = sorted(set(c['num_workers'] for c in configs))
    for aug_type in aug_types:
        throughputs = []
        for wc in worker_counts:
            matching = [r for r in results['results'] 
                       if r['test'] == aug_type and r['config']['num_workers'] == wc]
            if matching:
                avg_throughput = np.mean([r['avg_result']['samples_per_second'] for r in matching])
                throughputs.append(avg_throughput)
            else:
                throughputs.append(0)
        ax5.plot(worker_counts, throughputs, marker='^', label=aug_type)
    
    ax5.set_xlabel('Number of Workers')
    ax5.set_ylabel('Samples/Second')
    ax5.set_title('Impact of Worker Count')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Time breakdown (for all runs grouped by augmentation type)
    ax6 = plt.subplot(2, 3, 6)
    if results['results']:
        # Group results by augmentation type
        aug_type_groups = {}
        for result in results['results']:
            aug_type = result['test']
            if aug_type not in aug_type_groups:
                aug_type_groups[aug_type] = []
            aug_type_groups[aug_type].append(result)
        
        # Take top 4 augmentation types for clarity
        top_aug_types = list(aug_type_groups.keys())[:4]
        
        components = ['Data Loading', 'Transfer', 'Augmentation', 'Forward Pass', 'Other']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffff99', '#ffcc99']
        
        x_pos = np.arange(len(top_aug_types))
        width = 0.15  # Width of bars (adjusted for 5 components)
        
        for i, component in enumerate(components):
            values = []
            for aug_type in top_aug_types:
                # Calculate average for this component across all configs for this aug type
                component_times = []
                for result in aug_type_groups[aug_type]:
                    avg = result['avg_result']
                    total_batch_time = avg['mean_batch_time']
                    if total_batch_time > 0:
                        if component == 'Data Loading':
                            component_times.append(avg['data_loading_time'])
                        elif component == 'Transfer':
                            component_times.append(avg['transfer_time'])
                        elif component == 'Augmentation':
                            component_times.append(avg['augmentation_time'])
                        elif component == 'Forward Pass':
                            component_times.append(avg.get('forward_pass_time', 0))
                        else:  # Other
                            other_time = total_batch_time - avg['data_loading_time'] - avg['transfer_time'] - avg['augmentation_time'] - avg.get('forward_pass_time', 0)
                            component_times.append(max(0, other_time))
                
                # Use mean if we have data, otherwise 0
                values.append(np.mean(component_times) if component_times else 0)
            
            ax6.bar(x_pos + i * width, values, width, label=component, color=colors[i], alpha=0.8)
        
        ax6.set_xlabel('Augmentation Type')
        ax6.set_ylabel('Time (seconds)')
        title = 'Time Breakdown by Component'
        if results.get('forward_pass_enabled', False):
            title += f"\nFP={results.get('forward_pass_ms', 0)}ms"
            if results.get('model_complexity'):
                title += f" ({results.get('model_complexity')})"
        ax6.set_title(title)
        ax6.set_xticks(x_pos + width * 2)  # Center the labels between bars
        ax6.set_xticklabels(top_aug_types, rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Data Augmentation Profiling Results', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def save_results(results: Dict[str, Any], filepath: str):
    """Save profiling results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    converted_results = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=2)
    print(f"Results saved to {filepath}")


def print_summary(results: Dict[str, Any]):
    """Print a formatted summary of the profiling results."""
    print("\n" + "="*80)
    print("PROFILING SUMMARY")
    print("="*80)
    
    # Best configurations
    if 'best_configurations' in results['summary']:
        print("\n📊 Best Configuration for Each Augmentation Type:")
        print("-"*50)
        for aug_type, info in results['summary']['best_configurations'].items():
            print(f"{aug_type:30} → {info['samples_per_second']:>8.1f} samples/sec")
            print(f"{'':30}   (BS={info['config']['batch_size']}, Workers={info['config']['num_workers']})")
    
    # Speedup analysis
    if 'speedup_analysis' in results['summary']:
        print("\n⚡ Speedup vs CPU Baseline (no augmentation):")
        print("-"*50)
        for aug_type, speedup in results['summary']['speedup_analysis'].items():
            emoji = "🚀" if speedup > 1.5 else "✓" if speedup > 1.0 else "⚠️"
            print(f"{emoji} {aug_type:30} → {speedup:>6.2f}x")
    
    # Forward pass impact analysis
    if results.get('forward_pass_enabled', False) and 'forward_pass_impact' in results['summary']:
        print("\n📈 Impact of Forward Pass (Time Distribution):")
        print("-"*50)
        print(f"Forward pass time: {results.get('forward_pass_ms', 0)}ms")
        print(f"Model complexity: {results.get('model_complexity', 'N/A')}")
        print("")
        for aug_type, impact in results['summary']['forward_pass_impact'].items():
            print(f"{aug_type:30}")
            print(f"  Forward Pass: {impact['forward_pass_percentage']:5.1f}%")
            print(f"  Augmentation: {impact['augmentation_percentage']:5.1f}%")
            print(f"  Data Pipeline: {impact['data_pipeline_percentage']:5.1f}%")
    
    # Key insights
    print("\n💡 Key Insights:")
    print("-"*50)
    
    # Find fastest overall
    if results['results']:
        fastest = max(results['results'], key=lambda x: x['avg_result']['samples_per_second'])
        print(f"Fastest: {fastest['test']} at {fastest['avg_result']['samples_per_second']:.1f} samples/sec")
        
        # Find most efficient GPU augmentation
        gpu_results = [r for r in results['results'] if 'GPU' in r['test'] and 'aug' in r['test']]
        if gpu_results:
            best_gpu = max(gpu_results, key=lambda x: x['avg_result']['samples_per_second'])
            print(f"Best GPU Aug: {best_gpu['test']} at {best_gpu['avg_result']['samples_per_second']:.1f} samples/sec")
        
        # Analyze bottleneck shift with forward pass
        if results.get('forward_pass_enabled', False):
            print("\n🔍 Bottleneck Analysis:")
            # Find where augmentation matters most/least
            aug_impact = []
            for result in results['results']:
                if 'aug' in result['test']:
                    avg = result['avg_result']
                    aug_pct = (avg['augmentation_time'] / avg['mean_batch_time']) * 100
                    aug_impact.append((result['test'], aug_pct))
            
            if aug_impact:
                aug_impact.sort(key=lambda x: x[1], reverse=True)
                print(f"Augmentation impact (% of total time):")
                for name, pct in aug_impact[:3]:
                    print(f"  {name:30} → {pct:.1f}%")
                
                # Recommendation based on forward pass time
                fp_ms = results.get('forward_pass_ms', 0)
                if fp_ms < 20:
                    print("\n⚠️  With fast forward pass (<20ms), augmentation is likely bottleneck")
                    print("    → GPU batch augmentation recommended")
                elif fp_ms > 100:
                    print("\n✓  With slow forward pass (>100ms), augmentation impact is minimal")
                    print("    → CPU augmentation may be sufficient")
                else:
                    print("\n📌  With moderate forward pass (20-100ms), profile your specific model")
                    print("    → Choice depends on augmentation complexity")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Profile different data augmentation strategies (CPU vs GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small dataset
  python profile_augmentation.py --batch-sizes 32 --num-workers 4 --dataset-size 100 --num-runs 2
  
  # Test with simulated forward pass (fast model)
  python profile_augmentation.py --simulate-forward-pass --forward-pass-time 10
  
  # Test with simulated forward pass (slow model)
  python profile_augmentation.py --simulate-forward-pass --forward-pass-time 100
  
  # Comprehensive profiling with different model speeds
  python profile_augmentation.py --batch-sizes 16 32 64 --num-workers 0 2 4 8 \\
                                  --simulate-forward-pass --forward-pass-time 50
  
  # Test specific number of batches
  python profile_augmentation.py --num-batches 50 --batch-sizes 32 64
        """
    )
    
    parser.add_argument('--should-simulate-forward-pass', action='store_true',
                       help='Simulate model forward pass to measure impact on data loading')
    parser.add_argument('--forward-pass-time', type=float, default=5.0,
                       help='Simulated forward pass time in milliseconds (default: 50ms)')
    parser.add_argument('--model-complexity', choices=['light', 'medium', 'heavy'], 
                       default='medium',
                       help='Complexity of dummy model for forward pass (default: medium)')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[64],
                       help='Batch sizes to test (default: 64)')
    parser.add_argument('--num-workers', nargs='+', type=int, default=[0, 4, 8],
                       help='Number of workers to test (default: 0 4 8)')
    parser.add_argument('--dataset-size', type=int, default=1000,
                       help='Number of samples in dataset (default: 500)')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of runs per configuration for averaging (default: 3)')
    parser.add_argument('--num-batches', type=int, default=None,
                       help='Number of batches to process (default: all)')
    parser.add_argument('--no-disk-io', action='store_true',
                       help='Skip disk I/O simulation for faster testing')
    parser.add_argument('--save-results', type=str, default='profiling_results.json',
                       help='Path to save results JSON (default: profiling_results.json)')
    parser.add_argument('--save-plot', type=str, default='profiling_results.png',
                       help='Path to save plot (default: profiling_results.png)')
    parser.add_argument('--load-results', type=str, default=None,
                       help='Load and visualize existing results JSON file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DATA AUGMENTATION PROFILING SCRIPT")
    print("="*80)
    
    # If loading existing results
    if args.load_results:
        print(f"Loading results from {args.load_results}")
        with open(args.load_results, 'r') as f:
            results = json.load(f)
        print_summary(results)
        plot_results(results, args.save_plot)
        return
    
    # Print configuration
    print("\n📋 Configuration:")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Worker counts: {args.num_workers}")
    print(f"  Dataset size: {args.dataset_size}")
    print(f"  Runs per config: {args.num_runs}")
    print(f"  Simulate disk I/O: {not args.no_disk_io}")
    print(f"  Simulate forward pass: {args.should_simulate_forward_pass}")
    if args.should_simulate_forward_pass:
        print(f"  Forward pass time: {args.forward_pass_time}ms")
        print(f"  Model complexity: {args.model_complexity}")
    if args.num_batches:
        print(f"  Batches to process: {args.num_batches}")
    
    # Check available libraries
    print("\n🔧 Available Libraries:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Torchvision: {torchvision.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Transforms v2 available: {TRANSFORMS_V2_AVAILABLE}")
    
    # Run profiling
    print("\n🚀 Starting profiling...")
    results = run_comprehensive_profiling(
        batch_sizes=args.batch_sizes,
        num_workers_list=args.num_workers,
        dataset_size=args.dataset_size,
        num_runs=args.num_runs,
        num_batches=args.num_batches,
        simulate_disk_io=not args.no_disk_io,
        should_simulate_forward_pass=args.should_simulate_forward_pass,
        forward_pass_time_ms=args.forward_pass_time,
        model_complexity=args.model_complexity
    )
    
    # Generate dynamic filenames if forward pass simulation is enabled
    if args.should_simulate_forward_pass:
        # Extract base name and extension from the default filenames
        results_base = os.path.splitext(args.save_results)[0]
        plot_base = os.path.splitext(args.save_plot)[0]
        results_ext = os.path.splitext(args.save_results)[1]
        plot_ext = os.path.splitext(args.save_plot)[1]
        
        # Create new filenames with model complexity and forward pass time
        results_file = f"{results_base}_{args.model_complexity}_fp{args.forward_pass_time}ms{results_ext}"
        plot_file = f"{plot_base}_{args.model_complexity}_fp{args.forward_pass_time}ms{plot_ext}"
        
        print(f"\n📁 Using dynamic filenames for forward pass simulation:")
        print(f"   Results: {results_file}")
        print(f"   Plot: {plot_file}")
    else:
        results_file = args.save_results
        plot_file = args.save_plot
    
    # Save results
    save_results(results, results_file)
    
    # Print summary
    print_summary(results)
    
    # Plot results
    plot_results(results, plot_file)
    
    print("\n✅ Profiling complete!")
    print(f"   Results saved to: {results_file}")
    print(f"   Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()