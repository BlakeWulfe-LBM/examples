#!/usr/bin/env python3
"""
Clean comparison of augmentation strategies: None vs CPU vs GPU
Maximum code sharing to ensure fair comparison.
"""

import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, Any, List
import argparse
from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Check for v2 transforms (GPU batch support)
try:
    import torchvision.transforms.v2 as transforms_v2
    HAS_V2 = True
except ImportError:
    HAS_V2 = False
    print("Warning: torchvision.transforms.v2 not available")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    total_time: float
    num_batches: int
    num_samples: int
    samples_per_second: float
    seconds_per_batch: float


class SimpleDataset(Dataset):
    """Simple synthetic dataset with optional CPU transforms."""
    
    def __init__(self, 
                 size: int = 1000,
                 image_shape: tuple = (3, 224, 224),
                 cpu_transform=None):
        """
        Args:
            size: Number of samples
            image_shape: Shape of images (C, H, W)
            cpu_transform: Optional transform to apply in __getitem__ (CPU worker)
        """
        self.size = size
        self.image_shape = image_shape
        self.cpu_transform = cpu_transform
        
        # Generate synthetic data (simulate images stored as numpy arrays)
        np.random.seed(42)
        self.data = []
        for _ in range(size):
            # Create fake PIL image to simulate real image loading
            arr = np.random.randint(0, 255, (image_shape[1], image_shape[2], 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            self.data.append(img)
        
        # Random labels
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        img = self.data[idx]
        
        # Apply CPU transform if provided (this happens in DataLoader workers)
        if self.cpu_transform is not None:
            img = self.cpu_transform(img)
        else:
            # Just convert to tensor if no transform
            img = transforms.ToTensor()(img)
        
        return img, self.labels[idx]


def create_model(complexity='medium', device='cuda'):
    """
    Create a CNN model for benchmarking.
    
    Args:
        complexity: 'light', 'medium', or 'heavy'
        device: Device to place model on
    """
    if complexity == 'light':
        # Very simple model - fast forward pass
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
    elif complexity == 'medium':
        # Medium model
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
    elif complexity == 'heavy':
        # Heavy model - slower forward pass
        model = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Fourth block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Classifier
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    else:
        raise ValueError(f"Unknown model complexity: {complexity}")
    
    return model.to(device)


def create_cpu_augmentation():
    """Create CPU augmentation pipeline (applied in DataLoader workers)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def create_gpu_augmentation():
    """Create GPU batch augmentation pipeline (applied after batching)."""
    if not HAS_V2:
        return None
    
    return nn.Sequential(
        transforms_v2.RandomResizedCrop(224, antialias=True),
        transforms_v2.RandomHorizontalFlip(p=0.5),
        transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    )


def benchmark_method(
    method: str,
    dataset_size: int = 100,
    batch_size: int = 32,
    num_batches: int = 10,
    num_workers: int = 0,
    model_complexity: str = 'medium',
    device: str = 'cuda'
) -> BenchmarkResult:
    """
    Benchmark a single augmentation method.
    
    Args:
        method: One of 'none', 'cpu', 'gpu'
        dataset_size: Number of samples in dataset
        batch_size: Batch size
        num_batches: Number of batches to process
        num_workers: DataLoader workers
        device: Device to use
    """
    
    # Create dataset based on method
    if method == 'none':
        # No augmentation - just ToTensor in dataset
        dataset = SimpleDataset(size=dataset_size, cpu_transform=None)
        gpu_transform = None
    elif method == 'cpu':
        # CPU augmentation in DataLoader workers
        dataset = SimpleDataset(size=dataset_size, cpu_transform=create_cpu_augmentation())
        gpu_transform = None
    elif method == 'gpu':
        # GPU batch augmentation after loading
        dataset = SimpleDataset(size=dataset_size, cpu_transform=None)
        gpu_transform = create_gpu_augmentation()
        if gpu_transform is None:
            raise RuntimeError("GPU augmentation not available (need torchvision.transforms.v2)")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create dataloader (same for all methods)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep consistent for benchmarking
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=(num_workers > 0),
        drop_last=True  # Keep batch sizes consistent
    )
    
    # Create model
    model = create_model(complexity=model_complexity, device=device)
    model.eval()
    
    # Move GPU transform to device if needed
    if gpu_transform is not None:
        gpu_transform = gpu_transform.to(device)
    
    # Warmup (process a few batches to warm up GPU)
    warmup_batches = min(3, len(loader))
    loader_iter = iter(loader)
    
    for _ in range(warmup_batches):
        images, labels = next(loader_iter)
        
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply GPU augmentation if needed
        if gpu_transform is not None:
            with torch.no_grad():
                images = gpu_transform(images)
        
        # Forward pass
        with torch.no_grad():
            _ = model(images)
    
    # Clear cache
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Actual benchmark
    num_samples_processed = 0
    num_batches_to_run = min(num_batches, len(loader) - warmup_batches)
    
    # Ensure GPU is ready
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx < warmup_batches:  # Skip warmup batches
            continue
        if batch_idx >= warmup_batches + num_batches_to_run:
            break
        
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply GPU augmentation if needed (ONLY for gpu method)
        if gpu_transform is not None and method == 'gpu':
            with torch.no_grad():
                images = gpu_transform(images)
        
        # Forward pass through model
        with torch.no_grad():
            outputs = model(images)
        
        # Ensure computation is done
        if device == 'cuda':
            torch.cuda.synchronize()
        
        num_samples_processed += images.size(0)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    return BenchmarkResult(
        method=method,
        total_time=total_time,
        num_batches=num_batches_to_run,
        num_samples=num_samples_processed,
        samples_per_second=num_samples_processed / total_time if total_time > 0 else 0,
        seconds_per_batch=total_time / num_batches_to_run if num_batches_to_run > 0 else 0
    )


def run_worker_sweep(
    worker_counts: List[int],
    dataset_size: int = 1000,
    batch_size: int = 32,
    num_batches: int = 20,
    model_complexity: str = 'medium',
    num_runs: int = 3,
    plot_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run benchmarks across different worker counts and create plots.
    
    Args:
        worker_counts: List of worker counts to test
        dataset_size: Dataset size
        batch_size: Batch size
        num_batches: Number of batches to process
        model_complexity: Model complexity ('light', 'medium', 'heavy')
        num_runs: Number of runs per configuration
        plot_path: Optional path to save plot
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    print(f"Model complexity: {model_complexity}")
    
    methods = ['none', 'cpu']
    if HAS_V2 and device == 'cuda':
        methods.append('gpu')
    else:
        print("Skipping GPU augmentation (not available)")
    
    # Results storage: method -> worker_count -> performance
    results = {method: {} for method in methods}
    
    for num_workers in worker_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {num_workers} workers")
        print(f"{'='*60}")
        
        for method in methods:
            print(f"\nMethod: {method}")
            
            run_results = []
            for run in range(num_runs):
                result = benchmark_method(
                    method=method,
                    dataset_size=dataset_size,
                    batch_size=batch_size,
                    num_batches=num_batches,
                    num_workers=num_workers,
                    model_complexity=model_complexity,
                    device=device
                )
                run_results.append(result.samples_per_second)
                print(f"  Run {run + 1}: {result.samples_per_second:.1f} samples/sec")
            
            avg_performance = np.mean(run_results)
            std_performance = np.std(run_results)
            results[method][num_workers] = {
                'mean': avg_performance,
                'std': std_performance,
                'runs': run_results
            }
            print(f"  Average: {avg_performance:.1f} ± {std_performance:.1f} samples/sec")
    
    # Create plot
    create_worker_plot(results, worker_counts, model_complexity, plot_path)
    
    return results


def create_worker_plot(results: Dict[str, Any], worker_counts: List[int], 
                       model_complexity: str, plot_path: Optional[str] = None):
    """Create a plot of performance vs number of workers."""
    
    plt.figure(figsize=(10, 6))
    
    # Colors for different methods
    colors = {
        'none': 'blue',
        'cpu': 'red',
        'gpu': 'green'
    }
    
    markers = {
        'none': 'o',
        'cpu': 's',
        'gpu': '^'
    }
    
    # Plot each method
    for method, color in colors.items():
        if method in results:
            x = sorted(worker_counts)
            y_mean = [results[method][w]['mean'] for w in x if w in results[method]]
            y_std = [results[method][w]['std'] for w in x if w in results[method]]
            
            if len(y_mean) == len(x):
                plt.errorbar(x, y_mean, yerr=y_std, 
                           color=color, marker=markers[method], 
                           markersize=8, linewidth=2, 
                           label=f'{method.capitalize()} augmentation',
                           capsize=5, capthick=2)
    
    plt.xlabel('Number of Workers', fontsize=12)
    plt.ylabel('Samples per Second', fontsize=12)
    plt.title(f'Data Loading Performance vs Number of Workers\n(Model: {model_complexity})', fontsize=14)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all worker counts
    plt.xticks(worker_counts)
    
    # Add minor grid
    plt.grid(True, which='minor', alpha=0.1)
    plt.minorticks_on()
    
    plt.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
    else:
        plt.show()


def run_comparison(
    dataset_size: int = 1000,
    batch_size: int = 32,
    num_batches: int = 20,
    num_workers: int = 0,
    model_complexity: str = 'medium',
    num_runs: int = 3
) -> Dict[str, Any]:
    """Run comparison of all methods."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    methods = ['none', 'cpu']
    if HAS_V2 and device == 'cuda':
        methods.append('gpu')
    else:
        print("Skipping GPU augmentation (not available)")
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing method: {method}")
        print(f"{'='*50}")
        
        method_results = []
        
        for run in range(num_runs):
            result = benchmark_method(
                method=method,
                dataset_size=dataset_size,
                batch_size=batch_size,
                num_batches=num_batches,
                num_workers=num_workers,
                model_complexity=model_complexity,
                device=device
            )
            
            method_results.append(result)
            print(f"Run {run + 1}: {result.samples_per_second:.1f} samples/sec, "
                  f"{result.seconds_per_batch*1000:.2f} ms/batch")
        
        # Average results
        avg_samples_per_sec = np.mean([r.samples_per_second for r in method_results])
        avg_ms_per_batch = np.mean([r.seconds_per_batch * 1000 for r in method_results])
        
        results[method] = {
            'runs': method_results,
            'avg_samples_per_sec': avg_samples_per_sec,
            'avg_ms_per_batch': avg_ms_per_batch
        }
        
        print(f"Average: {avg_samples_per_sec:.1f} samples/sec, {avg_ms_per_batch:.2f} ms/batch")
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print summary of results."""
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Performance comparison
    print("\nPerformance (samples/sec):")
    print("-" * 40)
    
    baseline = results.get('none', {}).get('avg_samples_per_sec', 0)
    
    for method in ['none', 'cpu', 'gpu']:
        if method in results:
            perf = results[method]['avg_samples_per_sec']
            speedup = perf / baseline if baseline > 0 else 0
            print(f"{method:10} {perf:8.1f} ({speedup:.2f}x)")
    
    # Check for the bug
    print("\n" + "="*60)
    print("BUG CHECK")
    print("="*60)
    
    if 'none' in results and 'gpu' in results:
        none_perf = results['none']['avg_samples_per_sec']
        gpu_perf = results['gpu']['avg_samples_per_sec']
        
        if gpu_perf > none_perf:
            print(f"❌ BUG DETECTED: GPU augmentation ({gpu_perf:.1f}) > No augmentation ({none_perf:.1f})")
            print("   This is unexpected - adding GPU work should not make it faster!")
        else:
            print(f"✅ Results look correct: No augmentation ({none_perf:.1f}) >= GPU augmentation ({gpu_perf:.1f})")


def main():
    parser = argparse.ArgumentParser(description='Compare augmentation strategies')
    parser.add_argument('--dataset-size', type=int, default=1000, help='Dataset size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-batches', type=int, default=20, help='Number of batches to process')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (single value)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs per method')
    parser.add_argument('--model-complexity', choices=['light', 'medium', 'heavy'], 
                       default='medium', help='Model complexity')
    parser.add_argument('--worker-sweep', action='store_true', 
                       help='Run sweep across multiple worker counts')
    parser.add_argument('--worker-counts', type=int, nargs='+', 
                       default=[0, 1, 2, 4, 8], 
                       help='Worker counts to test in sweep mode')
    parser.add_argument('--plot-path', type=str, default=None,
                       help='Path to save plot (if worker-sweep enabled)')
    
    args = parser.parse_args()
    
    print("Clean Augmentation Strategy Comparison")
    print("="*60)
    print(f"Configuration:")
    print(f"  Dataset size: {args.dataset_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num batches: {args.num_batches}")
    print(f"  Model complexity: {args.model_complexity}")
    print(f"  Num runs: {args.num_runs}")
    print(f"  PyTorch: {torch.__version__}")
    
    if args.worker_sweep:
        print(f"  Worker sweep mode: Testing {args.worker_counts}")
        if args.plot_path:
            print(f"  Plot will be saved to: {args.plot_path}")
        
        results = run_worker_sweep(
            worker_counts=args.worker_counts,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            model_complexity=args.model_complexity,
            num_runs=args.num_runs,
            plot_path=args.plot_path
        )
        
        # Print summary table
        print("\n" + "="*70)
        print("SUMMARY TABLE (samples/sec)")
        print("="*70)
        print(f"{'Workers':<10}", end='')
        for method in ['none', 'cpu', 'gpu']:
            if method in results:
                print(f"{method.capitalize():<20}", end='')
        print()
        print("-"*70)
        
        for num_workers in sorted(args.worker_counts):
            print(f"{num_workers:<10}", end='')
            for method in ['none', 'cpu', 'gpu']:
                if method in results and num_workers in results[method]:
                    mean = results[method][num_workers]['mean']
                    std = results[method][num_workers]['std']
                    print(f"{mean:>8.1f} ± {std:<6.1f}  ", end='')
            print()
        
    else:
        print(f"  Num workers: {args.num_workers}")
        
        results = run_comparison(
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            num_workers=args.num_workers,
            model_complexity=args.model_complexity,
            num_runs=args.num_runs
        )
        
        print_summary(results)


if __name__ == '__main__':
    main()