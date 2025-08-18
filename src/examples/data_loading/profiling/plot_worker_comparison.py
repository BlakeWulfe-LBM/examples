#!/usr/bin/env python3
"""
Worker Count Comparison Script

This script loads all profiling results JSON files and creates a multi-axis plot
showing how different augmentation methods perform with varying worker counts.

It automatically detects files with the pattern:
- profiling_results_*.json (for forward pass simulation)
- profiling_results.json (for no forward pass simulation)

Usage:
    python plot_worker_comparison.py --results-dir ./results/ --batch-size 32
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import glob


def load_results_files(results_dir: str) -> List[Dict]:
    """Load all profiling results JSON files from the specified directory."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all JSON files that match the expected patterns
    json_files = []
    
    # Pattern 1: profiling_results_*.json (with model complexity)
    pattern1_files = list(results_path.glob("profiling_results_*.json"))
    json_files.extend(pattern1_files)
    
    # Pattern 2: profiling_results.json (no forward pass simulation)
    pattern2_files = list(results_path.glob("profiling_results.json"))
    json_files.extend(pattern2_files)
    
    # Remove duplicates
    json_files = list(set(json_files))
    
    if not json_files:
        raise FileNotFoundError(f"No profiling results JSON files found in {results_dir}")
    
    print(f"Found {len(json_files)} profiling results files:")
    for json_file in json_files:
        print(f"  - {json_file.name}")
    
    # Load each file and extract metadata
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract metadata from filename
            filename = json_file.stem
            if filename == "profiling_results":
                # No forward pass simulation
                metadata = {
                    'filename': json_file.name,
                    'model_complexity': 'none',
                    'forward_pass_time': 0.0,
                    'forward_pass_enabled': False,
                    'filepath': str(json_file)
                }
            else:
                # Forward pass simulation - extract complexity and time
                match = re.search(r'profiling_results_(\w+)_fp([\d.]+)ms', filename)
                if match:
                    model_complexity = match.group(1)
                    forward_pass_time = float(match.group(2))
                    metadata = {
                        'filename': json_file.name,
                        'model_complexity': model_complexity,
                        'forward_pass_time': forward_pass_time,
                        'forward_pass_enabled': True,
                        'filepath': str(json_file)
                    }
                else:
                    print(f"  ⚠️  Skipped {json_file.name} (doesn't match expected pattern)")
                    continue
            
            # Add metadata to the data
            data['_metadata'] = metadata
            results.append(data)
            print(f"  ✓ Loaded {json_file.name} (complexity: {metadata['model_complexity']}, fp: {metadata['forward_pass_time']}ms)")
                
        except Exception as e:
            print(f"  ❌ Error loading {json_file.name}: {e}")
    
    if not results:
        raise ValueError("No valid results files could be loaded")
    
    return results


def extract_worker_performance_data(results: List[Dict], target_batch_size: int) -> Dict:
    """Extract performance data for the specified batch size across different worker counts."""
    performance_data = {}
    
    for result in results:
        metadata = result['_metadata']
        model_complexity = metadata['model_complexity']
        forward_pass_time = metadata['forward_pass_time']
        forward_pass_enabled = metadata['forward_pass_enabled']
        
        # Create a unique key for this configuration
        if forward_pass_enabled:
            config_key = f"{model_complexity}_fp{forward_pass_time}ms"
        else:
            config_key = "no_forward_pass"
        
        # Find results matching the target batch size
        matching_results = []
        for res in result['results']:
            if res['config']['batch_size'] == target_batch_size:
                matching_results.append(res)
        
        if not matching_results:
            print(f"⚠️  No results found for batch_size={target_batch_size} in {metadata['filename']}")
            continue
        
        # Debug: print the first result structure
        if matching_results and metadata['filename'] == 'profiling_results_heavy_fp5.0ms.json':
            print(f"🔍 Debug: First result structure in {metadata['filename']}:")
            print(f"    Keys: {list(matching_results[0].keys())}")
            if 'avg_result' in matching_results[0]:
                print(f"    avg_result keys: {list(matching_results[0]['avg_result'].keys())}")
            else:
                print(f"    No avg_result found")
        
        # Store performance data for each augmentation method
        for res in matching_results:
            try:
                aug_type = res['test']
                num_workers = res['config']['num_workers']
                
                # Check if avg_result exists and has the required field
                if 'avg_result' not in res:
                    print(f"⚠️  Missing 'avg_result' in result for {aug_type} in {metadata['filename']}")
                    continue
                
                if 'samples_per_second' not in res['avg_result']:
                    print(f"⚠️  Missing 'samples_per_second' in avg_result for {aug_type} in {metadata['filename']}")
                    print(f"    Available fields: {list(res['avg_result'].keys())}")
                    continue
                
                samples_per_sec = res['avg_result']['samples_per_second']
                
                if config_key not in performance_data:
                    performance_data[config_key] = {}
                
                if aug_type not in performance_data[config_key]:
                    performance_data[config_key][aug_type] = []
                
                performance_data[config_key][aug_type].append({
                    'num_workers': num_workers,
                    'samples_per_sec': samples_per_sec,
                    'filename': metadata['filename']
                })
                
            except Exception as e:
                print(f"❌ Error processing result for {res.get('test', 'unknown')} in {metadata['filename']}: {e}")
                print(f"    Result structure: {list(res.keys())}")
                continue
    
    return performance_data


def create_worker_comparison_plot(performance_data: Dict, batch_size: int, 
                                 save_path: Optional[str] = None) -> None:
    """Create a multi-axis plot comparing worker count performance across configurations."""
    
    if not performance_data:
        print("No performance data to plot")
        return
    
    # Get unique augmentation types across all configurations
    all_aug_types = set()
    for config_data in performance_data.values():
        all_aug_types.update(config_data.keys())
    
    aug_types = sorted(list(all_aug_types))
    
    # Define the desired order for configurations
    desired_order = ['no_forward_pass', 'light_fp5.0ms', 'medium_fp5.0ms', 'heavy_fp5.0ms']
    
    # Filter and sort configs according to desired order
    configs = []
    for desired_config in desired_order:
        if desired_config in performance_data:
            configs.append(desired_config)
    
    # Add any remaining configs that weren't in the desired order
    for config_key in sorted(performance_data.keys()):
        if config_key not in configs:
            configs.append(config_key)
    
    # Force 2x2 layout for the 4 main configurations
    n_rows, n_cols = 2, 2
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()
    
    # Colors for different augmentation types - using a more distinct color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot each configuration
    for i, config_key in enumerate(configs):
        if i >= len(axes):
            break
            
        ax = axes[i]
        config_data = performance_data[config_key]
        
        # Plot each augmentation method
        for j, aug_type in enumerate(aug_types):
            if aug_type in config_data:
                data = config_data[aug_type]
                
                # Sort by number of workers
                sorted_data = sorted(data, key=lambda x: x['num_workers'])
                workers = [d['num_workers'] for d in sorted_data]
                samples_per_sec = [d['samples_per_sec'] for d in sorted_data]
                
                ax.plot(workers, samples_per_sec, marker='o', linewidth=2, 
                       markersize=6, label=aug_type, color=colors[j])
        
        # Customize subplot
        ax.set_xlabel('Number of Workers')
        ax.set_ylabel('Samples/Second')
        
        # Create title
        if config_key == "no_forward_pass":
            title = f"No Forward Pass\n(Batch Size: {batch_size})"
        else:
            # Parse config key to extract info
            match = re.search(r'(\w+)_fp([\d.]+)ms', config_key)
            if match:
                complexity = match.group(1)
                fp_time = match.group(2)
                title = f"{complexity.title()} Model\nFP: {fp_time}ms (Batch: {batch_size})"
            else:
                title = f"{config_key}\n(Batch Size: {batch_size})"
        
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to show actual worker counts
        if config_data:
            all_workers = set()
            for aug_data in config_data.values():
                all_workers.update(d['num_workers'] for d in aug_data)
            ax.set_xticks(sorted(list(all_workers)))
    
    # Hide unused subplots
    for i in range(len(configs), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Worker Count Performance Comparison (Batch Size: {batch_size})', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def print_performance_summary(performance_data: Dict, batch_size: int):
    """Print a summary of the performance data."""
    print(f"\n{'='*80}")
    print(f"WORKER COUNT PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*80}")
    
    if not performance_data:
        print("No performance data available")
        return
    
    for config_key in sorted(performance_data.keys()):
        print(f"\n📊 Configuration: {config_key}")
        print("-" * 60)
        
        config_data = performance_data[config_key]
        
        # Find best performance for each augmentation method
        for aug_type in sorted(config_data.keys()):
            data = config_data[aug_type]
            
            if not data:
                continue
            
            # Find best performance
            best = max(data, key=lambda x: x['samples_per_sec'])
            worst = min(data, key=lambda x: x['samples_per_sec'])
            
            print(f"\n  {aug_type}:")
            print(f"    Best:  {best['samples_per_sec']:>8.1f} samples/sec (workers: {best['num_workers']})")
            print(f"    Worst: {worst['samples_per_sec']:>8.1f} samples/sec (workers: {worst['num_workers']})")
            
            # Calculate improvement
            if worst['samples_per_sec'] > 0:
                improvement = (best['samples_per_sec'] / worst['samples_per_sec'] - 1) * 100
                print(f"    Improvement: {improvement:>+6.1f}%")
            
            # Show all worker count results
            sorted_data = sorted(data, key=lambda x: x['num_workers'])
            print(f"    All results:")
            for d in sorted_data:
                print(f"      Workers {d['num_workers']:2d}: {d['samples_per_sec']:>8.1f} samples/sec")


def main():
    parser = argparse.ArgumentParser(
        description='Compare worker count performance across different configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare worker performance for batch_size=32
  python plot_worker_comparison.py --batch-size 32 --results-dir ./results/
  
  # Save the comparison plot
  python plot_worker_comparison.py --batch-size 64 --results-dir ./results/ --save-plot worker_comparison.png
        """
    )
    
    parser.add_argument('--batch-size', type=int, required=True,
                       help='Batch size to analyze')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing profiling results JSON files')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save the comparison plot')
    
    args = parser.parse_args()
    
    print("="*80)
    print("WORKER COUNT COMPARISON SCRIPT")
    print("="*80)
    print(f"Target batch size: {args.batch_size}")
    print(f"Results directory: {args.results_dir}")
    
    try:
        # Load results files
        print(f"\n📁 Loading profiling results files...")
        results = load_results_files(args.results_dir)
        
        # Extract performance data for the target batch size
        print(f"\n🔍 Extracting performance data...")
        performance_data = extract_worker_performance_data(results, args.batch_size)
        
        if not performance_data:
            print(f"❌ No performance data found for batch_size={args.batch_size}")
            return
        
        # Print summary
        print_performance_summary(performance_data, args.batch_size)
        
        # Create comparison plot
        print(f"\n📊 Creating worker comparison plot...")
        create_worker_comparison_plot(performance_data, args.batch_size, args.save_plot)
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 