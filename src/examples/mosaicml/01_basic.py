#!/usr/bin/env python3
"""
Basic MosaicML Streaming Example

This example demonstrates the core concepts of MosaicML Streaming:
1. Creating a synthetic dataset
2. Writing it to streaming format using MDSWriter
3. Reading it back using StreamingDataset
4. Basic iteration and indexing
"""

import os
import shutil
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

from streaming import MDSWriter, StreamingDataset, StreamingDataLoader


def create_synthetic_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Create a simple synthetic dataset with random features and labels."""
    print(f"Creating {num_samples} synthetic samples...")
    
    samples = []
    for i in range(num_samples):
        # Create random features (3D vector)
        features = np.random.randn(3).astype(np.float32)
        
        # Create a simple label based on feature values
        label = 1 if np.sum(features) > 0 else 0
        
        # Create metadata
        metadata = {
            'id': i,
            'timestamp': i * 1000,  # Simulate timestamps
        }
        
        sample = {
            'absolute_index': i,  # The absolute index of this sample in the dataset
            'features': features,
            'label': label,
            'metadata': metadata
        }
        samples.append(sample)
    
    return samples


def write_streaming_dataset(samples: List[Dict[str, Any]], output_dir: str):
    """Write the dataset to streaming format using MDSWriter."""
    print(f"Writing dataset to {output_dir}...")
    
    # Define the data types for each column
    columns = {
        'absolute_index': 'int',          # absolute index of the sample
        'features': 'ndarray:float32:3',  # 3D float32 array
        'label': 'int',                   # integer label
        'metadata': 'json',               # JSON metadata
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the dataset
    with MDSWriter(
        out=output_dir,
        columns=columns,
        compression='zstd:3',      # Use zstd compression
        hashes=['sha1', 'xxh3_64'],  # Hash algorithms for integrity
        size_limit=1 << 20,       # 1MB shard size limit
    ) as writer:
        for sample in tqdm(samples, desc="Writing samples"):
            writer.write(sample)
    
    print(f"Dataset written successfully to {output_dir}")


def read_streaming_dataset(remote_dir: str, local_dir: str):
    """Read the streaming dataset using StreamingDataset."""
    print(f"Reading dataset from {remote_dir} to {local_dir}...")
    
    # Create local directory for caching
    os.makedirs(local_dir, exist_ok=True)
    
    # Load the dataset
    dataset = StreamingDataset(
        remote=remote_dir,
        local=local_dir,
        shuffle=False,  # Keep deterministic order for demo
        batch_size=32,  # Optional: helps with worker distribution
    )
    
    print(f"Dataset loaded: {dataset.num_samples} samples, {len(dataset.shards)} shards")
    return dataset


def demonstrate_dataset_usage(dataset: StreamingDataset):
    """Demonstrate various ways to use the StreamingDataset."""
    print("\n--- Dataset Usage Examples ---")
    
    # 1. Basic iteration
    print("\n1. Iterating through first 5 samples:")
    for i, sample in enumerate(dataset):
        if i >= 5:
            break
        print(f"  Sample {i}: absolute_index={sample['absolute_index']}, features={sample['features'][:2]}..., label={sample['label']}")
    
    # 2. Indexing
    print("\n2. Indexing examples:")
    print(f"  Sample 10: {dataset[10]}")
    print(f"  Sample -1 (last): {dataset[-1]}")
    
    # 3. Slicing
    print("\n3. Slicing examples:")
    slice_samples = dataset[5:10]
    print(f"  Sliced samples 5-9: {len(slice_samples)} samples")
    
    # 4. Random access
    print("\n4. Random access:")
    indices = [0, 50, 100, -1]
    for idx in indices:
        sample = dataset[idx]
        print(f"  Sample {idx}: absolute_index={sample['absolute_index']}, label={sample['label']}, id={sample['metadata']['id']}")


def demonstrate_resumption(dataset: StreamingDataset):
    """Demonstrate fast resumption using StreamingDataLoader state management."""
    print("\n--- Fast Resumption Demo ---")
    print("This demonstrates how StreamingDataLoader maintains state for checkpointing and resumption.")
    
    # Create a StreamingDataLoader (this is what enables state management)
    dataloader = StreamingDataLoader(dataset, batch_size=8)
    
    print("\n1. First run - processing batches and saving state at batch 3:")
    state_dict = None
    processed_batches = []
    
    # Simulate processing some batches and then "failing" after batch 6
    for i, batch in enumerate(dataloader):
        # Process the batch
        batch_size = len(batch['absolute_index'])
        processed_batches.append({
            'batch_idx': i,
            'sample_indices': batch['absolute_index'].tolist(),
            'labels': batch['label'].tolist()
        })
        
        print(f"  Batch {i}: processed {batch_size} samples (indices: {batch['absolute_index'][:3]}...)")
        
        # Save state at batch 3 (simulating a checkpoint)
        if i == 3:
            state_dict = dataloader.state_dict()
            print(f"  ✓ Checkpoint saved at batch {i}")
        
        # "Fail" after batch 6 (simulating a crash/failure)
        if i == 6:
            print(f"  ✗ Training failed after batch {i}")
            break

    print(f"\n  Total batches processed before failure: {len(processed_batches)}")
    print(f"  Checkpoint saved at batch: 3")
    
    # Now demonstrate resumption
    print("\n2. Resuming from checkpoint:")
    
    # Create a new dataloader (simulating restart after failure)
    dataloader_resumed = StreamingDataLoader(dataset, batch_size=8)
    
    # Load the saved state
    dataloader_resumed.load_state_dict(state_dict)
    print(f"  ✓ State restored from checkpoint")
    
    # Continue processing from where we left off
    resumed_batches = []
    for i, batch in enumerate(dataloader_resumed):
        batch_size = len(batch['absolute_index'])
        resumed_batches.append({
            'batch_idx': i,
            'sample_indices': batch['absolute_index'].tolist(),
            'labels': batch['label'].tolist()
        })
        
        print(f"  Batch {i}: processed {batch_size} samples (indices: {batch['absolute_index'][:3]}...)")
        
        # Stop after a few more batches to demonstrate
        if i >= 5:
            break
    
    print(f"\n  Total batches processed after resumption: {len(resumed_batches)}")
    
    # Verify that we didn't duplicate any samples
    print("\n3. Verification - No duplicate samples:")
    all_processed_indices = []
    for batch_info in processed_batches:
        all_processed_indices.extend(batch_info['sample_indices'])
    
    all_resumed_indices = []
    for batch_info in resumed_batches:
        all_resumed_indices.extend(batch_info['sample_indices'])
    
    # Check for duplicates
    duplicates = set(all_processed_indices) & set(all_resumed_indices)
    if duplicates:
        print(f"  ⚠️  Found {len(duplicates)} duplicate samples: {sorted(duplicates)[:5]}...")
    else:
        print(f"  ✓ No duplicate samples found - resumption worked correctly!")
    
    print(f"\n  Total unique samples processed: {len(set(all_processed_indices + all_resumed_indices))}")
    
    # Show the power of stateful resumption
    print("\n4. Key Benefits of Streaming Resumption:")
    print("  • No need to re-process previous samples")
    print("  • Deterministic resumption from exact checkpoint")
    print("  • Works seamlessly with distributed training")
    print("  • Automatic state management in StreamingDataLoader")


def main():
    """Main function demonstrating the complete workflow."""
    print("=== MosaicML Streaming Basic Example ===\n")
    
    # Configuration
    output_dir = "./streaming_data"
    local_cache_dir = "./local_cache"
    
    # Clean up previous runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(local_cache_dir):
        shutil.rmtree(local_cache_dir)
    
    # Step 1: Create synthetic data
    samples = create_synthetic_data(num_samples=1000)
    print(f"Created {len(samples)} samples")
    
    # Step 2: Write to streaming format
    write_streaming_dataset(samples, output_dir)
    
    # Step 3: Read back using StreamingDataset
    dataset = read_streaming_dataset(output_dir, local_cache_dir)
    
    # Step 4: Demonstrate usage
    demonstrate_dataset_usage(dataset)
    
    # Step 5: Demonstrate fast resumption
    demonstrate_resumption(dataset)
    
    # Step 6: Show dataset properties
    print(f"\n--- Dataset Properties ---")
    print(f"Total samples: {dataset.num_samples}")
    print(f"Number of shards: {len(dataset.shards)}")
    print(f"Samples per shard: {dataset.samples_per_shard}")
    print(f"Local cache directory: {local_cache_dir}")
    
    print(f"\n=== Example completed successfully! ===")
    print(f"Dataset files are in: {output_dir}")
    print(f"Local cache is in: {local_cache_dir}")


if __name__ == "__main__":
    main() 