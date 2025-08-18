#!/usr/bin/env python3
"""
MosaicML Streaming with PIL Images Example

This example demonstrates working with images in MosaicML Streaming:
1. Creating synthetic image datasets with multiple images per sample
2. Writing PIL images to streaming format using MDSWriter
3. Reading and processing image data using StreamingDataset
4. Handling multiple images per sample
"""

import os
import shutil
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from streaming import MDSWriter, StreamingDataset, StreamingDataLoader


def create_synthetic_image(width: int = 64, height: int = 64, text: str = "A") -> Image.Image:
    """Create a synthetic image with text for demonstration purposes."""
    # Create a random colored background
    background_color = tuple(np.random.randint(0, 256, 3))
    
    # Create image with random background
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), text, font=font) if font else (0, 0, 20, 20)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text in contrasting color
    text_color = tuple(255 - c for c in background_color)  # Inverted colors
    draw.text((x, y), text, fill=text_color, font=font)
    
    return image


def create_synthetic_image_data(num_samples: int = 100, images_per_sample: int = 3) -> List[Dict[str, Any]]:
    """Create synthetic image dataset with multiple images per sample."""
    print(f"Creating {num_samples} samples with {images_per_sample} images each...")
    
    samples = []
    for i in range(num_samples):
        # Create multiple images for this sample
        images = []
        for j in range(images_per_sample):
            # Create image with sample ID and image index
            img = create_synthetic_image(
                width=64, 
                height=64, 
                text=f"{i}-{j}"
            )
            images.append(img)
        
        # Create a simple label based on sample ID
        label = i % 10  # 10 classes (0-9)
        
        # Create metadata
        metadata = {
            'id': i,
            'timestamp': i * 1000,
            'num_images': images_per_sample,
            'image_sizes': [(img.width, img.height) for img in images]
        }
        
        sample = {
            'absolute_index': i,
            'images': images,  # List of PIL images
            'label': label,
            'metadata': metadata
        }
        samples.append(sample)
    
    return samples


def write_image_streaming_dataset(samples: List[Dict[str, Any]], output_dir: str):
    """Write the image dataset to streaming format using MDSWriter."""
    print(f"Writing image dataset to {output_dir}...")
    
    # Define the data types for each column
    columns = {
        'absolute_index': 'int',          # absolute index of the sample
        'images': 'list[pil]',            # List of PIL images (MosaicML handles PIL format)
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
        for sample in tqdm(samples, desc="Writing image samples"):
            writer.write(sample)
    
    print(f"Image dataset written successfully to {output_dir}")


def read_image_streaming_dataset(remote_dir: str, local_dir: str):
    """Read the image streaming dataset using StreamingDataset."""
    print(f"Reading image dataset from {remote_dir} to {local_dir}...")
    
    # Create local directory for caching
    os.makedirs(local_dir, exist_ok=True)
    
    # Load the dataset
    dataset = StreamingDataset(
        remote=remote_dir,
        local=local_dir,
        shuffle=False,  # Keep deterministic order for demo
        batch_size=16,  # Smaller batch size for images
    )
    
    print(f"Image dataset loaded: {dataset.num_samples} samples, {len(dataset.shards)} shards")
    return dataset


def demonstrate_image_dataset_usage(dataset: StreamingDataset):
    """Demonstrate various ways to use the image StreamingDataset."""
    print("\n--- Image Dataset Usage Examples ---")
    
    # 1. Basic iteration with image inspection
    print("\n1. Iterating through first 3 samples:")
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        
        images = sample['images']
        print(f"  Sample {i}: absolute_index={sample['absolute_index']}, "
              f"label={sample['label']}, num_images={len(images)}")
        
        # Show image properties
        for j, img in enumerate(images):
            print(f"    Image {j}: size={img.size}, mode={img.mode}")
    
    # 2. Indexing specific samples
    print("\n2. Indexing specific samples:")
    sample_10 = dataset[10]
    print(f"  Sample 10: {len(sample_10['images'])} images, label={sample_10['label']}")
    
    # 3. Slicing
    print("\n3. Slicing examples:")
    slice_samples = dataset[5:8]
    print(f"  Sliced samples 5-7: {len(slice_samples)} samples")
    
    # 4. Random access
    print("\n4. Random access:")
    indices = [0, 25, 50, -1]
    for idx in indices:
        sample = dataset[idx]
        print(f"  Sample {idx}: absolute_index={sample['absolute_index']}, "
              f"label={sample['label']}, num_images={len(sample['images'])}")


def demonstrate_image_processing(dataset: StreamingDataset):
    """Demonstrate processing images from the dataset."""
    print("\n--- Image Processing Demo ---")
    
    # Get a sample
    sample = dataset[0]
    images = sample['images']
    
    print(f"Processing sample 0 with {len(images)} images:")
    
    # Process each image
    for i, img in enumerate(images):
        print(f"  Image {i}:")
        print(f"    Original size: {img.size}")
        print(f"    Mode: {img.mode}")
        print(f"    Format: {img.format}")
        
        # Convert to numpy array for analysis
        img_array = np.array(img)
        print(f"    Array shape: {img_array.shape}")
        print(f"    Value range: {img_array.min()} to {img_array.max()}")
        
        # Show some pixel values
        print(f"    Center pixel: {img_array[32, 32]}")
    
    # Demonstrate batch processing
    print(f"\nBatch processing example:")
    batch_samples = dataset[0:4]  # Get 4 samples
    
    total_images = sum(len(sample['images']) for sample in batch_samples)
    print(f"  Total images in batch: {total_images}")
    
    # Collect all images from batch
    all_images = []
    all_labels = []
    for sample in batch_samples:
        all_images.extend(sample['images'])
        all_labels.extend([sample['label']] * len(sample['images']))
    
    print(f"  Flattened batch: {len(all_images)} images, {len(all_labels)} labels")


def demonstrate_dataset_extension(output_dir: str):
    """Demonstrate creating an extended dataset with more samples."""
    print("\n--- Dataset Extension Demo ---")
    
    # Create 5 new samples with 2 images each
    new_samples = create_synthetic_image_data(num_samples=5, images_per_sample=2)
    
    # Update absolute_index to continue from where we left off
    existing_samples = 100  # From the original dataset
    for i, sample in enumerate(new_samples):
        sample['absolute_index'] = existing_samples + i
    
    print(f"Creating extended dataset with {len(new_samples)} additional samples...")
    
    # Create a new extended dataset (this will overwrite the original)
    extended_samples = create_synthetic_image_data(num_samples=105, images_per_sample=3)
    
    # Write the extended dataset
    columns = {
        'absolute_index': 'int',
        'images': 'list[pil]',
        'label': 'int',
        'metadata': 'json',
    }
    
    with MDSWriter(
        out=output_dir,
        columns=columns,
        compression='zstd:3',
        hashes=['sha1', 'xxh3_64'],
        size_limit=1 << 20,
        exist_ok=True  # This will overwrite the existing dataset
    ) as writer:
        for sample in tqdm(extended_samples, desc="Writing extended dataset"):
            writer.write(sample)
    
    print(f"Successfully created extended dataset with {len(extended_samples)} total samples")
    print("Note: MosaicML Streaming doesn't support appending to existing datasets.")
    print("Instead, you need to recreate the entire dataset with the new samples.")


def main():
    """Main function demonstrating the complete image workflow."""
    print("=== MosaicML Streaming with PIL Images Example ===\n")
    
    # Configuration
    output_dir = "./streaming_images"
    local_cache_dir = "./local_image_cache"
    
    # Clean up previous runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(local_cache_dir):
        shutil.rmtree(local_cache_dir)
    
    # Step 1: Create synthetic image data
    samples = create_synthetic_image_data(num_samples=100, images_per_sample=3)
    print(f"Created {len(samples)} samples with {len(samples[0]['images'])} images each")
    
    # Step 2: Write to streaming format
    write_image_streaming_dataset(samples, output_dir)
    
    # Step 3: Read back using StreamingDataset
    dataset = read_image_streaming_dataset(output_dir, local_cache_dir)
    
    # Step 4: Demonstrate usage
    demonstrate_image_dataset_usage(dataset)
    
    # Step 5: Demonstrate image processing
    demonstrate_image_processing(dataset)

if __name__ == "__main__":
    main() 