"""Example demonstrating WebDataset sharding using SimpleShardList with a class-based approach.

Run with (for example): 

uv run torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=29500 \
    src/examples/webdatasets/sharding.py \
    --num-shards 20 \
    --batch-size 8
"""

import webdataset as wds
import click
import os
import signal
import sys
import torch
import torch.distributed as dist
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ShardingConfig:
    """Configuration for sharding operations."""
    shuffle_buffer_size: int = 1000
    shuffle_initial: int = 100
    seed: int = 42
    batch_size: int = 32
    max_length: int = 512
    num_shards: int = 10


class SimpleShardingPipeline:
    """A simple WebDataset pipeline that demonstrates sharding with SimpleShardList."""
    
    def __init__(self, config: ShardingConfig):
        self.config = config
    
    def create_pipeline(self, shard_urls, checkpoint_num: int = 0) -> List:
        pipeline = [
            wds.SimpleShardList(shard_urls),
            wds.split_by_node,
            wds.split_by_worker,
        ]
        return pipeline
    
    def create_dataset(self, shard_urls, checkpoint_num: int = 0) -> wds.WebDataset:
        pipeline_stages = self.create_pipeline(shard_urls, checkpoint_num)
        return wds.DataPipeline(*pipeline_stages)


def setup_distributed():
    """Initialize distributed training if torchrun is used."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        print(f"Initializing distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl", 
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}"),
            init_method="env://",
        )
        print("before barrier")
        dist.barrier()
        print("after barrier")
        
        return rank, world_size, local_rank
    else:
        print("Running in single-process mode (use torchrun for distributed training)")
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\nReceived signal {signum}, cleaning up...")
    cleanup_distributed()
    sys.exit(0)


@click.command()
@click.option('--num-shards', default=10, help='Number of shards to generate')
@click.option('--batch-size', default=16, help='Batch size for processing')
def main(num_shards, batch_size):
    """Demonstrate WebDataset sharding with configurable distributed setup."""
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize distributed training if using torchrun
        rank, world_size, local_rank = setup_distributed()
        print(f"Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")
        
        # Generate shard URLs using list comprehension
        example_shards = [
            f"https://example.com/shards/{i:06d}.tar" 
            for i in range(num_shards)
        ]
        
        # Create configuration
        config = ShardingConfig(
            shuffle_buffer_size=1000,
            shuffle_initial=100,
            seed=42,
            batch_size=batch_size,
            max_length=256,
            num_shards=num_shards
        )
        
        # Create pipeline instance
        pipeline = SimpleShardingPipeline(config)
        
        # Create dataset
        dataset = pipeline.create_dataset(example_shards, checkpoint_num=0)
        
        if rank == 0:  # Only print from main process
            print("WebDataset pipeline created successfully!")
            print(f"Configuration: {config}")
            print(f"Number of shards: {len(example_shards)}")
            print(f"Distributed setup: {world_size} total processes")
        
        # Demonstrate pipeline stages
        if rank == 0:
            print("\nPipeline stages:")
            pipeline_stages = pipeline.create_pipeline(example_shards, checkpoint_num=0)
            for i, stage in enumerate(pipeline_stages):
                print(f"  {i}: {type(stage).__name__}")
        
        # Show distributed info
        if rank == 0:
            print(f"\nProcess info: rank={rank}, world_size={world_size}, local_rank={local_rank}")
            if "RANK" in os.environ:
                print("Environment variables:")
                for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"]:
                    print(f"  {key}: {os.environ.get(key, 'Not set')}")
        
        # Iterate through the dataset to see what shards are loaded
        if rank == 0:
            print("\nIterating through dataset to show shard information:")
            try:
                for i, sample in enumerate(dataset):
                    print(f"Sample {i}: {sample}")
                    # Only show first few samples to avoid overwhelming output
                    if i >= 5:
                        print("... (showing first 5 samples)")
                        break
            except Exception as e:
                print(f"Error iterating through dataset: {e}")
                print("This is expected for example URLs that don't exist")
        
        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()
        
        if rank == 0:
            print("\nPipeline demonstration complete!")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        if rank == 0:
            print("Pipeline demonstration failed!")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main() 