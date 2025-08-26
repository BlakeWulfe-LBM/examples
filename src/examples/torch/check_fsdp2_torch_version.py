# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "torch==2.4.0",
#     "torchdata<=0.8.0",
#     "pytorch-lightning",
#     "accelerate>=0.25.0",
#     "boto3[crt]==1.40.11",
#     "opensearch-py==2.3.0",
#     "torchvision",
# ]
# ///
import torch
import torchdata
from torchdata.datapipes.iter import (
    PinMemory,
    Prefetcher,
    SampleMultiplexer,
)
from torch.distributed._composable.fsdp import fully_shard 
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh 
import pytorch_lightning as pl
import accelerate
import torchvision

print(pl.__version__)
print(torch.__version__)
print(torchdata.__version__)
print(torchvision.__version__)