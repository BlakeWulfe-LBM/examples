# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "hydra-core",
#     "omegaconf",
# ]
# ///

from omegaconf import OmegaConf
from hydra.utils import instantiate
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_type: str = "linear"
    input_size: int = 784
    hidden_size: int = 128
    output_size: int = 10
    learning_rate: float = 0.001


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    model: ModelConfig = field(default_factory=ModelConfig)


class SimpleModel:
    def __init__(self, model_type: str, input_size: int, hidden_size: int, output_size: int):
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, x):
        print(f"Forward pass through {self.model_type} model: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
        return x
    
    def get_params(self):
        return f"Model: {self.model_type}, Params: {self.input_size * self.hidden_size + self.hidden_size * self.output_size}"


class Trainer:
    def __init__(self, batch_size: int, epochs: int, model: SimpleModel):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model
    
    def train(self):
        print(f"Training for {self.epochs} epochs with batch size {self.batch_size}")
        self.model.forward("sample_input")


def main():
    # Example 1: Create config with _target_ field
    model_config = OmegaConf.create({
        "_target_": "__main__.SimpleModel",
        "model_type": "transformer",
        "input_size": 512,
        "hidden_size": 256,
        "output_size": 100
    })
    
    # Example 1: Instantiate classes directly using Hydra
    print("=== Example 1: Model with _target_ in config ===")
    model1 = instantiate(model_config)
    model1.forward("test_input")
    print(f"Model params: {model1.get_params()}")

if __name__ == "__main__":
    main()
