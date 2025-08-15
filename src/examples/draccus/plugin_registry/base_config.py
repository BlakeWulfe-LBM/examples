"""
Base configuration class using PluginRegistry.

This demonstrates the automatic plugin discovery approach.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

import draccus


@dataclass
class ModelConfig(draccus.PluginRegistry, ABC, discover_packages_path="examples.draccus.plugin_registry"):
    """Base model configuration class using PluginRegistry."""
    
    hidden_dim: int = 768
    num_layers: int = 12
    seq_len: int = 1024
    
    @property
    @abstractmethod
    def model_type(cls) -> Type["Model"]:
        """Return the model class type."""
        pass
    
    def build(self, vocab_size: int) -> "Model":
        """Build the model instance."""
        return self.model_type.init(vocab_size, self)


class Model(ABC):
    """Abstract base model class."""
    
    @classmethod
    @abstractmethod
    def init(cls, vocab_size: int, config: ModelConfig) -> "Model":
        """Initialize the model."""
        pass 