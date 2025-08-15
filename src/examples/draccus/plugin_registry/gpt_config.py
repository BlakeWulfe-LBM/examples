"""
GPT model configuration for PluginRegistry.

This demonstrates how to register another subclass with PluginRegistry
in a separate file. The PluginRegistry will automatically discover
this when get_known_choices() is called.
"""

from dataclasses import dataclass
from typing import Type

from examples.draccus.plugin_registry.base_config import ModelConfig, Model


@ModelConfig.register_subclass("GptConfig")
@dataclass
class GptConfig(ModelConfig):
    """Configuration for GPT model."""
    
    mlp_scale: int = 4
    dropout: float = 0.1
    
    @property
    def model_type(cls) -> Type["GptModel"]:
        return GptModel


class GptModel(Model):
    """GPT model implementation."""
    
    def __init__(self, config: GptConfig):
        self.config = config
    
    @classmethod
    def init(cls, vocab_size: int, config: GptConfig) -> "GptModel":
        """Initialize the GPT model."""
        return cls(config)
    
    def __str__(self) -> str:
        return f"GptModel(hidden_dim={self.config.hidden_dim}, layers={self.config.num_layers}, mlp_scale={self.config.mlp_scale})" 