"""
Transformer model configuration for PluginRegistry.

This demonstrates how to register a subclass with PluginRegistry
in a separate file. The PluginRegistry will automatically discover
this when get_known_choices() is called.
"""

from dataclasses import dataclass
from typing import Type

from examples.draccus.plugin_registry.base_config import ModelConfig, Model


@ModelConfig.register_subclass("TransformerConfig")
@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for Transformer model."""
    
    num_heads: int = 12
    mlp_ratio: float = 4.0
    
    @property
    def model_type(cls) -> Type["TransformerModel"]:
        return TransformerModel


class TransformerModel(Model):
    """Transformer model implementation."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
    
    @classmethod
    def init(cls, vocab_size: int, config: TransformerConfig) -> "TransformerModel":
        """Initialize the transformer model."""
        return cls(config)
    
    def __str__(self) -> str:
        return f"TransformerModel(hidden_dim={self.config.hidden_dim}, layers={self.config.num_layers}, heads={self.config.num_heads})" 