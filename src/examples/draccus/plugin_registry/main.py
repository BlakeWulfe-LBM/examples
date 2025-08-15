"""
Main script demonstrating PluginRegistry usage.

This shows how to use the PluginRegistry with automatic plugin discovery.
Note that we don't need to explicitly import the subclasses - they're
discovered automatically!
"""

import draccus
from dataclasses import dataclass, field

# Only import the base config - subclasses are discovered automatically
from examples.draccus.plugin_registry.base_config import ModelConfig


@dataclass
class TrainingConfig:
    """Main training configuration."""
    
    model: ModelConfig = field(default_factory=lambda: ModelConfig.get_choice_class("TransformerConfig")())
    batch_size: int = 64
    learning_rate: float = 1e-4


def main():
    """Demonstrate PluginRegistry functionality."""
    
    print("PluginRegistry Example")
    print("=" * 50)
    
    # Show available choices (this triggers plugin discovery)
    print("Available model types (discovered automatically):")
    for name, config_class in ModelConfig.get_known_choices().items():
        print(f"  - {name}: {config_class}")
    
    # Create config from YAML-like dict
    config_dict = {
        "model": {
            "type": "GptConfig",
            "hidden_dim": 1024,
            "num_layers": 24,
            "mlp_scale": 4
        },
        "batch_size": 128,
        "learning_rate": 5e-5
    }
    
    # Parse the configuration
    config = draccus.decode(TrainingConfig, config_dict)
    
    print(f"\nParsed config:")
    print(f"  Model type: {type(config.model).__name__}")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Num layers: {config.model.num_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Build the model
    model = config.model.build(vocab_size=50000)
    print(f"\nBuilt model: {model}")
    
    # Show the encoded configuration
    print(f"\nEncoded config:")
    print(draccus.dump(config))
    
    print("\n" + "=" * 50)
    print("Key difference from ChoiceRegistry:")
    print("- No need to explicitly import subclasses")
    print("- Plugins are discovered automatically")
    print("- Lazy loading - only imported when needed")


if __name__ == "__main__":
    main() 