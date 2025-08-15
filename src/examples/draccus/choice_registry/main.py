"""
Main script demonstrating ChoiceRegistry usage.

This shows how to use the ChoiceRegistry with subclasses defined
in separate files.
"""

import draccus
from dataclasses import dataclass, field

# Import the base config and all subclasses to ensure registration
from examples.draccus.choice_registry.base_config import ModelConfig
from examples.draccus.choice_registry.transformer_config import TransformerConfig
from examples.draccus.choice_registry.gpt_config import GptConfig


@dataclass
class TrainingConfig:
    """Main training configuration."""
    
    model: ModelConfig = field(default_factory=TransformerConfig)
    batch_size: int = 64
    learning_rate: float = 1e-4


def main():
    """Demonstrate ChoiceRegistry functionality."""
    
    # Show available choices
    print("Available model types:")
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


if __name__ == "__main__":
    main() 