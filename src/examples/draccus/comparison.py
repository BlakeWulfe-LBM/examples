"""
Comparison script showing ChoiceRegistry vs PluginRegistry.

This demonstrates the key differences between the two approaches.
"""

import draccus
from dataclasses import dataclass, field

# Import ChoiceRegistry example
from examples.draccus.choice_registry.base_config import ModelConfig as ChoiceModelConfig
from examples.draccus.choice_registry.transformer_config import TransformerConfig
from examples.draccus.choice_registry.gpt_config import GptConfig

# Import PluginRegistry example
from examples.draccus.plugin_registry.base_config import ModelConfig as PluginModelConfig


@dataclass
class ChoiceTrainingConfig:
    """Training config using ChoiceRegistry."""
    
    model: ChoiceModelConfig = field(default_factory=TransformerConfig)
    batch_size: int = 64


@dataclass
class PluginTrainingConfig:
    """Training config using PluginRegistry."""
    
    model: PluginModelConfig = field(default_factory=lambda: PluginModelConfig.get_choice_class("TransformerConfig")())
    batch_size: int = 64


def demonstrate_choice_registry():
    """Demonstrate ChoiceRegistry behavior."""
    
    print("ChoiceRegistry Example")
    print("-" * 40)
    
    # Must explicitly import all subclasses
    print("1. Available choices (after explicit imports):")
    choices = ChoiceModelConfig.get_known_choices()
    for name, config_class in choices.items():
        print(f"   - {name}: {config_class}")
    
    # Parse config
    config_dict = {"model": {"type": "GptConfig", "hidden_dim": 1024}}
    config = draccus.decode(ChoiceTrainingConfig, config_dict)
    
    print(f"\n2. Parsed config: {type(config.model).__name__}")
    print(f"   Hidden dim: {config.model.hidden_dim}")
    
    # Build model
    model = config.model.build(vocab_size=50000)
    print(f"3. Built model: {model}")
    
    print()


def demonstrate_plugin_registry():
    """Demonstrate PluginRegistry behavior."""
    
    print("PluginRegistry Example")
    print("-" * 40)
    
    # Plugins discovered automatically
    print("1. Available choices (discovered automatically):")
    choices = PluginModelConfig.get_known_choices()
    for name, config_class in choices.items():
        print(f"   - {name}: {config_class}")
    
    # Parse config
    config_dict = {"model": {"type": "GptConfig", "hidden_dim": 1024}}
    config = draccus.decode(PluginTrainingConfig, config_dict)
    
    print(f"\n2. Parsed config: {type(config.model).__name__}")
    print(f"   Hidden dim: {config.model.hidden_dim}")
    
    # Build model
    model = config.model.build(vocab_size=50000)
    print(f"3. Built model: {model}")
    
    print()


def main():
    """Run both demonstrations."""
    
    print("ChoiceRegistry vs PluginRegistry Comparison")
    print("=" * 60)
    print()
    
    demonstrate_choice_registry()
    demonstrate_plugin_registry()
    
    print("Key Differences:")
    print("=" * 60)
    print("ChoiceRegistry:")
    print("  ✓ Simple and predictable")
    print("  ✓ All subclasses must be explicitly imported")
    print("  ✓ No automatic discovery")
    print("  ✓ Best for fixed, known choice sets")
    print()
    print("PluginRegistry:")
    print("  ✓ Automatic plugin discovery")
    print("  ✓ Lazy loading - only imported when needed")
    print("  ✓ Can split plugins across multiple packages")
    print("  ✓ Best for extensible plugin systems")
    print("  ✗ More complex setup")
    print("  ✗ Requires discover_packages_path configuration")


if __name__ == "__main__":
    main() 