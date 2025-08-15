# Draccus Registry Examples

This directory contains examples demonstrating the differences between `ChoiceRegistry` and `PluginRegistry` in the Draccus library.

## Structure

```
draccus/
├── choice_registry/          # ChoiceRegistry examples
│   ├── base_config.py       # Base configuration class
│   ├── transformer_config.py # Transformer config (separate file)
│   ├── gpt_config.py        # GPT config (separate file)
│   └── main.py              # Main script
├── plugin_registry/          # PluginRegistry examples
│   ├── base_config.py       # Base configuration class
│   ├── transformer_config.py # Transformer config (separate file)
│   ├── gpt_config.py        # GPT config (separate file)
│   └── main.py              # Main script
├── comparison.py             # Side-by-side comparison
└── README.md                 # This file
```

## Key Differences

### ChoiceRegistry
- **Simple and predictable**: All subclasses must be explicitly imported
- **No automatic discovery**: You control exactly which classes are available
- **Best for**: Fixed, known choice sets in the same codebase
- **Setup**: Just inherit from `ChoiceRegistry` and use `@register_subclass`

### PluginRegistry
- **Automatic discovery**: Plugins are found automatically from specified packages
- **Lazy loading**: Only imported when `get_known_choices()` is called
- **Best for**: Extensible plugin systems across multiple packages
- **Setup**: Inherit from `PluginRegistry` with `discover_packages_path`

## Running the Examples

### ChoiceRegistry Example
```bash
cd src/examples/draccus/choice_registry
python -m examples.draccus.choice_registry.main
```

### PluginRegistry Example
```bash
cd src/examples/draccus/plugin_registry
python -m examples.draccus.plugin_registry.main
```

### Comparison
```bash
cd src/examples/draccus
python -m examples.draccus.comparison
```

## Example Output

The examples will show:
1. Available model types
2. Configuration parsing from dictionaries
3. Model building
4. Configuration encoding

## Key Learning Points

1. **ChoiceRegistry**: Must explicitly import all subclasses to ensure registration
2. **PluginRegistry**: Automatically discovers plugins without explicit imports
3. **File separation**: Both examples show how to keep subclass configs in separate files
4. **Registration naming**: Uses exact class names (e.g., "TransformerConfig" not "Transformer")

## When to Use Which

- **Use ChoiceRegistry** when you have a fixed set of choices and want simplicity
- **Use PluginRegistry** when you need extensibility and want to allow third-party plugins 