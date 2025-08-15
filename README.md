# Examples

This repository contains example code demonstrating various patterns and libraries.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Setup

```bash
# Install dependencies
uv sync

# Activate virtual environment
uv shell

# Run tests
uv run pytest
```

### Adding Dependencies

```bash
# Add a dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name
``` 