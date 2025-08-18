#!/bin/bash

echo "Running light model profiling..."
uv run python src/examples/data_loading/profiling/profile_data_loading_2.py --model-complexity=light --should-simulate-forward-pass

echo "Running medium model profiling..."
uv run python src/examples/data_loading/profiling/profile_data_loading_2.py --model-complexity=medium --should-simulate-forward-pass

echo "Running heavy model profiling..."
uv run python src/examples/data_loading/profiling/profile_data_loading_2.py --model-complexity=heavy --should-simulate-forward-pass

echo "Running baseline profiling (no model)..."
uv run python src/examples/data_loading/profiling/profile_data_loading_2.py

echo "All profiling complete!" 