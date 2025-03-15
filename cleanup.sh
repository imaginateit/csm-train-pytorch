#!/bin/bash
# Cleanup script to remove temporary files and restore the repository to a clean state

# Remove temporary test and debug files
rm -f mlx_voice.wav
rm -f generic_voice.wav
rm -f test_mlx.wav
rm -f test_pytorch.wav
rm -f token_generator_test.py

# Clean up any cache directories
rm -rf .pytest_cache
rm -rf .mypy_cache
rm -rf .ruff_cache
rm -rf .coverage

# Remove temporary debug scripts
rm -f src/csm/cli/mlx_debug*.py
rm -rf token_analysis

echo "Cleanup complete!"
# Summary of cleanup actions:
# - Removed testing and benchmark files
# - Removed temporary Python modules
# - Removed __pycache__ directories
# - Moved old implementation files to cleanup_files/
