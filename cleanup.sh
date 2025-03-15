#!/bin/bash
# Cleanup script to remove temporary files and restore the repository to a clean state

# Remove any temporary audio and image files
rm -f *.wav
rm -f *.png
rm -f *.jpg
rm -f *.jpeg
rm -f *.mp3
rm -f *.mp4

# Remove temporary test and debug files
rm -f token_generator_test.py
rm -f test_exact_sampling.sh
rm -f test_tokens.sh
rm -f compare_sampling.sh

# Clean up any cache directories
rm -rf __pycache__
rm -rf .pytest_cache
rm -rf .mypy_cache
rm -rf .ruff_cache
rm -rf .coverage
rm -rf token_analysis

# Clean any temporary files
rm -rf temp_files/*
mkdir -p temp_files

# Remove Python cache directories
find . -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

echo "Cleanup complete!"

# Summary of cleanup actions:
# - Removed testing and benchmark files
# - Removed temporary Python modules
# - Removed media files (audio, images)
# - Removed __pycache__ directories
# - Cleared temporary directories
