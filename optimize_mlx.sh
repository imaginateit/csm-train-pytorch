#!/bin/bash
# Apply MLX optimizations for faster token generation

# Set helpful environment variables
export MLX_AUTOTUNE=1  # Enable autotuning

# Run the optimization script
echo "Applying MLX optimizations..."
python -m src.csm.cli.apply_mlx_optimizations --benchmark

# Make the script executable if it isn't already
chmod +x src/csm/cli/apply_mlx_optimizations.py

echo "Optimizations applied. To use in your code:"
echo "  from csm.cli.apply_mlx_optimizations import apply_general_mlx_optimizations"
echo "  apply_general_mlx_optimizations()"
echo ""
echo "To run a benchmark of the optimizations:"
echo "  ./optimize_mlx.sh"