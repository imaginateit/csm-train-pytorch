#!/bin/bash
# Test MLX token generation quality

# Create output directory
mkdir -p token_analysis

# Run the token test script
echo "Running MLX token test with exact sampling implementation..."
python -m src.csm.cli.mlx_token_test --mode all --iterations 1000

echo "Test complete. Check the token_analysis directory for detailed results."