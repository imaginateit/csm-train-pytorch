#!/bin/bash
# Script to run token generation tests comparing PyTorch and MLX implementations

# Directory for analysis results
OUTDIR="token_analysis"
mkdir -p $OUTDIR

# Text to test with
TEXT="Hello, this is a test of the Conversational Speech Model."

# Make sure we have required packages
pip install matplotlib numpy

# Run direct sampling comparison test
echo "===== Running direct sampling comparison test ====="
python -m src.csm.cli.test_token_generation --test-sampling --save-dir $OUTDIR

# Run hybrid implementation test
echo ""
echo "===== Running hybrid implementation test ====="
python -m src.csm.cli.test_token_generation --text "$TEXT" --save-dir $OUTDIR --debug

# Run exact implementation test
echo ""
echo "===== Running exact implementation test ====="
python -m src.csm.cli.test_token_generation --text "$TEXT" --save-dir $OUTDIR --use-exact --debug

# List generated files
echo ""
echo "===== Generated analysis files ====="
ls -la $OUTDIR

echo ""
echo "Analysis complete. Results saved to $OUTDIR/"