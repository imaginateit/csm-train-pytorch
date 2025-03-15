#!/bin/bash
# Simple script to test the exact MLX sampling implementation

TEXT="Hello, this is a test of the exact MLX sampling implementation."
TEMP=0.9
TOPK=50

echo "Generating audio with exact MLX sampling..."
csm-generate-mlx --text "$TEXT" --output "mlx_exact.wav" --use-exact-sampling --temperature $TEMP --topk $TOPK --debug

echo "Audio saved to mlx_exact.wav"
echo "Check audio quality to verify the exact MLX sampling implementation produces high-quality results."