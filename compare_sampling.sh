#!/bin/bash
# Script to compare all three sampling approaches

# Set constants
TEXT="Hello, this is a test of different sampling methods for the Conversational Speech Model."
SEED=42
TEMPERATURE=0.9
TOPK=50

echo "=== Comparing MLX Sampling Methods ==="
echo ""
echo "Text: $TEXT"
echo "Seed: $SEED"
echo "Temperature: $TEMPERATURE"
echo "Top-k: $TOPK"
echo ""

# Generate with pure MLX sampling
echo "=== Generating with pure MLX sampling ==="
csm-generate-mlx --text "$TEXT" --output "mlx_pure.wav" --seed $SEED --temperature $TEMPERATURE --topk $TOPK --debug

# Generate with exact MLX sampling
echo ""
echo "=== Generating with exact MLX sampling ==="
csm-generate-mlx --text "$TEXT" --output "mlx_exact.wav" --use-exact-sampling --seed $SEED --temperature $TEMPERATURE --topk $TOPK --debug

# Generate with PyTorch token generation (hybrid mode)
echo ""
echo "=== Generating with PyTorch token generation (hybrid mode) ==="
csm-generate-mlx --text "$TEXT" --output "mlx_hybrid.wav" --pytorch-tokens --seed $SEED --temperature $TEMPERATURE --topk $TOPK --debug

# Compare file sizes
echo ""
echo "=== Audio File Comparison ==="
ls -lh mlx_pure.wav mlx_exact.wav mlx_hybrid.wav

# Play the audio files if macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "Press Enter to play pure MLX audio"
    read
    afplay mlx_pure.wav
    
    echo "Press Enter to play exact MLX audio"
    read
    afplay mlx_exact.wav
    
    echo "Press Enter to play hybrid audio"
    read
    afplay mlx_hybrid.wav
fi

echo ""
echo "Done! Compare the audio quality between the three approaches."