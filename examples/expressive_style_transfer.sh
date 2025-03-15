#!/bin/bash
# Expressive style transfer fine-tuning example
# This example focuses on capturing expressive style elements by using
# more target modules and higher rank

# Create output directory
mkdir -p ./fine_tuned_models/expressive_style

# Run expressive style fine-tuning
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_models/expressive_style \
  --audio-dir ./data/expressive/audio \
  --transcript-dir ./data/expressive/transcripts \
  --alignment-dir ./data/expressive/alignments \
  --speaker-id 0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --target-modules q_proj k_proj v_proj o_proj \
  --learning-rate 3e-5 \
  --semantic-weight 150.0 \
  --acoustic-weight 2.0 \
  --batch-size 2 \
  --epochs 15 \
  --save-mode both \
  --generate-samples \
  --sample-prompt "This example demonstrates expressive style transfer with higher emotion and dynamic range."