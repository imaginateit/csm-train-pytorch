#!/bin/bash
# Low-resource fine-tuning example
# This example is optimized for minimal data and compute resources

# Create output directory
mkdir -p ./fine_tuned_models/low_resource

# Run low-resource fine-tuning
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_models/low_resource \
  --audio-dir ./data/limited/audio \
  --transcript-dir ./data/limited/transcripts \
  --speaker-id 0 \
  --lora-r 4 \
  --lora-alpha 8 \
  --target-modules q_proj v_proj \
  --target-layers 16 17 18 19 \
  --learning-rate 1e-4 \
  --max-seq-len 1024 \
  --batch-size 1 \
  --epochs 20 \
  --save-mode lora \
  --generate-samples \
  --sample-prompt "This example demonstrates fine-tuning with limited data and computational resources."