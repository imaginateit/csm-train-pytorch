#!/bin/bash
# Multi-speaker fine-tuning example
# This example trains multiple speakers in a single run

# Create output directory
mkdir -p ./fine_tuned_models/multi_speaker

# Run multi-speaker fine-tuning
python -m csm.cli.finetune_lora_multi \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_models/multi_speaker \
  --speakers-config ./examples/speakers_config.json \
  --lora-r 8 \
  --lora-alpha 16 \
  --target-modules q_proj v_proj \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --epochs 5 \
  --save-mode lora \
  --generate-samples \
  --sample-prompt "This is a multi-speaker fine-tuning example."