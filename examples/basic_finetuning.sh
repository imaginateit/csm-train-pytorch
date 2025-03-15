#!/bin/bash
# Basic LoRA fine-tuning example

# Create output directory
mkdir -p ./fine_tuned_models/basic

# Run basic fine-tuning with default parameters
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_models/basic \
  --audio-dir ./data/speaker1/audio \
  --transcript-dir ./data/speaker1/transcripts \
  --speaker-id 1 \
  --batch-size 2 \
  --epochs 5 \
  --generate-samples \
  --sample-prompt "This is a basic fine-tuning example."