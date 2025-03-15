#!/bin/bash
# Resume fine-tuning from a checkpoint example

# Create output directory
mkdir -p ./fine_tuned_models/resume

# Run initial fine-tuning (short duration)
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_models/resume \
  --audio-dir ./data/speaker1/audio \
  --transcript-dir ./data/speaker1/transcripts \
  --speaker-id 1 \
  --epochs 1 \
  --save-every 10

# Resume fine-tuning from the last checkpoint
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_models/resume \
  --audio-dir ./data/speaker1/audio \
  --transcript-dir ./data/speaker1/transcripts \
  --speaker-id 1 \
  --epochs 5 \
  --resume-from ./fine_tuned_models/resume/checkpoint_latest.safetensors \
  --generate-samples \
  --sample-prompt "This example demonstrates resuming fine-tuning from a checkpoint."