#!/bin/bash
# Advanced LoRA fine-tuning with more control over parameters

# Create output directory
mkdir -p ./fine_tuned_models/advanced

# Run advanced fine-tuning with custom parameters
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_models/advanced \
  --audio-dir ./data/speaker1/audio \
  --transcript-dir ./data/speaker1/transcripts \
  --alignment-dir ./data/speaker1/alignments \
  --speaker-id 1 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules q_proj v_proj o_proj \
  --target-layers 16 17 18 19 \
  --learning-rate 5e-5 \
  --semantic-weight 120.0 \
  --acoustic-weight 1.0 \
  --weight-decay 0.01 \
  --batch-size 4 \
  --epochs 10 \
  --val-split 0.1 \
  --val-every 50 \
  --save-every 200 \
  --max-grad-norm 1.0 \
  --save-mode both \
  --generate-samples \
  --sample-prompt "This is an advanced fine-tuning example."