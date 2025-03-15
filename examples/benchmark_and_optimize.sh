#!/bin/bash
# Benchmark and optimization example
# This script runs benchmarks to determine optimal LoRA configuration

# Create output directories
mkdir -p ./benchmark_results

# Run benchmark with different LoRA configurations
python -m csm.training.run_lora_benchmark \
  --output-dir ./benchmark_results \
  --ranks 4,8,16,32 \
  --modules q_proj,q_proj+v_proj,q_proj+v_proj+o_proj,q_proj+k_proj+v_proj+o_proj

# Run standard MLX benchmark for comparison
python -m csm.training.run_mlx_benchmark \
  --output-dir ./benchmark_results

# Find best configuration from benchmark results
echo "Benchmark results saved to ./benchmark_results"
echo "Review lora_benchmark_summary.json for optimal configuration"

# Example fine-tuning using benchmark results
# Uncomment and modify with the optimal configuration found from benchmarks
: '
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_models/optimized \
  --audio-dir ./data/speaker1/audio \
  --transcript-dir ./data/speaker1/transcripts \
  --speaker-id 1 \
  --lora-r 8 \
  --lora-alpha 16 \
  --target-modules q_proj v_proj \
  --batch-size 4 \
  --epochs 5 \
  --generate-samples
'