#!/usr/bin/env python
"""
LoRA Training Benchmark for CSM.

This script runs comprehensive benchmarks for LoRA fine-tuning implementation,
comparing different configurations and measuring performance.

Usage:
  python -m csm.training.run_lora_benchmark [--tiny] [--output-dir OUTPUT_DIR]
  
Options:
  --tiny          Run with tiny model for quick testing
  --output-dir    Directory to save benchmark results (default: ./benchmark_results)
  --ranks         Comma-separated list of LoRA ranks to benchmark (default: 8,16,32)
  --modules       Comma-separated list of module configs to benchmark 
                  (default: q_proj,q_proj+v_proj,q_proj+v_proj+o_proj)
"""

import os
import sys
import time
import json
import argparse
import tempfile
import numpy as np
from pathlib import Path

# Check for MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available. This benchmark requires MLX to run.")
    
try:
    import safetensors.numpy
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("safetensors.numpy not available. This benchmark requires safetensors.")


def create_tiny_model(output_path):
    """Create a tiny model for benchmarking."""
    if not HAS_SAFETENSORS:
        print("Cannot create tiny model: safetensors.numpy not available")
        return None
    
    try:
        # Set dimensions for the tiny model
        hidden_size = 16
        num_heads = 2
        head_dim = hidden_size // num_heads
        seq_len = 32
        vocab_size = 100
        
        # Create a tiny 'model' with more comprehensive parameters
        weights = {
            # Model configuration
            "backbone.hidden_size": np.array(hidden_size, dtype=np.int32),
            "backbone.num_heads": np.array(num_heads, dtype=np.int32),
            "backbone.head_dim": np.array(head_dim, dtype=np.int32),
            "backbone.num_layers": np.array(1, dtype=np.int32),
            "backbone.vocab_size": np.array(vocab_size, dtype=np.int32),
            
            # Backbone components
            "backbone.embed_dim": np.array(hidden_size, dtype=np.int32),
            
            # Layer 0 - General 
            "backbone.layers.0.input_layernorm_weight": np.ones(hidden_size, dtype=np.float32),
            "backbone.layers.0.input_layernorm_bias": np.zeros(hidden_size, dtype=np.float32),
            "backbone.layers.0.post_attention_layernorm_weight": np.ones(hidden_size, dtype=np.float32),
            "backbone.layers.0.post_attention_layernorm_bias": np.zeros(hidden_size, dtype=np.float32),
            
            # Layer 0 - Attention
            "backbone.layers.0.attn.q_proj_weight": np.ones((hidden_size, hidden_size), dtype=np.float32) * 0.1,
            "backbone.layers.0.attn.q_proj_bias": np.zeros(hidden_size, dtype=np.float32),
            "backbone.layers.0.attn.k_proj_weight": np.ones((hidden_size, hidden_size), dtype=np.float32) * 0.1,
            "backbone.layers.0.attn.k_proj_bias": np.zeros(hidden_size, dtype=np.float32),
            "backbone.layers.0.attn.v_proj_weight": np.ones((hidden_size, hidden_size), dtype=np.float32) * 0.1,
            "backbone.layers.0.attn.v_proj_bias": np.zeros(hidden_size, dtype=np.float32),
            "backbone.layers.0.attn.o_proj_weight": np.ones((hidden_size, hidden_size), dtype=np.float32) * 0.1,
            "backbone.layers.0.attn.o_proj_bias": np.zeros(hidden_size, dtype=np.float32),
            
            # Layer 0 - MLP
            "backbone.layers.0.mlp.gate_proj_weight": np.ones((hidden_size * 4, hidden_size), dtype=np.float32) * 0.1,
            "backbone.layers.0.mlp.gate_proj_bias": np.zeros(hidden_size * 4, dtype=np.float32),
            "backbone.layers.0.mlp.up_proj_weight": np.ones((hidden_size * 4, hidden_size), dtype=np.float32) * 0.1,
            "backbone.layers.0.mlp.up_proj_bias": np.zeros(hidden_size * 4, dtype=np.float32),
            "backbone.layers.0.mlp.down_proj_weight": np.ones((hidden_size, hidden_size * 4), dtype=np.float32) * 0.1,
            "backbone.layers.0.mlp.down_proj_bias": np.zeros(hidden_size, dtype=np.float32),
            
            # Final layer norm
            "backbone.final_layernorm_weight": np.ones(hidden_size, dtype=np.float32),
            "backbone.final_layernorm_bias": np.zeros(hidden_size, dtype=np.float32),
            
            # Decoder - configuration
            "decoder.hidden_size": np.array(hidden_size, dtype=np.int32),
            "decoder.num_heads": np.array(num_heads, dtype=np.int32),
            "decoder.head_dim": np.array(head_dim, dtype=np.int32),
            "decoder.num_layers": np.array(1, dtype=np.int32),
            
            # Decoder components
            "decoder.embed_dim": np.array(hidden_size, dtype=np.int32),
            
            # Decoder Layer 0 - General
            "decoder.layers.0.input_layernorm_weight": np.ones(hidden_size, dtype=np.float32),
            "decoder.layers.0.input_layernorm_bias": np.zeros(hidden_size, dtype=np.float32),
            "decoder.layers.0.post_attention_layernorm_weight": np.ones(hidden_size, dtype=np.float32),
            "decoder.layers.0.post_attention_layernorm_bias": np.zeros(hidden_size, dtype=np.float32),
            
            # Decoder Layer 0 - Attention
            "decoder.layers.0.attn.q_proj_weight": np.ones((hidden_size, hidden_size), dtype=np.float32) * 0.1,
            "decoder.layers.0.attn.q_proj_bias": np.zeros(hidden_size, dtype=np.float32),
            "decoder.layers.0.attn.k_proj_weight": np.ones((hidden_size, hidden_size), dtype=np.float32) * 0.1, 
            "decoder.layers.0.attn.k_proj_bias": np.zeros(hidden_size, dtype=np.float32),
            "decoder.layers.0.attn.v_proj_weight": np.ones((hidden_size, hidden_size), dtype=np.float32) * 0.1,
            "decoder.layers.0.attn.v_proj_bias": np.zeros(hidden_size, dtype=np.float32),
            "decoder.layers.0.attn.o_proj_weight": np.ones((hidden_size, hidden_size), dtype=np.float32) * 0.1,
            "decoder.layers.0.attn.o_proj_bias": np.zeros(hidden_size, dtype=np.float32),
            
            # Decoder Layer 0 - MLP
            "decoder.layers.0.mlp.gate_proj_weight": np.ones((hidden_size * 4, hidden_size), dtype=np.float32) * 0.1,
            "decoder.layers.0.mlp.gate_proj_bias": np.zeros(hidden_size * 4, dtype=np.float32),
            "decoder.layers.0.mlp.up_proj_weight": np.ones((hidden_size * 4, hidden_size), dtype=np.float32) * 0.1,
            "decoder.layers.0.mlp.up_proj_bias": np.zeros(hidden_size * 4, dtype=np.float32),
            "decoder.layers.0.mlp.down_proj_weight": np.ones((hidden_size, hidden_size * 4), dtype=np.float32) * 0.1,
            "decoder.layers.0.mlp.down_proj_bias": np.zeros(hidden_size, dtype=np.float32),
            
            # Final layer norm
            "decoder.final_layernorm_weight": np.ones(hidden_size, dtype=np.float32),
            "decoder.final_layernorm_bias": np.zeros(hidden_size, dtype=np.float32),
            
            # Embedding components
            "text_embeddings": np.ones((vocab_size, hidden_size), dtype=np.float32) * 0.1,
            "audio_embeddings": np.ones((vocab_size, hidden_size), dtype=np.float32) * 0.1,
            
            # Heads
            "codebook0_head": np.ones((hidden_size, vocab_size), dtype=np.float32) * 0.1,
            "audio_head.0": np.ones((hidden_size, vocab_size), dtype=np.float32) * 0.1,
            "projection": np.ones((hidden_size, hidden_size), dtype=np.float32) * 0.1,
            
            # Positional information
            "backbone.rope.cos_cached": np.ones((seq_len, head_dim), dtype=np.float32) * 0.1,
            "backbone.rope.sin_cached": np.ones((seq_len, head_dim), dtype=np.float32) * 0.1,
            "decoder.rope.cos_cached": np.ones((seq_len, head_dim), dtype=np.float32) * 0.1,
            "decoder.rope.sin_cached": np.ones((seq_len, head_dim), dtype=np.float32) * 0.1
        }
        
        safetensors.numpy.save_file(weights, output_path)
        
        # Create metadata file
        metadata = {
            "epoch": 0,
            "global_step": 0,
            "loss": 1.0,
            "model_path": output_path
        }
        
        metadata_path = output_path.replace(".safetensors", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
            
        print(f"Created tiny model at {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error creating tiny model: {e}")
        return None


def create_mock_dataset(size=10, seq_len=16, batch_size=2):
    """Create a mock dataset for benchmarking."""
    if not HAS_MLX:
        print("Cannot create mock dataset: MLX not available")
        return None
    
    class MockMLXDataset:
        def __init__(self, size=size, seq_len=seq_len):
            self.size = size
            self.seq_len = seq_len
        
        def __len__(self):
            return self.size
        
        def get_batch(self, batch_idx, batch_size):
            # Create a dummy batch
            input_tokens = mx.zeros((batch_size, self.seq_len, 3), dtype=mx.int32)
            input_masks = mx.ones((batch_size, self.seq_len, 3), dtype=mx.float32)
            target_audio_tokens = mx.zeros((batch_size, self.seq_len, 2), dtype=mx.int32)
            
            return {
                "input_tokens": input_tokens,
                "input_masks": input_masks,
                "target_audio_tokens": target_audio_tokens
            }
    
    return MockMLXDataset()


def run_lora_benchmark(model_path, output_dir, lora_ranks=None, module_configs=None, tiny=False):
    """Run LoRA training benchmark with different configurations."""
    if not HAS_MLX:
        print("Cannot run LoRA benchmark: MLX not available")
        return None
    
    print(f"\n{'-'*80}")
    print(f"RUNNING LORA TRAINING BENCHMARK")
    print(f"{'-'*80}")
    
    # Default configurations if not provided
    if lora_ranks is None:
        lora_ranks = [8, 16, 32]
    
    if module_configs is None:
        module_configs = [
            ["q_proj"],
            ["q_proj", "v_proj"],
            ["q_proj", "v_proj", "o_proj"]
        ]
    
    # Results dictionary
    results = {
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "device": "Apple Silicon",
        "model_path": model_path,
        "tiny_model": tiny,
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {}
    }
    
    # Create mock dataset for testing
    mock_dataset = create_mock_dataset(size=10)
    
    # Run benchmarks for each configuration
    for lora_r in lora_ranks:
        for module_config in module_configs:
            config_name = f"r{lora_r}_{'_'.join(module_config)}"
            print(f"\n{'-'*40}")
            print(f"Benchmarking LoRA config: {config_name}")
            print(f"LoRA rank: {lora_r}, Target modules: {', '.join(module_config)}")
            print(f"{'-'*40}")
            
            # Initialize LoRA trainer with the current configuration
            try:
                from csm.training.lora_trainer import CSMLoRATrainer
                
                # 1. Model Loading Benchmark
                print("\n1. Model Loading + LoRA Initialization")
                start_time = time.time()
                
                trainer = CSMLoRATrainer(
                    model_path=model_path,
                    output_dir=output_dir,
                    lora_r=lora_r,
                    lora_alpha=lora_r * 2.0,  # Common practice is alpha = 2*r
                    target_modules=module_config,
                    learning_rate=1e-4
                )
                
                loading_time = time.time() - start_time
                print(f"   Loading time: {loading_time:.6f} seconds")
                
                # 2. Parameter Count Analysis
                print("\n2. Parameter Analysis")
                # Count trainable (LoRA) parameters
                lora_params = trainer.model.get_lora_params()
                lora_param_count = sum(np.prod(p.shape) for p in lora_params.values())
                
                # Estimate total model parameters
                total_param_count = 0
                if hasattr(trainer.model, 'backbone') and hasattr(trainer.model.backbone, 'base_model'):
                    backbone_model = trainer.model.backbone.base_model
                    # Count parameters in backbone
                    if hasattr(backbone_model, 'parameters'):
                        backbone_params = backbone_model.parameters()
                        backbone_param_count = sum(np.prod(p.shape) for p in backbone_params.values())
                        total_param_count += backbone_param_count
                
                if hasattr(trainer.model, 'decoder') and hasattr(trainer.model.decoder, 'base_model'):
                    decoder_model = trainer.model.decoder.base_model
                    # Count parameters in decoder
                    if hasattr(decoder_model, 'parameters'):
                        decoder_params = decoder_model.parameters()
                        decoder_param_count = sum(np.prod(p.shape) for p in decoder_params.values())
                        total_param_count += decoder_param_count
                
                # Calculate parameter reduction percentage
                param_percentage = (lora_param_count / total_param_count * 100) if total_param_count > 0 else 0
                
                print(f"   LoRA Parameters: {lora_param_count:,}")
                print(f"   Total Model Parameters: {total_param_count:,}")
                print(f"   Parameter Ratio: {param_percentage:.4f}%")
                
                # 3. Optimizer Preparation Benchmark
                print("\n3. Optimizer Preparation")
                start_time = time.time()
                trainer.prepare_optimizer()
                optimizer_prep_time = time.time() - start_time
                print(f"   Optimizer preparation time: {optimizer_prep_time:.6f} seconds")
                
                # 4. Forward Pass Benchmark (LoRA)
                print("\n4. Forward Pass (with LoRA)")
                # Get a batch
                batch = mock_dataset.get_batch(0, 2)
                
                # Measure forward pass time
                from csm.training.utils import compute_loss_mlx
                
                # Warmup run
                _, _ = compute_loss_mlx(
                    trainer.model,
                    batch["input_tokens"],
                    batch["input_masks"],
                    batch["target_audio_tokens"],
                    semantic_weight=100.0, 
                    acoustic_weight=1.0
                )
                
                # Timed runs
                forward_times = []
                for _ in range(5):
                    start_time = time.time()
                    _, _ = compute_loss_mlx(
                        trainer.model,
                        batch["input_tokens"],
                        batch["input_masks"],
                        batch["target_audio_tokens"],
                        semantic_weight=100.0, 
                        acoustic_weight=1.0
                    )
                    forward_times.append(time.time() - start_time)
                
                avg_forward_time = sum(forward_times) / len(forward_times)
                print(f"   Average forward pass time: {avg_forward_time:.6f} seconds")
                
                # 5. Training Step Benchmark (Forward + Backward with LoRA)
                print("\n5. Training Step (with LoRA)")
                
                # Warmup
                _ = trainer.train_step(batch)
                
                # Timed runs
                train_step_times = []
                for _ in range(5):
                    start_time = time.time()
                    _ = trainer.train_step(batch)
                    train_step_times.append(time.time() - start_time)
                
                avg_train_step_time = sum(train_step_times) / len(train_step_times)
                print(f"   Average training step time: {avg_train_step_time:.6f} seconds")
                
                # Calculate tokens per second
                batch_size = batch["input_tokens"].shape[0]
                seq_len = batch["input_tokens"].shape[1]
                tokens_per_batch = batch_size * seq_len
                tokens_per_second = tokens_per_batch / avg_train_step_time
                
                print(f"   Throughput: {tokens_per_second:.2f} tokens/second")
                
                # 6. Memory Usage (Placeholder)
                print("\n6. Memory Usage (Placeholder)")
                print("   Memory usage statistics not directly available for MLX")
                
                # 7. Checkpoint Saving Benchmark (LoRA-only)
                print("\n7. LoRA Checkpoint Saving")
                
                # Timed runs for LoRA-only saving
                lora_save_times = []
                for i in range(3):
                    temp_save_path = os.path.join(output_dir, f"benchmark_lora_{i}.safetensors")
                    start_time = time.time()
                    trainer.save_model(temp_save_path, save_mode="lora")
                    lora_save_times.append(time.time() - start_time)
                
                avg_lora_save_time = sum(lora_save_times) / len(lora_save_times)
                print(f"   Average LoRA checkpoint save time: {avg_lora_save_time:.6f} seconds")
                
                # 8. Full Model Saving Benchmark (with merged weights)
                print("\n8. Full Model Checkpoint Saving")
                
                # Timed runs for full model saving
                full_save_times = []
                for i in range(3):
                    temp_save_path = os.path.join(output_dir, f"benchmark_full_{i}.safetensors")
                    start_time = time.time()
                    trainer.save_model(temp_save_path, save_mode="full")
                    full_save_times.append(time.time() - start_time)
                
                avg_full_save_time = sum(full_save_times) / len(full_save_times)
                print(f"   Average full model save time: {avg_full_save_time:.6f} seconds")
                
                # Store results for this configuration
                results["benchmarks"][config_name] = {
                    "lora_rank": lora_r,
                    "target_modules": module_config,
                    "parameters": {
                        "lora_parameters": int(lora_param_count),
                        "total_parameters": int(total_param_count),
                        "parameter_percentage": float(param_percentage)
                    },
                    "timing": {
                        "loading_time": float(loading_time),
                        "optimizer_preparation": float(optimizer_prep_time),
                        "forward_pass": {
                            "average_time": float(avg_forward_time),
                            "samples": [float(t) for t in forward_times]
                        },
                        "training_step": {
                            "average_time": float(avg_train_step_time),
                            "samples": [float(t) for t in train_step_times],
                            "throughput": {
                                "tokens_per_second": float(tokens_per_second),
                                "batch_size": int(batch_size),
                                "sequence_length": int(seq_len)
                            }
                        },
                        "checkpoint_saving": {
                            "lora_only": {
                                "average_time": float(avg_lora_save_time),
                                "samples": [float(t) for t in lora_save_times]
                            },
                            "full_model": {
                                "average_time": float(avg_full_save_time),
                                "samples": [float(t) for t in full_save_times]
                            }
                        }
                    }
                }
                
            except Exception as e:
                print(f"Error benchmarking configuration {config_name}: {e}")
                import traceback
                print(traceback.format_exc())
                
                # Record the error in results
                results["benchmarks"][config_name] = {
                    "lora_rank": lora_r,
                    "target_modules": module_config,
                    "error": str(e)
                }
    
    # Calculate comparative metrics across configurations
    if len(results["benchmarks"]) > 1:
        print(f"\n{'-'*80}")
        print(f"COMPARATIVE RESULTS")
        print(f"{'-'*80}")
        
        # Extract key metrics for comparison
        metrics = {}
        baseline_config = None
        
        for config_name, benchmark in results["benchmarks"].items():
            if "error" not in benchmark:
                if baseline_config is None:
                    baseline_config = config_name
                
                metrics[config_name] = {
                    "lora_rank": benchmark["lora_rank"],
                    "target_modules": benchmark["target_modules"],
                    "parameter_percentage": benchmark["parameters"]["parameter_percentage"],
                    "forward_time": benchmark["timing"]["forward_pass"]["average_time"],
                    "training_time": benchmark["timing"]["training_step"]["average_time"],
                    "tokens_per_second": benchmark["timing"]["training_step"]["throughput"]["tokens_per_second"],
                    "lora_save_time": benchmark["timing"]["checkpoint_saving"]["lora_only"]["average_time"],
                    "full_save_time": benchmark["timing"]["checkpoint_saving"]["full_model"]["average_time"]
                }
        
        # Print comparison table
        if baseline_config:
            print("\nConfiguration Comparison:")
            print(f"{'Config':<20} {'Param %':<10} {'Train Time':<15} {'Tokens/sec':<15} {'LoRA Save':<15}")
            print("-" * 80)
            
            baseline = metrics[baseline_config]
            for config_name, metric in metrics.items():
                relative_speed = f"{metric['tokens_per_second'] / baseline['tokens_per_second']:.2f}x"
                print(f"{config_name:<20} {metric['parameter_percentage']:<10.4f} "
                      f"{metric['training_time']:<15.6f} {metric['tokens_per_second']:<15.2f} "
                      f"{metric['lora_save_time']:<15.6f}")
            
            # Add comparative analysis to results
            results["comparative_analysis"] = {
                "baseline_config": baseline_config,
                "metrics": metrics
            }
        
        # Parameter efficiency vs speed trade-off analysis
        print("\nEfficiency vs Speed Analysis:")
        efficiency_metrics = []
        
        for config_name, metric in metrics.items():
            efficiency_metrics.append({
                "config": config_name,
                "parameter_percentage": metric["parameter_percentage"],
                "tokens_per_second": metric["tokens_per_second"],
                "efficiency_score": metric["tokens_per_second"] / (metric["parameter_percentage"] + 0.1)  # Add small constant to avoid division by zero
            })
        
        # Sort by efficiency score (higher is better)
        efficiency_metrics.sort(key=lambda x: x["efficiency_score"], reverse=True)
        
        print(f"{'Config':<20} {'Param %':<10} {'Tokens/sec':<15} {'Efficiency Score':<20}")
        print("-" * 80)
        for item in efficiency_metrics:
            print(f"{item['config']:<20} {item['parameter_percentage']:<10.4f} "
                  f"{item['tokens_per_second']:<15.2f} {item['efficiency_score']:<20.2f}")
        
        # Add efficiency analysis to results
        results["efficiency_analysis"] = efficiency_metrics
    
    # Save benchmark results
    results_path = os.path.join(output_dir, "lora_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to {results_path}")
    return results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LoRA Training Benchmark for CSM")
    parser.add_argument("--tiny", action="store_true", help="Run with tiny model for quick testing")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Directory to save benchmark results")
    parser.add_argument("--ranks", default="8,16,32", help="Comma-separated list of LoRA ranks to benchmark")
    parser.add_argument("--modules", default="q_proj,q_proj+v_proj,q_proj+v_proj+o_proj", 
                       help="Comma-separated list of module configs to benchmark")
    
    args = parser.parse_args()
    
    # Parse ranks
    lora_ranks = [int(r) for r in args.ranks.split(",")]
    
    # Parse module configs
    module_configs = []
    for config in args.modules.split(","):
        module_configs.append(config.split("+"))
    
    return args, lora_ranks, module_configs


def main():
    """Run LoRA benchmarks."""
    # Parse arguments
    args, lora_ranks, module_configs = parse_args()
    
    # Check requirements
    missing_requirements = []
    if not HAS_MLX:
        missing_requirements.append("MLX")
    if not HAS_SAFETENSORS:
        missing_requirements.append("safetensors")
    
    if missing_requirements:
        print(f"ERROR: Missing required packages: {', '.join(missing_requirements)}")
        print("Please install these packages to run the benchmark.")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Benchmark results will be saved to {args.output_dir}")
    
    # Create model for benchmarking
    if args.tiny:
        model_path = os.path.join(args.output_dir, "tiny_benchmark_model.safetensors")
        model_path = create_tiny_model(model_path)
        if not model_path:
            print("Failed to create tiny model.")
            return 1
    else:
        # Use model specified in arguments or a default path
        model_path = "model.safetensors"  # Default - user would override
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist.")
            print("Creating a tiny model instead...")
            model_path = os.path.join(args.output_dir, "tiny_benchmark_model.safetensors")
            model_path = create_tiny_model(model_path)
            if not model_path:
                print("Failed to create tiny model.")
                return 1
    
    # Run LoRA benchmark
    results = run_lora_benchmark(
        model_path=model_path,
        output_dir=args.output_dir,
        lora_ranks=lora_ranks,
        module_configs=module_configs,
        tiny=args.tiny
    )
    
    # Create combined results summary
    summary = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "python_version": sys.version,
            "os": os.name,
            "platform": sys.platform
        },
        "mlx_available": HAS_MLX,
        "safetensors_available": HAS_SAFETENSORS,
        "model_path": model_path,
        "tiny_model": args.tiny,
        "lora_ranks_tested": lora_ranks,
        "module_configs_tested": module_configs,
        "results": results
    }
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "lora_benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBenchmark summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())