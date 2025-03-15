#!/usr/bin/env python
"""
MLX Training Benchmark for CSM.

This script runs comprehensive benchmarks for MLX training implementation,
comparing it to PyTorch training where possible.

Usage:
  python -m csm.training.run_mlx_benchmark [--tiny] [--output-dir OUTPUT_DIR]
  
Options:
  --tiny         Run with tiny model for quick testing
  --output-dir   Directory to save benchmark results (default: ./benchmark_results)
"""

import os
import sys
import time
import json
import argparse
import tempfile
import numpy as np
from pathlib import Path

# Check for MLX and PyTorch
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available. This benchmark requires MLX to run.")
    
try:
    import torch
    import torch.nn as torch_nn
    import torch.optim as torch_optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. Some comparative benchmarks will be skipped.")

try:
    import safetensors.numpy
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("safetensors.numpy not available. This benchmark requires safetensors.")


def create_tiny_model(output_path):
    """Create a tiny model for benchmarking with proper structure for LoRA."""
    if not HAS_SAFETENSORS:
        print("Cannot create tiny model: safetensors.numpy not available")
        return None
    
    try:
        # Create a tiny model with minimal parameters but proper structure for LoRA
        hidden_size = 16
        num_layers = 2
        num_heads = 2
        head_dim = hidden_size // num_heads
        
        # Create weights dictionary with proper transformer structures
        weights = {
            # Model dimensions
            "backbone.hidden_size": np.array(hidden_size, dtype=np.int32),
            "backbone.num_heads": np.array(num_heads, dtype=np.int32),
            "backbone.num_layers": np.array(num_layers, dtype=np.int32),
            "backbone.head_dim": np.array(head_dim, dtype=np.int32),
            
            "decoder.hidden_size": np.array(hidden_size, dtype=np.int32),
            "decoder.num_heads": np.array(num_heads, dtype=np.int32),
            "decoder.num_layers": np.array(num_layers, dtype=np.int32),
            "decoder.head_dim": np.array(head_dim, dtype=np.int32),
            
            # Embeddings
            "text_embeddings": np.zeros((100, hidden_size), dtype=np.float32),
            "audio_embeddings": np.zeros((100, hidden_size), dtype=np.float32),
            
            # Heads
            "codebook0_head": np.zeros((hidden_size, 100), dtype=np.float32),
            "audio_head.0": np.zeros((hidden_size, 100), dtype=np.float32),
            "projection": np.zeros((hidden_size, hidden_size), dtype=np.float32)
        }
        
        # Add backbone layers with proper components
        for i in range(num_layers):
            # Attention components
            weights[f"backbone.layers.{i}.q_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"backbone.layers.{i}.k_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"backbone.layers.{i}.v_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"backbone.layers.{i}.o_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            
            # Add bias terms
            weights[f"backbone.layers.{i}.q_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.k_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.v_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.o_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            
            # MLP components
            weights[f"backbone.layers.{i}.gate_proj_weight"] = np.zeros((hidden_size * 4, hidden_size), dtype=np.float32)
            weights[f"backbone.layers.{i}.up_proj_weight"] = np.zeros((hidden_size * 4, hidden_size), dtype=np.float32)
            weights[f"backbone.layers.{i}.down_proj_weight"] = np.zeros((hidden_size, hidden_size * 4), dtype=np.float32)
            
            # Layernorm components
            weights[f"backbone.layers.{i}.input_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.input_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.post_attention_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.post_attention_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
        
        # Add decoder layers with proper components
        for i in range(num_layers):
            # Attention components
            weights[f"decoder.layers.{i}.q_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"decoder.layers.{i}.k_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"decoder.layers.{i}.v_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"decoder.layers.{i}.o_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            
            # Add bias terms
            weights[f"decoder.layers.{i}.q_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.k_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.v_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.o_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            
            # MLP components
            weights[f"decoder.layers.{i}.gate_proj_weight"] = np.zeros((hidden_size * 4, hidden_size), dtype=np.float32)
            weights[f"decoder.layers.{i}.up_proj_weight"] = np.zeros((hidden_size * 4, hidden_size), dtype=np.float32)
            weights[f"decoder.layers.{i}.down_proj_weight"] = np.zeros((hidden_size, hidden_size * 4), dtype=np.float32)
            
            # Layernorm components
            weights[f"decoder.layers.{i}.input_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.input_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.post_attention_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.post_attention_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
        
        # Add final layernorm components
        weights["backbone.final_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
        weights["backbone.final_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
        weights["decoder.final_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
        weights["decoder.final_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
        
        # Save weights to safetensors
        safetensors.numpy.save_file(weights, output_path)
        
        # Create metadata file
        metadata = {
            "epoch": 0,
            "global_step": 0,
            "loss": 1.0,
            "model_path": output_path,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads
        }
        
        metadata_path = output_path.replace(".safetensors", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Created tiny model at {output_path} with proper LoRA structure")
        return output_path
    
    except Exception as e:
        print(f"Error creating tiny model: {e}")
        import traceback
        print(traceback.format_exc())
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


def run_mlx_benchmark(model_path, output_dir, tiny=False):
    """Run MLX training benchmark."""
    if not HAS_MLX:
        print("Cannot run MLX benchmark: MLX not available")
        return None
    
    from csm.training.mlx_trainer import CSMMLXTrainer
    
    print(f"\n{'-'*80}")
    print(f"RUNNING MLX TRAINING BENCHMARK")
    print(f"{'-'*80}")
    
    # Results dictionary
    results = {
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "device": "Apple Silicon",
        "model_path": model_path,
        "tiny_model": tiny,
        "benchmarks": {}
    }
    
    # 1. Model Loading Benchmark
    print("\n1. Model Loading Benchmark")
    start_time = time.time()
    trainer = CSMMLXTrainer(
        model_path=model_path,
        output_dir=output_dir,
        learning_rate=1e-4,
        backbone_lr_multiplier=0.5,
        decoder_lr_multiplier=1.0,
        embedding_lr_multiplier=0.5,
        semantic_weight=100.0,
        acoustic_weight=1.0
    )
    model_load_time = time.time() - start_time
    print(f"   Model loading time: {model_load_time:.6f} seconds")
    results["benchmarks"]["model_loading"] = {
        "time_seconds": model_load_time
    }
    
    # 2. Optimizer Preparation Benchmark
    print("\n2. Optimizer Preparation Benchmark")
    start_time = time.time()
    trainer.prepare_optimizer()
    optimizer_prep_time = time.time() - start_time
    print(f"   Optimizer preparation time: {optimizer_prep_time:.6f} seconds")
    results["benchmarks"]["optimizer_preparation"] = {
        "time_seconds": optimizer_prep_time
    }
    
    # 3. Forward Pass Benchmark
    print("\n3. Forward Pass Benchmark")
    # Create mock dataset for testing
    mock_dataset = create_mock_dataset(size=10)
    
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
    results["benchmarks"]["forward_pass"] = {
        "time_seconds": avg_forward_time,
        "samples": forward_times
    }
    
    # 4. Training Step Benchmark (Forward + Backward)
    print("\n4. Training Step Benchmark")
    
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
    results["benchmarks"]["train_step"] = {
        "time_seconds": avg_train_step_time,
        "samples": train_step_times,
        "throughput": {
            "tokens_per_second": tokens_per_second,
            "batch_size": batch_size,
            "sequence_length": seq_len
        }
    }
    
    # 5. Checkpoint Saving Benchmark
    print("\n5. Checkpoint Saving Benchmark")
    from csm.training.utils import save_checkpoint_mlx
    
    # Warmup
    save_checkpoint_mlx(
        trainer.model,
        trainer.optimizer,
        epoch=0,
        global_step=0,
        loss=1.0,
        save_dir=output_dir,
        name="warmup_checkpoint"
    )
    
    # Timed runs
    checkpoint_save_times = []
    for i in range(3):
        start_time = time.time()
        save_checkpoint_mlx(
            trainer.model,
            trainer.optimizer,
            epoch=0,
            global_step=i,
            loss=1.0,
            save_dir=output_dir,
            name=f"benchmark_checkpoint_{i}"
        )
        checkpoint_save_times.append(time.time() - start_time)
    
    avg_checkpoint_save_time = sum(checkpoint_save_times) / len(checkpoint_save_times)
    print(f"   Average checkpoint save time: {avg_checkpoint_save_time:.6f} seconds")
    results["benchmarks"]["checkpoint_saving"] = {
        "time_seconds": avg_checkpoint_save_time,
        "samples": checkpoint_save_times
    }
    
    # 6. Checkpoint Loading Benchmark
    print("\n6. Checkpoint Loading Benchmark")
    from csm.training.utils import load_checkpoint_mlx
    
    # Use the last saved checkpoint
    checkpoint_path = os.path.join(output_dir, f"benchmark_checkpoint_{i}.safetensors")
    
    # Timed runs
    checkpoint_load_times = []
    for _ in range(3):
        start_time = time.time()
        load_checkpoint_mlx(
            checkpoint_path,
            trainer.model,
            trainer.optimizer
        )
        checkpoint_load_times.append(time.time() - start_time)
    
    avg_checkpoint_load_time = sum(checkpoint_load_times) / len(checkpoint_load_times)
    print(f"   Average checkpoint load time: {avg_checkpoint_load_time:.6f} seconds")
    results["benchmarks"]["checkpoint_loading"] = {
        "time_seconds": avg_checkpoint_load_time,
        "samples": checkpoint_load_times
    }
    
    # 7. Memory Usage Statistics
    # Note: MLX doesn't provide direct memory statistics like PyTorch
    # This is a placeholder - in a real benchmark, you'd want to use 
    # platform-specific tools for Apple Silicon memory usage
    
    print("\n7. Memory Usage (Placeholder)")
    print("   Memory usage statistics not directly available for MLX")
    print("   Use Activity Monitor or other Apple Silicon tools for memory profiling")
    
    # Save benchmark results
    results_path = os.path.join(output_dir, "mlx_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to {results_path}")
    return results


def run_pytorch_comparison(model_path, output_dir, tiny=False):
    """Run PyTorch comparison benchmark if available."""
    if not HAS_TORCH:
        print("Cannot run PyTorch comparison: PyTorch not available")
        return None
    
    print(f"\n{'-'*80}")
    print(f"RUNNING PYTORCH COMPARISON BENCHMARK")
    print(f"{'-'*80}")
    
    # Results dictionary
    results = {
        "torch_version": torch.__version__,
        "device": "CPU" if not torch.cuda.is_available() else "CUDA",
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "model_path": model_path,
        "tiny_model": tiny,
        "benchmarks": {}
    }
    
    # In a real benchmark, you would implement PyTorch equivalents
    # of all the MLX benchmarks above. For this example, we'll just
    # do some basic tensor operations to compare raw performance.
    
    # Use CPU for fair comparison with MLX on Apple Silicon
    device = torch.device("cpu")
    
    # Basic matrix multiplication benchmark
    print("\nBasic Matrix Multiplication Benchmark")
    
    # PyTorch timing
    x_torch = torch.randn(1000, 1000, device=device)
    
    # Warmup
    y_torch = torch.matmul(x_torch, x_torch)
    
    # Timed runs
    torch_times = []
    for _ in range(5):
        start_time = time.time()
        y_torch = torch.matmul(x_torch, x_torch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        torch_times.append(time.time() - start_time)
    
    avg_torch_time = sum(torch_times) / len(torch_times)
    print(f"   PyTorch avg time: {avg_torch_time:.6f} seconds")
    
    # MLX timing
    x_mlx = mx.random.normal((1000, 1000))
    
    # Warmup
    y_mlx = mx.matmul(x_mlx, x_mlx)
    mx.eval(y_mlx)
    
    # Timed runs
    mlx_times = []
    for _ in range(5):
        start_time = time.time()
        y_mlx = mx.matmul(x_mlx, x_mlx)
        mx.eval(y_mlx)  # Force evaluation
        mlx_times.append(time.time() - start_time)
    
    avg_mlx_time = sum(mlx_times) / len(mlx_times)
    print(f"   MLX avg time: {avg_mlx_time:.6f} seconds")
    
    # Calculate speedup
    if avg_torch_time > 0 and avg_mlx_time > 0:
        speedup = avg_torch_time / avg_mlx_time
        print(f"   Speedup (MLX vs PyTorch): {speedup:.2f}x")
    else:
        speedup = float('nan')
        print("   Could not calculate speedup")
    
    # Save benchmark results
    results["benchmarks"]["matrix_multiplication"] = {
        "pytorch_time_seconds": avg_torch_time,
        "mlx_time_seconds": avg_mlx_time,
        "speedup": speedup,
        "pytorch_samples": torch_times,
        "mlx_samples": mlx_times
    }
    
    # Save results
    results_path = os.path.join(output_dir, "pytorch_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparison results saved to {results_path}")
    return results


def main():
    """Run MLX benchmarks and PyTorch comparison."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="MLX Training Benchmark for CSM")
    parser.add_argument("--tiny", action="store_true", help="Run with tiny model for quick testing")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Directory to save benchmark results")
    args = parser.parse_args()
    
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
    
    # Run MLX benchmark
    mlx_results = run_mlx_benchmark(
        model_path=model_path,
        output_dir=args.output_dir,
        tiny=args.tiny
    )
    
    # Run PyTorch comparison if available
    if HAS_TORCH:
        pytorch_results = run_pytorch_comparison(
            model_path=model_path,
            output_dir=args.output_dir,
            tiny=args.tiny
        )
    
    # Create combined results
    combined_results = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "python_version": sys.version,
            "os": os.name,
            "platform": sys.platform
        },
        "mlx_available": HAS_MLX,
        "pytorch_available": HAS_TORCH,
        "safetensors_available": HAS_SAFETENSORS,
        "model_path": model_path,
        "tiny_model": args.tiny,
        "mlx_results": mlx_results,
        "pytorch_results": pytorch_results if HAS_TORCH else None
    }
    
    # Save combined results
    combined_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(combined_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nBenchmark summary saved to {combined_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())