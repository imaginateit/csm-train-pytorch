"""
Benchmarking and optimization tools for LoRA fine-tuning.

This module provides benchmarking utilities for measuring performance
of LoRA fine-tuning on Apple Silicon devices with MLX acceleration.
"""

import os
import time
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Set

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .lora_trainer import CSMLoRATrainer
from .utils import setup_logger


class BenchmarkConfig:
    """Configuration for LoRA fine-tuning benchmarks."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        lora_ranks: List[int] = [4, 8, 16, 32],
        target_modules_options: List[List[str]] = [["q_proj", "v_proj"], ["q_proj", "k_proj", "v_proj", "o_proj"]],
        batch_sizes: List[int] = [1, 2, 4, 8],
        seq_lengths: List[int] = [1024, 2048, 4096],
        num_epochs: int = 3,
        num_steps: int = 50,
        warmup_steps: int = 10
    ):
        """
        Initialize benchmark configuration.
        
        Args:
            model_path: Path to model to benchmark
            output_dir: Directory to save benchmark results
            lora_ranks: List of LoRA ranks to benchmark
            target_modules_options: List of module sets to benchmark
            batch_sizes: List of batch sizes to benchmark
            seq_lengths: List of sequence lengths to benchmark
            num_epochs: Number of epochs to run for each configuration
            num_steps: Number of steps to run for each configuration
            warmup_steps: Number of warmup steps to run before measuring
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.lora_ranks = lora_ranks
        self.target_modules_options = target_modules_options
        self.batch_sizes = batch_sizes
        self.seq_lengths = seq_lengths
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
        
        # Create descriptive names for target modules options
        self.target_modules_names = []
        for modules in target_modules_options:
            if set(modules) == {"q_proj", "v_proj"}:
                self.target_modules_names.append("minimal")
            elif set(modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}:
                self.target_modules_names.append("attention")
            elif "gate_proj" in modules and "up_proj" in modules and "down_proj" in modules:
                self.target_modules_names.append("full")
            else:
                self.target_modules_names.append("custom")


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self):
        """Initialize empty benchmark results."""
        self.results = []
    
    def add_result(
        self,
        lora_r: int,
        target_modules: List[str],
        batch_size: int,
        seq_length: int,
        step_times: List[float],
        memory_usage: Dict[str, Any],
        training_loss: float,
        parameters_count: int
    ):
        """
        Add a benchmark result.
        
        Args:
            lora_r: LoRA rank used
            target_modules: Target modules used
            batch_size: Batch size used
            seq_length: Sequence length used
            step_times: List of step times in seconds
            memory_usage: Dictionary of memory usage statistics
            training_loss: Final training loss
            parameters_count: Number of trainable parameters
        """
        result = {
            "lora_r": lora_r,
            "target_modules": target_modules,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "step_times": step_times,
            "avg_step_time": np.mean(step_times),
            "min_step_time": np.min(step_times),
            "max_step_time": np.max(step_times),
            "std_step_time": np.std(step_times),
            "memory_usage": memory_usage,
            "training_loss": float(training_loss),
            "parameters_count": parameters_count,
            "timestamp": time.time()
        }
        self.results.append(result)
    
    def save(self, output_path: str):
        """
        Save benchmark results to a JSON file.
        
        Args:
            output_path: Path to save results to
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to serializable format
        serializable_results = []
        for result in self.results:
            # Create a copy of the result to avoid modifying the original
            serializable_result = result.copy()
            
            # Convert numpy arrays and values to Python types
            for key, value in serializable_result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.number):
                    serializable_result[key] = value.item()
            
            serializable_results.append(serializable_result)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump({
                "results": serializable_results,
                "metadata": {
                    "timestamp": time.time(),
                    "mlx_version": os.environ.get("MLX_VERSION", "unknown"),
                    "device": self._get_device_info()
                }
            }, f, indent=2)
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device."""
        device_info = {}
        
        try:
            import platform
            device_info["platform"] = platform.platform()
            device_info["processor"] = platform.processor()
        except:
            pass
        
        try:
            # Get macOS-specific info
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                device_info["cpu"] = result.stdout.strip()
                
            # Get memory info
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                try:
                    mem_bytes = int(result.stdout.strip())
                    device_info["memory_gb"] = mem_bytes / (1024**3)
                except:
                    pass
                    
        except:
            pass
            
        return device_info
    
    def generate_report(self, output_path: str):
        """
        Generate a markdown report of benchmark results.
        
        Args:
            output_path: Path to save report to
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get device info
        device_info = self._get_device_info()
        
        # Start writing the report
        with open(output_path, 'w') as f:
            # Header
            f.write("# LoRA Fine-tuning Benchmark Report\n\n")
            
            # Device information
            f.write("## Device Information\n\n")
            for key, value in device_info.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Number of configurations tested**: {len(self.results)}\n")
            f.write(f"- **Test date**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write(f"- **MLX Version**: {os.environ.get('MLX_VERSION', 'unknown')}\n\n")
            
            # Performance comparison table - by LoRA rank
            f.write("## Performance by LoRA Rank\n\n")
            f.write("| LoRA Rank | Avg Step Time (s) | Memory Usage (GB) | Parameters |\n")
            f.write("|-----------|-------------------|------------------|------------|\n")
            
            # Group by LoRA rank
            lora_ranks = set(result["lora_r"] for result in self.results)
            for rank in sorted(lora_ranks):
                rank_results = [r for r in self.results if r["lora_r"] == rank]
                avg_step_time = np.mean([r["avg_step_time"] for r in rank_results])
                avg_memory = np.mean([r["memory_usage"].get("peak_gb", 0) for r in rank_results 
                                     if "peak_gb" in r["memory_usage"]])
                avg_params = np.mean([r["parameters_count"] for r in rank_results])
                
                f.write(f"| {rank} | {avg_step_time:.4f} | {avg_memory:.2f} | {int(avg_params):,} |\n")
            
            f.write("\n")
            
            # Performance comparison table - by target modules
            f.write("## Performance by Target Modules\n\n")
            f.write("| Target Modules | Avg Step Time (s) | Memory Usage (GB) | Parameters |\n")
            f.write("|---------------|-------------------|------------------|------------|\n")
            
            # Group by target modules
            module_sets = set(tuple(sorted(result["target_modules"])) for result in self.results)
            for modules in sorted(module_sets):
                module_results = [r for r in self.results if sorted(r["target_modules"]) == sorted(modules)]
                avg_step_time = np.mean([r["avg_step_time"] for r in module_results])
                avg_memory = np.mean([r["memory_usage"].get("peak_gb", 0) for r in module_results 
                                     if "peak_gb" in r["memory_usage"]])
                avg_params = np.mean([r["parameters_count"] for r in module_results])
                
                module_str = ", ".join(modules)
                f.write(f"| {module_str} | {avg_step_time:.4f} | {avg_memory:.2f} | {int(avg_params):,} |\n")
            
            f.write("\n")
            
            # Performance comparison table - by batch size
            f.write("## Performance by Batch Size\n\n")
            f.write("| Batch Size | Avg Step Time (s) | Memory Usage (GB) |\n")
            f.write("|------------|-------------------|------------------|\n")
            
            # Group by batch size
            batch_sizes = set(result["batch_size"] for result in self.results)
            for size in sorted(batch_sizes):
                size_results = [r for r in self.results if r["batch_size"] == size]
                avg_step_time = np.mean([r["avg_step_time"] for r in size_results])
                avg_memory = np.mean([r["memory_usage"].get("peak_gb", 0) for r in size_results 
                                     if "peak_gb" in r["memory_usage"]])
                
                f.write(f"| {size} | {avg_step_time:.4f} | {avg_memory:.2f} |\n")
            
            f.write("\n")
            
            # Performance comparison table - by sequence length
            f.write("## Performance by Sequence Length\n\n")
            f.write("| Sequence Length | Avg Step Time (s) | Memory Usage (GB) |\n")
            f.write("|-----------------|-------------------|------------------|\n")
            
            # Group by sequence length
            seq_lengths = set(result["seq_length"] for result in self.results)
            for length in sorted(seq_lengths):
                length_results = [r for r in self.results if r["seq_length"] == length]
                avg_step_time = np.mean([r["avg_step_time"] for r in length_results])
                avg_memory = np.mean([r["memory_usage"].get("peak_gb", 0) for r in length_results 
                                     if "peak_gb" in r["memory_usage"]])
                
                f.write(f"| {length} | {avg_step_time:.4f} | {avg_memory:.2f} |\n")
            
            f.write("\n")
            
            # Top 5 fastest configurations
            f.write("## Top 5 Fastest Configurations\n\n")
            f.write("| Rank | Modules | Batch Size | Seq Length | Step Time (s) | Memory (GB) | Parameters |\n")
            f.write("|------|---------|------------|------------|---------------|-------------|------------|\n")
            
            # Sort by step time
            sorted_results = sorted(self.results, key=lambda r: r["avg_step_time"])
            for i, result in enumerate(sorted_results[:5]):
                modules_str = ", ".join(result["target_modules"])
                memory = result["memory_usage"].get("peak_gb", 0)
                f.write(f"| {result['lora_r']} | {modules_str} | {result['batch_size']} | {result['seq_length']} | " +
                       f"{result['avg_step_time']:.4f} | {memory:.2f} | {result['parameters_count']:,} |\n")
            
            f.write("\n")
            
            # Top 5 memory-efficient configurations
            f.write("## Top 5 Memory-Efficient Configurations\n\n")
            f.write("| Rank | Modules | Batch Size | Seq Length | Step Time (s) | Memory (GB) | Parameters |\n")
            f.write("|------|---------|------------|------------|---------------|-------------|------------|\n")
            
            # Sort by memory usage
            memory_results = [r for r in self.results if "peak_gb" in r["memory_usage"]]
            sorted_results = sorted(memory_results, key=lambda r: r["memory_usage"]["peak_gb"])
            for i, result in enumerate(sorted_results[:5]):
                modules_str = ", ".join(result["target_modules"])
                memory = result["memory_usage"].get("peak_gb", 0)
                f.write(f"| {result['lora_r']} | {modules_str} | {result['batch_size']} | {result['seq_length']} | " +
                       f"{result['avg_step_time']:.4f} | {memory:.2f} | {result['parameters_count']:,} |\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Find best performance/memory trade-off
            valid_results = [r for r in self.results if "peak_gb" in r["memory_usage"]]
            if valid_results:
                # Normalize step times and memory usage to 0-1 range
                min_time = min(r["avg_step_time"] for r in valid_results)
                max_time = max(r["avg_step_time"] for r in valid_results)
                time_range = max_time - min_time
                
                min_mem = min(r["memory_usage"]["peak_gb"] for r in valid_results)
                max_mem = max(r["memory_usage"]["peak_gb"] for r in valid_results)
                mem_range = max_mem - min_mem
                
                # Calculate score (lower is better)
                for r in valid_results:
                    norm_time = (r["avg_step_time"] - min_time) / time_range if time_range > 0 else 0
                    norm_mem = (r["memory_usage"]["peak_gb"] - min_mem) / mem_range if mem_range > 0 else 0
                    r["score"] = norm_time * 0.5 + norm_mem * 0.5
                
                # Get best trade-off
                best_result = min(valid_results, key=lambda r: r["score"])
                
                f.write("### Best Overall Configuration\n\n")
                f.write(f"- **LoRA Rank**: {best_result['lora_r']}\n")
                f.write(f"- **Target Modules**: {', '.join(best_result['target_modules'])}\n")
                f.write(f"- **Batch Size**: {best_result['batch_size']}\n")
                f.write(f"- **Sequence Length**: {best_result['seq_length']}\n")
                f.write(f"- **Avg Step Time**: {best_result['avg_step_time']:.4f} seconds\n")
                f.write(f"- **Memory Usage**: {best_result['memory_usage'].get('peak_gb', 0):.2f} GB\n")
                f.write(f"- **Parameters Count**: {best_result['parameters_count']:,}\n\n")
            
            # Best for low memory
            f.write("### Best Configuration for Low Memory\n\n")
            memory_results = [r for r in self.results if "peak_gb" in r["memory_usage"]]
            if memory_results:
                low_mem_result = min(memory_results, key=lambda r: r["memory_usage"]["peak_gb"])
                
                f.write(f"- **LoRA Rank**: {low_mem_result['lora_r']}\n")
                f.write(f"- **Target Modules**: {', '.join(low_mem_result['target_modules'])}\n")
                f.write(f"- **Batch Size**: {low_mem_result['batch_size']}\n")
                f.write(f"- **Sequence Length**: {low_mem_result['seq_length']}\n")
                f.write(f"- **Avg Step Time**: {low_mem_result['avg_step_time']:.4f} seconds\n")
                f.write(f"- **Memory Usage**: {low_mem_result['memory_usage'].get('peak_gb', 0):.2f} GB\n")
                f.write(f"- **Parameters Count**: {low_mem_result['parameters_count']:,}\n\n")
            
            # Best for speed
            f.write("### Best Configuration for Speed\n\n")
            fastest_result = min(self.results, key=lambda r: r["avg_step_time"])
            
            f.write(f"- **LoRA Rank**: {fastest_result['lora_r']}\n")
            f.write(f"- **Target Modules**: {', '.join(fastest_result['target_modules'])}\n")
            f.write(f"- **Batch Size**: {fastest_result['batch_size']}\n")
            f.write(f"- **Sequence Length**: {fastest_result['seq_length']}\n")
            f.write(f"- **Avg Step Time**: {fastest_result['avg_step_time']:.4f} seconds\n")
            f.write(f"- **Memory Usage**: {fastest_result['memory_usage'].get('peak_gb', 0):.2f} GB\n")
            f.write(f"- **Parameters Count**: {fastest_result['parameters_count']:,}\n\n")
            
            # General recommendations
            f.write("### General Recommendations\n\n")
            f.write("1. **LoRA Rank**: Lower rank (4-8) is more memory efficient, while higher rank (16-32) may capture more details but requires more memory and compute.\n")
            f.write("2. **Target Modules**: Targeting only query and value projections (`q_proj`, `v_proj`) provides a good balance of performance and quality.\n")
            f.write("3. **Batch Size**: Larger batch sizes improve throughput but increase memory usage. Find the largest batch size that fits in your memory.\n")
            f.write("4. **Sequence Length**: Shorter sequences allow larger batch sizes, while longer sequences may improve result quality.\n")
            f.write("5. **Memory-Performance Tradeoff**: For memory-constrained devices, use lower LoRA rank (4) and target fewer modules.\n")
            f.write("6. **Apple Silicon Optimization**: Batch size effectiveness varies by Apple chip generation - tune this parameter for your specific device.\n")


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage statistics
    """
    memory_usage = {}
    
    try:
        # Try to get memory info using psutil
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        memory_usage["rss_bytes"] = memory_info.rss
        memory_usage["rss_gb"] = memory_info.rss / (1024**3)
        memory_usage["vms_bytes"] = memory_info.vms
        memory_usage["vms_gb"] = memory_info.vms / (1024**3)
    except (ImportError, AttributeError):
        # Fallback to subprocess on macOS
        try:
            import subprocess
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                rss_kb = int(result.stdout.strip())
                memory_usage["rss_bytes"] = rss_kb * 1024
                memory_usage["rss_gb"] = rss_kb / (1024**2)
        except:
            pass
    
    return memory_usage


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """
    Run LoRA fine-tuning benchmark with various configurations.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Benchmark results
    """
    logger = setup_logger("lora_benchmark")
    results = BenchmarkResults()
    
    # Check for MLX
    if not HAS_MLX:
        logger.error("MLX is required for LoRA benchmarking")
        return results
    
    # Get MLX version for the results
    try:
        import mlx
        os.environ["MLX_VERSION"] = getattr(mlx, "__version__", "unknown")
    except (ImportError, AttributeError):
        os.environ["MLX_VERSION"] = "unknown"
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create synthetic data for benchmarking
    def create_synthetic_batch(batch_size, seq_len):
        """Create synthetic batch for benchmarking."""
        return {
            "input_tokens": mx.zeros((batch_size, seq_len, 3), dtype=mx.int32),
            "input_masks": mx.ones((batch_size, seq_len, 3), dtype=mx.float32),
            "target_audio_tokens": mx.zeros((batch_size, seq_len, 2), dtype=mx.int32)
        }
    
    # Run benchmarks for each configuration
    total_configs = (
        len(config.lora_ranks) *
        len(config.target_modules_options) *
        len(config.batch_sizes) *
        len(config.seq_lengths)
    )
    
    logger.info(f"Running {total_configs} benchmark configurations")
    
    config_idx = 0
    
    # For each LoRA rank
    for lora_r in config.lora_ranks:
        lora_alpha = lora_r * 2  # Common practice: alpha = 2 * r
        
        # For each target modules option
        for modules_idx, target_modules in enumerate(config.target_modules_options):
            target_name = config.target_modules_names[modules_idx]
            
            # For each batch size
            for batch_size in config.batch_sizes:
                
                # For each sequence length
                for seq_len in config.seq_lengths:
                    config_idx += 1
                    logger.info(f"Running configuration {config_idx}/{total_configs}:")
                    logger.info(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}")
                    logger.info(f"  Target modules: {target_modules} ({target_name})")
                    logger.info(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
                    
                    try:
                        # Create output directory for this configuration
                        config_dir = config.output_dir / f"r{lora_r}_{target_name}_bs{batch_size}_seq{seq_len}"
                        os.makedirs(config_dir, exist_ok=True)
                        
                        # Initialize trainer for this configuration
                        trainer = CSMLoRATrainer(
                            model_path=config.model_path,
                            output_dir=str(config_dir),
                            lora_r=lora_r,
                            lora_alpha=lora_alpha,
                            target_modules=target_modules
                        )
                        
                        # Prepare optimizer
                        trainer.prepare_optimizer()
                        
                        # Get parameter count
                        parameters_count = 0
                        if hasattr(trainer.model, "get_lora_params"):
                            lora_params = trainer.model.get_lora_params()
                            parameters_count = sum(np.prod(p.shape) for p in lora_params.values())
                        
                        # Create synthetic batch
                        batch = create_synthetic_batch(batch_size, seq_len)
                        
                        # Run warmup steps
                        logger.info(f"Running {config.warmup_steps} warmup steps")
                        for _ in range(config.warmup_steps):
                            trainer.train_step(batch)
                        
                        # Ensure warmup steps are evaluated
                        mx.eval(mx.array(0.0))
                        
                        # Run benchmark steps
                        step_times = []
                        peak_memory = 0
                        
                        logger.info(f"Running {config.num_steps} benchmark steps")
                        for step in range(config.num_steps):
                            # Record memory before step
                            pre_mem = get_memory_usage()
                            
                            # Run step and time it
                            start_time = time.time()
                            loss = trainer.train_step(batch)
                            
                            # Ensure evaluation happens (MLX is lazy)
                            mx.eval(mx.array(0.0))
                            
                            # Record time
                            step_time = time.time() - start_time
                            step_times.append(step_time)
                            
                            # Record memory after step
                            post_mem = get_memory_usage()
                            if "rss_gb" in post_mem:
                                peak_memory = max(peak_memory, post_mem["rss_gb"])
                            
                            # Log progress
                            if (step + 1) % 10 == 0 or step == 0:
                                logger.info(f"  Step {step + 1}/{config.num_steps}: {step_time:.4f}s")
                        
                        # Record final memory usage
                        memory_usage = get_memory_usage()
                        memory_usage["peak_gb"] = peak_memory
                        
                        # Record results
                        results.add_result(
                            lora_r=lora_r,
                            target_modules=target_modules,
                            batch_size=batch_size,
                            seq_length=seq_len,
                            step_times=step_times,
                            memory_usage=memory_usage,
                            training_loss=float(loss.item()) if hasattr(loss, 'item') else float(loss),
                            parameters_count=parameters_count
                        )
                        
                        logger.info(f"Configuration completed successfully")
                        logger.info(f"  Avg step time: {np.mean(step_times):.4f}s")
                        logger.info(f"  Peak memory: {peak_memory:.2f} GB")
                        logger.info(f"  Parameters: {parameters_count:,}")
                        
                    except Exception as e:
                        logger.error(f"Error running configuration: {e}")
                        logger.exception(e)
    
    return results


def main():
    """Run the LoRA benchmark as a standalone script."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark LoRA fine-tuning with MLX")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model to benchmark")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--lora-ranks", type=int, nargs="+", default=[4, 8, 16, 32],
                        help="LoRA ranks to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[1024, 2048, 4096],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of steps to run for each configuration")
    parser.add_argument("--warmup-steps", type=int, default=10,
                        help="Number of warmup steps to run before measuring")
    parser.add_argument("--minimal-only", action="store_true",
                        help="Only benchmark minimal target modules (q_proj, v_proj)")
    
    args = parser.parse_args()
    
    # Configure target modules options
    if args.minimal_only:
        target_modules_options = [["q_proj", "v_proj"]]
    else:
        target_modules_options = [
            ["q_proj", "v_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj"]
        ]
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        lora_ranks=args.lora_ranks,
        target_modules_options=target_modules_options,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps
    )
    
    # Setup logger
    logger = setup_logger("lora_benchmark", 
                         log_file=os.path.join(args.output_dir, "benchmark.log"))
    logger.info("Starting LoRA benchmark")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Run benchmark
    results = run_benchmark(config)
    
    # Save results
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    results.save(results_path)
    logger.info(f"Saved raw benchmark results to {results_path}")
    
    # Generate report
    report_path = os.path.join(args.output_dir, "benchmark_report.md")
    results.generate_report(report_path)
    logger.info(f"Generated benchmark report at {report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())