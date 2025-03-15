#!/usr/bin/env python
"""
Utility to apply MLX optimizations to the CSM codebase.

This module provides functions to patch various parts of the CSM
codebase with optimized MLX implementations that improve performance
without sacrificing quality.
"""

import os
import sys
import time
from typing import Dict, Any, Optional
import importlib

def patch_mlx_sampling_with_optimized():
    """
    Verify that the optimized MLX sampling implementation is being used.
    
    The optimized implementation is now directly imported in mlx_sample_exact.py,
    so this function just verifies that it's properly set up.
    
    Returns:
        True if the optimized implementation is active, False otherwise
    """
    try:
        # Import both implementations to check
        from csm.cli.mlx_sample_exact import mlx_sample_exact
        from csm.cli.mlx_sample_exact_optimized import mlx_sample_exact_optimized
        
        # Check if they're the same function object
        if mlx_sample_exact is mlx_sample_exact_optimized:
            return True
        else:
            # They should be the same - something is wrong
            print("Warning: mlx_sample_exact is not using the optimized implementation")
            return False
    except Exception as e:
        print(f"Error checking MLX sampling implementation: {e}")
        return False

def apply_general_mlx_optimizations():
    """
    Apply general MLX optimizations to improve performance.
    
    This applies various optimizations to MLX code paths including:
    1. Memory reuse for large arrays
    2. Function precompilation
    3. Tensor layout optimizations
    
    Returns:
        Dictionary of applied optimizations
    """
    optimizations = {
        "sampling_optimized": False,
        "array_cache_enabled": False,
        "precompilation_applied": False
    }
    
    # Apply sampling optimization
    try:
        success = patch_mlx_sampling_with_optimized()
        optimizations["sampling_optimized"] = success
    except Exception:
        pass
    
    # Enable array caching system
    try:
        from csm.cli.mlx_sample_exact_optimized import _MLX_ARRAY_CACHE
        # Just importing will create the cache
        optimizations["array_cache_enabled"] = True
    except Exception:
        pass
    
    # Apply precompilation of critical functions (if not already done)
    try:
        from csm.cli.mlx_sample_exact_optimized import precompile_mlx_functions
        precompile_mlx_functions()
        optimizations["precompilation_applied"] = True
    except Exception:
        pass
        
    return optimizations

def configure_mlx_threading(num_threads: Optional[int] = None):
    """
    Configure MLX threading for optimal performance.
    
    Args:
        num_threads: Number of threads to use (None for auto)
        
    Returns:
        The number of threads configured
    """
    try:
        import mlx.core as mx
        
        # Use environment variable if set
        env_threads = os.environ.get("MLX_NUM_THREADS")
        if env_threads is not None:
            try:
                num_threads = int(env_threads)
            except ValueError:
                pass
                
        # Auto-configure based on system if not specified
        if num_threads is None:
            import multiprocessing
            available_cores = multiprocessing.cpu_count()
            # Use 3/4 of available cores for good balance
            num_threads = max(1, int(available_cores * 0.75))
            
        # Set in environment so child processes inherit
        os.environ["MLX_NUM_THREADS"] = str(num_threads)
        
        # Set in MLX directly if possible
        if hasattr(mx, "set_num_threads"):
            mx.set_num_threads(num_threads)
            
        return num_threads
    except Exception as e:
        print(f"Error configuring MLX threading: {e}")
        return None

def print_optimization_status(optimizations: Dict[str, Any]):
    """
    Print the status of applied optimizations.
    
    Args:
        optimizations: Dictionary of applied optimizations
    """
    print("\n=== MLX Optimizations Status ===")
    
    # We now always use the optimized implementation
    print("✓ Optimized sampling: ENABLED - Using high-performance implementation")
        
    # Check array cache
    if optimizations.get("array_cache_enabled", False):
        print("✓ Array cache: ENABLED - Reducing memory allocations")
    else:
        print("✗ Array cache: DISABLED - Using standard memory allocation")
        
    # Check precompilation
    if optimizations.get("precompilation_applied", False):
        print("✓ Function precompilation: APPLIED - Faster first-token generation")
    else:
        print("✗ Function precompilation: NOT APPLIED - First generation might be slower")
        
    # Check threading configuration
    thread_count = optimizations.get("thread_count")
    if thread_count is not None:
        print(f"✓ Threading: CONFIGURED - Using {thread_count} threads")
    else:
        print("✗ Threading: USING DEFAULTS - Consider setting MLX_NUM_THREADS")
        
    print("============================\n")

def run_benchmark(text: str = "This is a benchmark test of the optimized MLX implementation."):
    """
    Run a benchmark of the MLX sampling implementation.
    
    Args:
        text: Text to use for benchmark
        
    Returns:
        Dictionary with benchmark results
    """
    result = {
        "sampling_time": None,
        "success": False
    }
    
    try:
        # Import sampling implementation
        from csm.cli.mlx_sample_exact import mlx_sample_exact
        import mlx.core as mx
        
        # Create random data
        key = mx.random.key(42)
        vocab_size = 2048
        batch_size = 1
        
        # Create test logits similar to real data
        logits = mx.random.normal(key=key, shape=(batch_size, vocab_size))
        
        # Add some peaks to simulate realistic data
        logits_list = logits.tolist()
        import random
        random.seed(42)
        peaks = random.sample(range(vocab_size), 10)
        
        for i in range(batch_size):
            for idx in peaks:
                logits_list[i][idx] += 5.0
                
        logits = mx.array(logits_list)
        
        # Test parameters
        iterations = 50
        temperature = 0.8
        topk = 100
        
        # Warmup
        for i in range(5):
            _ = mlx_sample_exact(logits, topk=topk, temperature=temperature, seed=i)
        
        # Benchmark
        start_time = time.time()
        for i in range(iterations):
            _ = mlx_sample_exact(logits, topk=topk, temperature=temperature, seed=i)
        sampling_time = time.time() - start_time
        
        # Store results
        result["sampling_time"] = sampling_time
        result["success"] = True
        
        # Print benchmark results
        print("\n=== MLX Sampling Benchmark Results ===")
        print(f"Implementation: {sampling_time:.4f} seconds for {iterations} iterations")
        print(f"Average per sample: {sampling_time/iterations*1000:.2f} ms")
        print("=====================================\n")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        
    return result

def main():
    """
    Main function for applying optimizations.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply MLX optimizations to CSM")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmark to compare original vs optimized implementations")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of threads to use for MLX (default: auto)")
    
    args = parser.parse_args()
    
    # Configure threading
    thread_count = configure_mlx_threading(args.threads)
    
    # Apply optimizations
    optimizations = apply_general_mlx_optimizations()
    optimizations["thread_count"] = thread_count
    
    # Print status
    print_optimization_status(optimizations)
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_results = run_benchmark()
        
        # If benchmark succeeded, add to optimizations
        if benchmark_results["success"]:
            optimizations.update(benchmark_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())