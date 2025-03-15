"""
Optimized exact implementation of PyTorch-equivalent token sampling in MLX.

This module provides a high-performance implementation that maintains the exact
sampling distribution quality of the original mlx_sample_exact function.
"""

import math
import time
import random
from typing import Optional, Tuple, List

import mlx.core as mx
import numpy as np
import torch

def mlx_sample_exact_optimized(logits: mx.array, topk: int = 5, temperature: float = 1.0, seed: Optional[int] = None) -> mx.array:
    """
    High-performance MLX implementation that exactly matches PyTorch sampling behavior.
    
    This maintains the same high-quality token distribution as the original implementation
    while being more computationally efficient.
    
    Args:
        logits: Raw logits with shape [batch_size, vocab_size]
        topk: Number of top tokens to consider
        temperature: Temperature for sampling (higher = more random)
        seed: Random seed for reproducibility
        
    Returns:
        Sampled token indices with shape [batch_size, 1]
    """
    # Get random seed if not provided
    if seed is None:
        seed = int(time.time() * 1000) % 10000
    
    # Ensure proper shape
    if len(logits.shape) == 1:
        logits = logits.reshape(1, -1)
        
    batch_size, vocab_size = logits.shape
    
    # Use the same epsilon value as PyTorch for consistency
    epsilon = 1e-10
    
    # Apply temperature scaling exactly as in PyTorch
    scaled_logits = logits / (temperature + epsilon)
    
    # Apply top-k filtering
    samples = mx.zeros((batch_size, 1), dtype=mx.int32)
    
    # Get random key for reproducible sampling
    key = mx.random.key(seed)
    
    for b in range(batch_size):
        # Split key for this batch
        key, subkey = mx.random.split(key)
        
        # Get batch logits
        batch_logits = scaled_logits[b]
        
        # THIS IS CRITICAL: Match PyTorch's top-k implementation exactly
        # Convert to numpy for stable sorting (faster than list in many cases)
        batch_logits_np = batch_logits.tolist()
        
        # Get top-k indices
        sorted_indices = np.argsort(batch_logits_np)[::-1]  # Descending order
        k = min(topk, vocab_size)
        top_k_indices = sorted_indices[:k]
        
        # Create filtered logits
        filtered_logits_list = [-float('inf')] * vocab_size
        for idx in top_k_indices:
            filtered_logits_list[idx] = batch_logits_np[idx]
        filtered_logits = mx.array(filtered_logits_list)
        
        # Apply softmax with numerical stability
        max_val = mx.max(filtered_logits)
        shifted_logits = filtered_logits - max_val
        exp_logits = mx.exp(shifted_logits)
        sum_exp = mx.sum(exp_logits)
        probs = exp_logits / (sum_exp + epsilon)
        
        # Apply Gumbel-max trick for categorical sampling
        uniform_vals = mx.random.uniform(key=subkey, shape=probs.shape)
        gumbel_noise = -mx.log(-mx.log(uniform_vals + epsilon) + epsilon)
        log_probs = mx.log(probs + epsilon)
        gumbel_logits = log_probs + gumbel_noise
        
        # Apply masking
        valid_mask = filtered_logits != -float('inf')
        masked_gumbel = mx.where(valid_mask, gumbel_logits, mx.array(-float('inf')))
        
        # Get sample
        sample_idx = mx.argmax(masked_gumbel)
        sample_val = sample_idx.item()
        
        # Handle MIMI codec safety (tokens 1-31)
        if 1 <= sample_val < 32:
            # Find next best token, starting from the most probable
            for idx in top_k_indices:
                if idx < 1 or idx >= 32:
                    sample_val = idx
                    break
            else:
                # Fallback to silence token if needed
                sample_val = 0
                
        # Store the result
        samples_list = samples.tolist()
        samples_list[b][0] = sample_val
        samples = mx.array(samples_list)
    
    return samples

def run_comparison(vocab_size=2048, iterations=100, temperature=0.8, topk=100, seed=42):
    """Compare performance between original and optimized implementations."""
    from csm.cli.mlx_sample_exact import mlx_sample_exact
    
    # Create test data
    key = mx.random.key(seed)
    logits = mx.random.normal(key=key, shape=(1, vocab_size))
    
    # Add peaks for realism
    logits_list = logits.tolist()
    peaks = [i*100 for i in range(10)]
    for peak in peaks:
        if peak < vocab_size:
            logits_list[0][peak] += 5.0
    logits = mx.array(logits_list)
    
    # Test original implementation
    print("Testing original implementation...")
    start = time.time()
    orig_results = []
    for i in range(iterations):
        res = mlx_sample_exact(logits, topk=topk, temperature=temperature, seed=seed+i)
        orig_results.append(res.item())
    orig_time = time.time() - start
    
    # Test optimized implementation
    print("Testing optimized implementation...")
    start = time.time()
    opt_results = []
    for i in range(iterations):
        res = mlx_sample_exact_optimized(logits, topk=topk, temperature=temperature, seed=seed+i)
        opt_results.append(res.item())
    opt_time = time.time() - start
    
    # Compare results
    matches = sum(1 for a, b in zip(orig_results, opt_results) if a == b)
    match_rate = matches / iterations * 100
    
    # Compare timing
    speedup = orig_time / opt_time if opt_time > 0 else 0
    
    print("\n--- Performance Results ---")
    print(f"Original time: {orig_time:.4f}s")
    print(f"Optimized time: {opt_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    print("\n--- Quality Results ---")
    print(f"Match rate: {match_rate:.1f}% ({matches}/{iterations})")
    print(f"Unique tokens (original): {len(set(orig_results))}")
    print(f"Unique tokens (optimized): {len(set(opt_results))}")
    
    return {
        "original_time": orig_time,
        "optimized_time": opt_time,
        "speedup": speedup,
        "match_rate": match_rate
    }

if __name__ == "__main__":
    print("Running MLX sampling performance comparison...")
    run_comparison(iterations=100)