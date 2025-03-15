#!/usr/bin/env python
"""
Simple test for the MLX sampling function.
"""

import torch
import mlx.core as mx
import numpy as np
import time
import random
from collections import Counter

from csm.cli.mlx_embedding import mlx_sample_topk

def test_mlx_sampling():
    """Test the MLX sampling function with controlled inputs."""
    # Create a batch of logits
    batch_size = 1
    vocab_size = 2051
    temperature = 1.2
    topk = 50
    
    # Create deterministic logits
    np.random.seed(42)
    logits_np = np.random.randn(batch_size, vocab_size).astype(np.float32)
    
    # Make some tokens more likely (higher logits)
    # Ensure our preferred "safe" tokens have higher probability
    safe_tokens = [0, 42, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for token in safe_tokens:
        logits_np[:, token] += 5.0
    
    # Make problematic tokens less likely (lower logits)
    for token in range(1, 32):
        logits_np[:, token] -= 10.0
    
    # Convert to MLX array
    logits_mx = mx.array(logits_np)
    
    # Also convert to PyTorch for comparison
    logits_pt = torch.tensor(logits_np)
    
    # Run the MLX sampling function
    print("Testing MLX sampling...")
    
    # Sample many times to get distribution
    num_samples = 1000
    mlx_samples = []
    start_time = time.time()
    
    # Create new random key each time to ensure variety
    for i in range(num_samples):
        # Use a different random seed for each sample
        random_seed = 42 + i
        sample = mlx_sample_topk(logits_mx, topk=topk, temperature=temperature, seed=random_seed)
        mlx_samples.append(sample.item())
    
    end_time = time.time()
    
    # Analyze results
    mlx_counter = Counter(mlx_samples)
    most_common = mlx_counter.most_common(10)
    
    print(f"MLX sampling took {end_time - start_time:.4f} seconds for {num_samples} samples")
    print(f"\nMost common tokens:")
    for token, count in most_common:
        print(f"  Token {token}: {count} times ({count/num_samples*100:.2f}%)")
    
    # Check for problematic tokens
    problematic = [token for token in mlx_samples if 0 < token < 32]
    if problematic:
        print(f"\nWARNING: Found {len(problematic)} problematic tokens (1-31)")
        for token in sorted(set(problematic)):
            count = problematic.count(token)
            print(f"  Token {token}: {count} times ({count/num_samples*100:.2f}%)")
    else:
        print("\nNo problematic tokens found - sampling is working correctly!")
    
    # Check if safe tokens are being selected
    safe_counts = sum(mlx_counter.get(token, 0) for token in safe_tokens)
    print(f"\nSafe tokens selected: {safe_counts} times ({safe_counts/num_samples*100:.2f}%)")
    
    # Distribution uniformity check
    unique_tokens = len(mlx_counter)
    print(f"\nUnique tokens sampled: {unique_tokens} out of {vocab_size} possible")

if __name__ == "__main__":
    test_mlx_sampling()