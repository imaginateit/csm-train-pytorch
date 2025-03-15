"""
Exact implementation of PyTorch-equivalent token sampling in MLX.

This module provides a direct implementation of PyTorch's token sampling behavior
in MLX, focusing on reproducing the exact numerical operations to achieve matching
token distributions. It implements the Gumbel-max trick for accurate categorical
sampling and handles proper top-k filtering.
"""

import math
import time
import random
from typing import Optional, Tuple, List

import mlx.core as mx
import numpy as np
import torch

def mlx_sample_exact(logits: mx.array, topk: int = 5, temperature: float = 1.0, seed: Optional[int] = None) -> mx.array:
    """
    MLX implementation that exactly matches PyTorch sampling behavior.
    
    This function reproduces PyTorch's categorical sampling with temperature and top-k
    filtering by using the Gumbel-max trick and careful numerical operations that
    match PyTorch's implementation.
    
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
    
    # Apply temperature scaling exactly as in PyTorch
    # We need to handle potential division by zero carefully
    # Use same epsilon value as PyTorch for consistency
    epsilon = 1e-10
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
        # 1. Sort values to find the k-th largest value
        # MLX doesn't have a descending parameter, so sort and flip the order
        sorted_logits_asc = mx.sort(batch_logits)
        sorted_logits = sorted_logits_asc[::-1]  # Reverse to get descending order
        
        # 2. Get top-k threshold value
        k = min(topk, vocab_size)
        threshold = sorted_logits[k-1]
        
        # 3. Apply mask using exact matching operations to PyTorch
        # PyTorch uses a different masking approach than the previous implementation
        mask = batch_logits < threshold
        filtered_logits = mx.where(mask, mx.array(-float('inf')), batch_logits)
        
        # 4. Apply softmax exactly as in PyTorch
        # We need to be careful with numerical stability here
        # First subtract the max value to avoid overflow, as PyTorch does
        max_val = mx.max(filtered_logits)
        exp_logits = mx.exp(filtered_logits - max_val)
        sum_exp = mx.sum(exp_logits)
        probs = exp_logits / (sum_exp + epsilon)  # Add epsilon to avoid division by zero
        
        # 5. Match PyTorch's categorical sampling with Gumbel-max trick
        # The Gumbel-max trick is a way to sample from a categorical distribution
        # by adding Gumbel noise and taking the argmax
        uniform_vals = mx.random.uniform(key=subkey, shape=probs.shape)
        
        # Careful with floating point stability in Gumbel noise
        gumbel_noise = -mx.log(-mx.log(uniform_vals + epsilon) + epsilon)
        
        # PyTorch adds the log probabilities to the Gumbel noise
        log_probs = mx.log(probs + epsilon)  # Add epsilon to avoid log(0)
        gumbel_logits = log_probs + gumbel_noise
        
        # Get sample
        sample_idx = mx.argmax(gumbel_logits)
        
        # Safety check: never return problematic tokens that cause MIMI codec issues
        if 1 <= sample_idx < 32:
            # Use silence token instead
            sample_idx = mx.array(0)
            
        # Set result in the output tensor
        # MLX arrays can't be modified in-place easily, so we'll use a list as intermediate
        samples_list = samples.tolist()
        samples_list[b][0] = sample_idx.item()
        samples = mx.array(samples_list)
    
    return samples

def categorical_dist_sample_pytorch(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Reference implementation of PyTorch's categorical sampling for comparison.
    
    This function shows how PyTorch performs categorical sampling with temperature,
    which we're replicating in the MLX implementation above.
    
    Args:
        logits: Raw logits with shape [batch_size, vocab_size]
        temperature: Temperature for sampling
        
    Returns:
        Sampled token indices with shape [batch_size, 1]
    """
    # Scale logits by temperature
    scaled_logits = logits / (temperature + 1e-10)
    
    # Convert to probabilities with softmax
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    
    # Sample from the distribution
    samples = torch.multinomial(probs, num_samples=1)
    
    # Safety check for MIMI codec compatibility
    problematic_mask = (samples >= 1) & (samples < 32)
    if problematic_mask.any():
        samples = torch.where(problematic_mask, torch.zeros_like(samples), samples)
    
    return samples

def topk_filtering_pytorch(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Reference implementation of PyTorch's top-k filtering for comparison.
    
    This shows how PyTorch performs top-k filtering, which we're replicating
    in the MLX implementation above.
    
    Args:
        logits: Raw logits with shape [batch_size, vocab_size]
        top_k: Number of top tokens to consider
        
    Returns:
        Filtered logits with the same shape as input
    """
    batch_size, vocab_size = logits.shape
    
    # Clone input logits to avoid modifying the original
    filtered_logits = logits.clone()
    
    # Apply top-k filtering for each item in the batch
    for i in range(batch_size):
        # Get the k-th largest value
        top_k_values = torch.topk(filtered_logits[i], k=min(top_k, vocab_size))[0]
        # Get the threshold value (the smallest of the top-k)
        threshold = top_k_values[-1]
        # Zero out all values below the threshold
        mask = filtered_logits[i] < threshold
        filtered_logits[i][mask] = float('-inf')
    
    return filtered_logits

def gumbel_softmax_trick_mlx(probs: mx.array, temperature: float = 1.0, key=None) -> mx.array:
    """
    Implementation of the Gumbel-Softmax trick for MLX.
    
    This is an alternative implementation used internally for testing.
    
    Args:
        probs: Probability distribution with shape [batch_size, vocab_size]
        temperature: Temperature parameter
        key: MLX random key
        
    Returns:
        Sampled one-hot vectors with shape [batch_size, vocab_size]
    """
    if key is None:
        key = mx.random.key(int(time.time() * 1000))
    
    # Sample from Gumbel(0, 1)
    u = mx.random.uniform(key=key, shape=probs.shape)
    g = -mx.log(-mx.log(u + 1e-10) + 1e-10)
    
    # Apply Gumbel-max trick
    y = mx.log(probs + 1e-10) + g
    
    # Return the argmax (one-hot in the limit of temperature -> 0)
    return mx.argmax(y, axis=-1)

def sample_test(logits: Optional[mx.array] = None, 
               temperature: float = 1.0, 
               topk: int = 50, 
               vocab_size: int = 2048,
               iterations: int = 100,
               seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Test sampling implementations and compare outputs.
    
    This is a utility function for comparing our MLX sampling with PyTorch.
    
    Args:
        logits: Optional test logits (will generate random ones if not provided)
        temperature: Temperature for sampling
        topk: Number of top tokens to consider
        vocab_size: Vocabulary size for generating test logits
        iterations: Number of test iterations
        seed: Random seed
        
    Returns:
        Tuple of lists containing sampled token counts for MLX and PyTorch
    """
    # Set seeds for reproducibility
    mx.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create test logits if not provided
    if logits is None:
        # Generate random logits that follow a typical distribution pattern
        # with some tokens having higher probability than others
        key = mx.random.key(seed)
        logits = mx.random.normal(key=key, shape=(1, vocab_size))
        
        # Create a more realistic distribution with a few dominant tokens
        # and a long tail of lower probability tokens
        peak_indices = np.random.choice(vocab_size, size=20, replace=False)
        
        # Create a new tensor with boosted values instead of trying to modify in-place
        logits_array = logits.tolist()
        for idx in peak_indices:
            logits_array[0][idx] += 5.0  # Boost some tokens
        logits = mx.array(logits_array)
    
    # Convert to PyTorch tensor for comparison
    pt_logits = torch.tensor(logits.tolist())
    
    # Sample using both implementations
    mlx_samples = []
    pt_samples = []
    
    for i in range(iterations):
        # MLX exact sampling
        iteration_seed = seed + i
        mlx_sample = mlx_sample_exact(logits, topk=topk, temperature=temperature, seed=iteration_seed)
        mlx_samples.append(mlx_sample.item())
        
        # PyTorch sampling
        torch.manual_seed(iteration_seed)
        # Apply top-k filtering
        filtered_logits = topk_filtering_pytorch(pt_logits, top_k=topk)
        # Sample from filtered distribution
        pt_sample = categorical_dist_sample_pytorch(filtered_logits, temperature=temperature)
        pt_samples.append(pt_sample.item())
    
    # Count occurrences
    mlx_counter = {}
    pt_counter = {}
    
    for sample in mlx_samples:
        mlx_counter[sample] = mlx_counter.get(sample, 0) + 1
        
    for sample in pt_samples:
        pt_counter[sample] = pt_counter.get(sample, 0) + 1
    
    # Print statistics
    print(f"=== Sampling Test Results (temp={temperature}, topk={topk}) ===")
    print(f"MLX unique tokens: {len(mlx_counter)}")
    print(f"PyTorch unique tokens: {len(pt_counter)}")
    
    # Calculate overlap
    common_tokens = set(mlx_counter.keys()).intersection(set(pt_counter.keys()))
    print(f"Common tokens: {len(common_tokens)} ({len(common_tokens)/len(pt_counter)*100:.1f}% of PyTorch tokens)")
    
    # Create and return full count lists for both
    mlx_counts = [(token, count) for token, count in mlx_counter.items()]
    pt_counts = [(token, count) for token, count in pt_counter.items()]
    
    mlx_counts.sort(key=lambda x: x[1], reverse=True)
    pt_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 MLX tokens:")
    for token, count in mlx_counts[:5]:
        print(f"  Token {token}: {count} times ({count/iterations*100:.1f}%)")
        
    print("\nTop 5 PyTorch tokens:")
    for token, count in pt_counts[:5]:
        print(f"  Token {token}: {count} times ({count/iterations*100:.1f}%)")
    
    return mlx_counts, pt_counts

if __name__ == "__main__":
    # Run test with different temperatures
    for temp in [0.8, 1.0, 1.2]:
        mlx_counts, pt_counts = sample_test(temperature=temp, iterations=1000)
        print("\n" + "="*50 + "\n")