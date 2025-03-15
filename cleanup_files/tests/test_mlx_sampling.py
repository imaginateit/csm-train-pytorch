#!/usr/bin/env python
"""
Test script to compare MLX sampling approaches directly.

This script tests both the standard MLX sampling and the exact MLX sampling
implementations to verify their behavior and demonstrate the quality improvement.
"""

import sys
import os
import time
import mlx.core as mx
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from csm.cli.mlx_sample_exact import mlx_sample_exact
from csm.cli.mlx_embedding import mlx_sample_topk, mlx_sample_categorical

def create_test_logits(vocab_size=2048, num_peaks=50, peak_boost=5.0, seed=42):
    """Create realistic test logits with some dominant tokens."""
    # Set random seeds
    mx.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create normal distribution of logits
    key = mx.random.key(seed)
    logits = mx.random.normal(key=key, shape=(1, vocab_size))
    
    # Boost some tokens to create peaks
    logits_array = logits.tolist()
    peak_indices = np.random.choice(vocab_size, size=num_peaks, replace=False)
    for idx in peak_indices:
        logits_array[0][idx] += peak_boost
    
    return mx.array(logits_array)
    
def visualize_sampling(logits, samples_dict, title="Sampling Comparison", 
                      filename="sampling_comparison.png"):
    """Visualize the sampling distributions."""
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot token distributions
    plt.subplot(2, 1, 1)
    
    # Create histogram data
    bins = 50
    for name, samples in samples_dict.items():
        plt.hist(samples, bins=bins, alpha=0.5, label=name)
    
    plt.title(f"{title} - Token Distribution")
    plt.xlabel("Token Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot token counts
    plt.subplot(2, 1, 2)
    bar_width = 0.35
    index = np.arange(5)
    
    # Get top 5 tokens for each method
    for i, (name, samples) in enumerate(samples_dict.items()):
        counter = Counter(samples)
        top_tokens = counter.most_common(5)
        
        # Extract values for plotting
        tokens = [t for t, _ in top_tokens]
        counts = [c for _, c in top_tokens]
        
        # Plot bars
        plt.bar(index + i * bar_width, counts, bar_width, label=name)
        
        # Add token labels
        for j, token in enumerate(tokens):
            plt.text(j + i * bar_width, counts[j] + 0.5, str(token), 
                    ha='center', va='bottom', rotation=0)
    
    plt.title(f"{title} - Top 5 Tokens")
    plt.xlabel("Rank")
    plt.ylabel("Count")
    plt.xticks(index + bar_width / 2, [f"#{i+1}" for i in range(5)])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")

def run_comparison(vocab_size=2048, iterations=1000, temperature=0.9, topk=50, seed=42):
    """Compare different sampling approaches."""
    print(f"Running sampling comparison with {iterations} iterations")
    print(f"Parameters: vocab_size={vocab_size}, temp={temperature}, topk={topk}, seed={seed}")
    
    # Create test logits
    logits = create_test_logits(vocab_size=vocab_size, seed=seed)
    
    # Convert to PyTorch for comparison
    pt_logits = torch.tensor(logits.tolist())
    
    # Store samples
    standard_samples = []
    exact_samples = []
    categorical_samples = []
    pytorch_samples = []
    
    # Run sampling
    for i in range(iterations):
        # Set iteration seed
        iteration_seed = seed + i
        
        # Standard MLX sampling
        standard_sample = mlx_sample_topk(
            logits, topk=topk, temperature=temperature, seed=iteration_seed
        ).item()
        standard_samples.append(standard_sample)
        
        # Exact MLX sampling
        exact_sample = mlx_sample_exact(
            logits, topk=topk, temperature=temperature, seed=iteration_seed
        ).item()
        exact_samples.append(exact_sample)
        
        # Categorical MLX sampling
        categorical_sample = mlx_sample_categorical(
            logits, temperature=temperature, seed=iteration_seed
        ).item()
        categorical_samples.append(categorical_sample)
        
        # PyTorch sampling
        torch.manual_seed(iteration_seed)
        # Apply scaling
        scaled_logits = pt_logits / temperature
        # Convert to probabilities
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        # Sample from distribution
        pt_sample = torch.multinomial(probs, num_samples=1).item()
        pytorch_samples.append(pt_sample)
    
    # Count unique tokens
    unique_standard = len(set(standard_samples))
    unique_exact = len(set(exact_samples))
    unique_categorical = len(set(categorical_samples))
    unique_pytorch = len(set(pytorch_samples))
    
    # Print stats
    print(f"\nUnique tokens:")
    print(f"  Standard MLX: {unique_standard}")
    print(f"  Exact MLX: {unique_exact}")
    print(f"  Categorical MLX: {unique_categorical}")
    print(f"  PyTorch: {unique_pytorch}")
    
    # Calculate overlap with PyTorch
    overlap_standard = len(set(standard_samples).intersection(set(pytorch_samples)))
    overlap_exact = len(set(exact_samples).intersection(set(pytorch_samples)))
    overlap_categorical = len(set(categorical_samples).intersection(set(pytorch_samples)))
    
    print(f"\nOverlap with PyTorch:")
    print(f"  Standard MLX: {overlap_standard}/{unique_pytorch} ({overlap_standard/unique_pytorch*100:.1f}%)")
    print(f"  Exact MLX: {overlap_exact}/{unique_pytorch} ({overlap_exact/unique_pytorch*100:.1f}%)")
    print(f"  Categorical MLX: {overlap_categorical}/{unique_pytorch} ({overlap_categorical/unique_pytorch*100:.1f}%)")
    
    # Calculate distribution similarity
    def distribution_similarity(dist1, dist2):
        counter1 = Counter(dist1)
        counter2 = Counter(dist2)
        
        # Calculate total probability mass
        total1 = len(dist1)
        total2 = len(dist2)
        
        # Find common tokens
        common = set(counter1.keys()).intersection(set(counter2.keys()))
        
        # Calculate similarity (overlap of probability mass)
        similarity = 0
        for token in common:
            prob1 = counter1[token] / total1
            prob2 = counter2[token] / total2
            similarity += min(prob1, prob2)
            
        return similarity
    
    sim_standard = distribution_similarity(standard_samples, pytorch_samples)
    sim_exact = distribution_similarity(exact_samples, pytorch_samples)
    sim_categorical = distribution_similarity(categorical_samples, pytorch_samples)
    
    print(f"\nDistribution similarity with PyTorch:")
    print(f"  Standard MLX: {sim_standard*100:.1f}%")
    print(f"  Exact MLX: {sim_exact*100:.1f}%")
    print(f"  Categorical MLX: {sim_categorical*100:.1f}%")
    
    # Visualize results
    os.makedirs("token_analysis", exist_ok=True)
    output_file = os.path.join("token_analysis", f"sampling_comparison_t{temperature}_k{topk}.png")
    
    visualize_sampling(
        logits,
        {
            "Standard MLX": standard_samples,
            "Exact MLX": exact_samples,
            "PyTorch": pytorch_samples
        },
        title=f"Sampling Comparison (temp={temperature}, topk={topk})",
        filename=output_file
    )
    
    # Return samples for further analysis
    return {
        "standard": standard_samples,
        "exact": exact_samples,
        "categorical": categorical_samples,
        "pytorch": pytorch_samples
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MLX sampling implementations")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of sampling iterations")
    parser.add_argument("--vocab-size", type=int, default=2048,
                        help="Vocabulary size for test logits")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=50,
                        help="Top-k value for filtering")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("token_analysis", exist_ok=True)
    
    # Run comparison
    run_comparison(
        vocab_size=args.vocab_size,
        iterations=args.iterations,
        temperature=args.temperature,
        topk=args.topk,
        seed=args.seed
    )