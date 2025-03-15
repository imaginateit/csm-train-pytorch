#!/usr/bin/env python
"""
Advanced token distribution similarity test for MLX exact sampling implementation.

This script performs a comprehensive analysis of token distributions between
PyTorch and the exact MLX sampling implementation, generating detailed metrics
and visualizations to validate the quality of the MLX implementation.
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch
import random

# Add parent dir to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import MLX if available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX is not available. This test requires MLX.")
    sys.exit(1)

# Import our exact MLX sampling implementation
from csm.cli.mlx_sample_exact import mlx_sample_exact

def reference_pytorch_sample(logits, topk=50, temperature=1.0, seed=42):
    """
    Reference implementation of PyTorch's top-k + categorical sampling.
    
    Args:
        logits: Input logits tensor
        topk: Top-k filtering parameter
        temperature: Sampling temperature
        seed: Random seed
        
    Returns:
        Sampled token indices
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Convert to PyTorch tensor if needed
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits.tolist())
    
    # Make sure we have 2D shape [batch_size, vocab_size]
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(0)
        
    # Apply temperature scaling
    scaled_logits = logits / (temperature + 1e-10)
    
    # Apply top-k filtering
    batch_size, vocab_size = scaled_logits.shape
    filtered_logits = torch.full_like(scaled_logits, float('-inf'))
    
    for i in range(batch_size):
        # Get top-k values and indices
        top_values, top_indices = torch.topk(scaled_logits[i], min(topk, vocab_size))
        
        # Only keep top-k values, set others to -inf
        filtered_logits[i, top_indices] = scaled_logits[i, top_indices]
    
    # Apply softmax
    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
    
    # Sample from distribution
    samples = torch.multinomial(probs, num_samples=1)
    
    # Filter problematic tokens (1-31) for MIMI codec compatibility
    problematic = (samples >= 1) & (samples < 32)
    samples = torch.where(problematic, torch.zeros_like(samples), samples)
    
    return samples

def generate_realistic_logits(batch_size=1, vocab_size=2048, seed=42):
    """
    Generate realistic logits that mimic actual model outputs.
    
    Args:
        batch_size: Number of batches
        vocab_size: Vocabulary size
        seed: Random seed
        
    Returns:
        MLX array of logits with shape [batch_size, vocab_size]
    """
    # Set seeds
    np.random.seed(seed)
    key = mx.random.key(seed)
    
    # Base distribution (normal)
    logits = mx.random.normal(key=key, shape=(batch_size, vocab_size))
    
    # Convert to list for easier manipulation
    logits_list = logits.tolist()
    
    # Create realistic patterns:
    # 1. Primary peaks (tokens that typically dominate)
    primary_peaks = np.random.choice(vocab_size, size=5, replace=False)
    for i in range(batch_size):
        for idx in primary_peaks:
            logits_list[i][idx] += 8.0  # Strong boost
    
    # 2. Secondary peaks (common tokens)
    secondary_peaks = np.random.choice(vocab_size, size=15, replace=False)
    for i in range(batch_size):
        for idx in secondary_peaks:
            if idx not in primary_peaks:
                logits_list[i][idx] += 4.0  # Medium boost
    
    # 3. Tertiary peaks (occasional tokens)
    tertiary_peaks = np.random.choice(vocab_size, size=30, replace=False)
    for i in range(batch_size):
        for idx in tertiary_peaks:
            if idx not in primary_peaks and idx not in secondary_peaks:
                logits_list[i][idx] += 2.0  # Light boost
    
    # 4. Value valleys (rarely-used tokens)
    valleys = np.random.choice(vocab_size, size=100, replace=False)
    for i in range(batch_size):
        for idx in valleys:
            if (idx not in primary_peaks and 
                idx not in secondary_peaks and 
                idx not in tertiary_peaks):
                logits_list[i][idx] -= 3.0  # Reduce probability
    
    # 5. Problematic tokens (1-31) with some high values to test safety
    problematic = np.random.choice(range(1, 32), size=5, replace=False)
    for i in range(batch_size):
        for idx in problematic:
            logits_list[i][idx] += 6.0  # Make likely to test safety
    
    return mx.array(logits_list)

def analyze_distributions(mlx_samples, pt_samples, iterations, temperature, topk):
    """
    Analyze token distributions between MLX and PyTorch sampling.
    
    Args:
        mlx_samples: List of MLX-generated samples
        pt_samples: List of PyTorch-generated samples
        iterations: Number of samples generated
        temperature: Temperature used for sampling
        topk: Top-k parameter used for sampling
        
    Returns:
        Dictionary of metrics
    """
    # Count occurrences
    mlx_counter = Counter(mlx_samples)
    pt_counter = Counter(pt_samples)
    
    # Find common tokens
    mlx_tokens = set(mlx_counter.keys())
    pt_tokens = set(pt_counter.keys())
    common_tokens = mlx_tokens.intersection(pt_tokens)
    
    # Calculate token overlap
    if len(pt_tokens) > 0:
        token_overlap = len(common_tokens) / len(pt_tokens)
    else:
        token_overlap = 0
    
    # Calculate distribution similarity (Jaccard similarity)
    similarity = 0
    for token in common_tokens:
        mlx_prob = mlx_counter[token] / iterations
        pt_prob = pt_counter[token] / iterations
        similarity += min(mlx_prob, pt_prob)
    
    # Calculate position-by-position match rate
    position_matches = sum(1 for i in range(len(mlx_samples)) if mlx_samples[i] == pt_samples[i])
    position_match_rate = position_matches / iterations if iterations > 0 else 0
    
    # Calculate KL divergence (lower is better)
    kl_div = 0
    all_tokens = mlx_tokens.union(pt_tokens)
    for token in all_tokens:
        mlx_prob = (mlx_counter.get(token, 0) + 1e-10) / (iterations + 1e-10)
        pt_prob = (pt_counter.get(token, 0) + 1e-10) / (iterations + 1e-10)
        if mlx_prob > 0 and pt_prob > 0:
            kl_div += mlx_prob * np.log(mlx_prob / pt_prob)
    
    # Check for problematic tokens (1-31)
    mlx_problematic = sum(1 for t in mlx_samples if 1 <= t <= 31)
    pt_problematic = sum(1 for t in pt_samples if 1 <= t <= 31)
    
    # Calculate rank correlation
    mlx_ranks = {token: rank for rank, (token, _) in 
                enumerate(sorted(mlx_counter.items(), key=lambda x: x[1], reverse=True))}
    pt_ranks = {token: rank for rank, (token, _) in 
               enumerate(sorted(pt_counter.items(), key=lambda x: x[1], reverse=True))}
    
    rank_diffs = []
    for token in common_tokens:
        if token in mlx_ranks and token in pt_ranks:
            rank_diffs.append(abs(mlx_ranks[token] - pt_ranks[token]))
    
    avg_rank_diff = sum(rank_diffs) / len(rank_diffs) if rank_diffs else float('inf')
    
    # Create metrics dictionary
    metrics = {
        'mlx_unique': len(mlx_tokens),
        'pt_unique': len(pt_tokens),
        'token_overlap': token_overlap * 100,
        'distribution_similarity': similarity * 100,
        'position_match_rate': position_match_rate * 100,
        'kl_divergence': kl_div,
        'avg_rank_diff': avg_rank_diff,
        'mlx_problematic': mlx_problematic,
        'pt_problematic': pt_problematic,
        'temperature': temperature,
        'topk': topk
    }
    
    return metrics

def run_sampling_test(temperature=1.0, topk=50, iterations=1000, seed=42, 
                      batch_size=1, vocab_size=2048, print_results=True):
    """
    Run sampling test comparing MLX exact implementation with PyTorch.
    
    Args:
        temperature: Sampling temperature
        topk: Top-k parameter
        iterations: Number of samples to generate
        seed: Base random seed
        batch_size: Batch size for logits
        vocab_size: Vocabulary size
        print_results: Whether to print detailed results
        
    Returns:
        Dictionary of metrics
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    mx.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Generate realistic logits
    logits = generate_realistic_logits(batch_size, vocab_size, seed)
    
    # Convert to PyTorch tensor
    pt_logits = torch.tensor(logits.tolist())
    
    # Store samples
    mlx_samples = []
    pt_samples = []
    
    # Generate samples with both implementations
    for i in range(iterations):
        iteration_seed = seed + i
        
        # Sample with MLX exact implementation
        mlx_sample = mlx_sample_exact(logits, topk=topk, temperature=temperature, seed=iteration_seed)
        mlx_samples.append(mlx_sample.item())
        
        # Sample with PyTorch reference implementation
        torch.manual_seed(iteration_seed)
        pt_sample = reference_pytorch_sample(pt_logits, topk=topk, temperature=temperature, seed=iteration_seed)
        pt_samples.append(pt_sample.item())
    
    # Analyze distributions
    metrics = analyze_distributions(mlx_samples, pt_samples, iterations, temperature, topk)
    
    # Print results if requested
    if print_results:
        print(f"\n==== Sampling Test (temp={temperature}, topk={topk}, iters={iterations}) ====")
        print(f"MLX unique tokens: {metrics['mlx_unique']}")
        print(f"PyTorch unique tokens: {metrics['pt_unique']}")
        print(f"Token overlap: {metrics['token_overlap']:.2f}%")
        print(f"Distribution similarity: {metrics['distribution_similarity']:.2f}%")
        print(f"Position match rate: {metrics['position_match_rate']:.2f}%")
        print(f"KL divergence: {metrics['kl_divergence']:.4f}")
        print(f"Average rank difference: {metrics['avg_rank_diff']:.2f}")
        print(f"Problematic tokens: MLX={metrics['mlx_problematic']}, PyTorch={metrics['pt_problematic']}")
        
        # Show top tokens
        mlx_counter = Counter(mlx_samples)
        pt_counter = Counter(pt_samples)
        
        mlx_top = sorted(mlx_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        pt_top = sorted(pt_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\nTop 5 MLX tokens:")
        for token, count in mlx_top:
            print(f"  Token {token}: {count} times ({count/iterations*100:.1f}%)")
            
        print("\nTop 5 PyTorch tokens:")
        for token, count in pt_top:
            print(f"  Token {token}: {count} times ({count/iterations*100:.1f}%)")
    
    return metrics, (mlx_samples, pt_samples)

def run_parameter_sweep(temperatures=[0.5, 0.7, 0.9, 1.0, 1.2, 1.5], 
                        topk_values=[10, 20, 50, 100, 200, 400],
                        iterations=1000, seeds=5):
    """
    Run parameter sweep to find optimal settings for MLX sampling.
    
    Args:
        temperatures: List of temperatures to test
        topk_values: List of top-k values to test
        iterations: Number of samples per test
        seeds: Number of different random seeds to average over
        
    Returns:
        List of metrics dictionaries
    """
    print("\n===== PARAMETER SWEEP FOR OPTIMAL SAMPLING SETTINGS =====")
    print(f"Testing {len(temperatures)} temperatures × {len(topk_values)} topk values × {seeds} seeds")
    print(f"Each test generates {iterations} samples")
    
    all_results = []
    best_similarity = 0
    best_config = None
    
    # Create output directory for results
    os.makedirs('token_analysis', exist_ok=True)
    
    # Run tests for each parameter combination, averaging over multiple seeds
    for temp in temperatures:
        for topk in topk_values:
            config_results = []
            
            for seed_idx in range(seeds):
                seed = 42 + seed_idx * 1000
                
                # Run test with minimal output
                print(f"\nTesting: temp={temp}, topk={topk}, seed={seed}")
                metrics, _ = run_sampling_test(
                    temperature=temp, 
                    topk=topk, 
                    iterations=iterations,
                    seed=seed,
                    print_results=False
                )
                
                config_results.append(metrics)
            
            # Average metrics over seeds
            avg_metrics = {
                'temperature': temp,
                'topk': topk,
                'token_overlap': np.mean([m['token_overlap'] for m in config_results]),
                'distribution_similarity': np.mean([m['distribution_similarity'] for m in config_results]),
                'position_match_rate': np.mean([m['position_match_rate'] for m in config_results]),
                'kl_divergence': np.mean([m['kl_divergence'] for m in config_results]),
                'avg_rank_diff': np.mean([m['avg_rank_diff'] for m in config_results])
            }
            
            # Print averaged results
            print(f"\nAVG RESULTS: temp={temp}, topk={topk}")
            print(f"Token overlap: {avg_metrics['token_overlap']:.2f}%")
            print(f"Distribution similarity: {avg_metrics['distribution_similarity']:.2f}%")
            print(f"Position match rate: {avg_metrics['position_match_rate']:.2f}%")
            
            # Track best configuration
            if avg_metrics['distribution_similarity'] > best_similarity:
                best_similarity = avg_metrics['distribution_similarity']
                best_config = (temp, topk)
            
            all_results.append(avg_metrics)
    
    # Print summary table
    print("\n===== PARAMETER SWEEP SUMMARY =====")
    print(f"{'Temperature':<12} {'TopK':<8} {'Similarity':<12} {'Overlap':<10} {'Pos Match':<10}")
    print("="*55)
    
    for result in all_results:
        print(f"{result['temperature']:<12.1f} {result['topk']:<8d} "
              f"{result['distribution_similarity']:<12.2f}% "
              f"{result['token_overlap']:<10.2f}% "
              f"{result['position_match_rate']:<10.2f}%")
    
    # Print best configuration
    print(f"\nBest configuration: temperature={best_config[0]}, topk={best_config[1]}")
    print(f"Best similarity: {best_similarity:.2f}%")
    
    # Generate heatmap visualization
    create_parameter_heatmap(all_results, temperatures, topk_values)
    
    return all_results, best_config

def create_parameter_heatmap(results, temperatures, topk_values):
    """
    Create heatmap visualization of parameter sweep results.
    
    Args:
        results: List of metrics dictionaries
        temperatures: List of temperatures tested
        topk_values: List of top-k values tested
    """
    # Create data matrix
    similarity_matrix = np.zeros((len(temperatures), len(topk_values)))
    
    for result in results:
        temp_idx = temperatures.index(result['temperature'])
        topk_idx = topk_values.index(result['topk'])
        similarity_matrix[temp_idx, topk_idx] = result['distribution_similarity']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Distribution Similarity (%)')
    
    # Set labels
    plt.xlabel('Top-K Value')
    plt.ylabel('Temperature')
    
    # Set ticks
    plt.xticks(range(len(topk_values)), topk_values)
    plt.yticks(range(len(temperatures)), temperatures)
    
    # Add values to cells
    for i in range(len(temperatures)):
        for j in range(len(topk_values)):
            plt.text(j, i, f"{similarity_matrix[i, j]:.1f}%", 
                     ha="center", va="center", color="white" if similarity_matrix[i, j] < 85 else "black")
    
    plt.title('MLX vs PyTorch Sampling Distribution Similarity (%)')
    plt.tight_layout()
    
    # Save to file
    plt.savefig('token_analysis/parameter_heatmap.png')
    print("\nParameter heatmap saved to token_analysis/parameter_heatmap.png")

def detailed_analysis(temperature, topk, iterations=5000, seed=42):
    """
    Run detailed analysis with optimal parameters.
    
    Args:
        temperature: Temperature parameter to use
        topk: Top-k parameter to use
        iterations: Number of samples
        seed: Random seed
    """
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS WITH OPTIMAL PARAMETERS (temp={temperature}, topk={topk})")
    print(f"{'='*70}")
    
    # Run test with more iterations for statistical confidence
    metrics, (mlx_samples, pt_samples) = run_sampling_test(
        temperature=temperature,
        topk=topk,
        iterations=iterations,
        seed=seed,
        print_results=True
    )
    
    # Create distribution histograms
    plt.figure(figsize=(15, 10))
    
    # Create token counters
    mlx_counter = Counter(mlx_samples)
    pt_counter = Counter(pt_samples)
    
    # Get top tokens for comparison
    mlx_top = sorted(mlx_counter.items(), key=lambda x: x[1], reverse=True)[:50]
    pt_top = sorted(pt_counter.items(), key=lambda x: x[1], reverse=True)[:50]
    
    # Plot histogram of top token frequencies
    plt.subplot(2, 1, 1)
    
    # Display top tokens
    x = np.arange(min(20, len(mlx_top)))
    width = 0.35
    
    # MLX bars
    plt.bar(x - width/2, [count/iterations*100 for _, count in mlx_top[:20]], 
            width, label='MLX', alpha=0.7, color='green')
    
    # PyTorch bars
    plt.bar(x + width/2, [count/iterations*100 for _, count in pt_top[:20]], 
            width, label='PyTorch', alpha=0.7, color='blue')
    
    plt.xlabel('Token Rank')
    plt.ylabel('Frequency (%)')
    plt.title('Top 20 Token Frequencies')
    plt.xticks(x, [f"{token}" for token, _ in mlx_top[:20]])
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot token frequency correlation
    plt.subplot(2, 1, 2)
    
    # Collect common tokens
    common_tokens = set(mlx_counter.keys()).intersection(set(pt_counter.keys()))
    common_tokens = list(common_tokens)[:100]  # Limit to top 100 for clarity
    
    # Get probabilities
    mlx_probs = [mlx_counter.get(token, 0)/iterations*100 for token in common_tokens]
    pt_probs = [pt_counter.get(token, 0)/iterations*100 for token in common_tokens]
    
    # Create scatter plot
    plt.scatter(pt_probs, mlx_probs, alpha=0.6)
    
    # Add perfect correlation line
    max_val = max(max(mlx_probs), max(pt_probs))
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    plt.xlabel('PyTorch Token Frequency (%)')
    plt.ylabel('MLX Token Frequency (%)')
    plt.title('Token Frequency Correlation')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('token_analysis/frequency_analysis.png')
    print("Token frequency analysis saved to token_analysis/frequency_analysis.png")
    
    # Calculate additional metrics
    # Token distribution entropy (higher means more diverse)
    mlx_probs = np.array([count/iterations for count in mlx_counter.values()])
    pt_probs = np.array([count/iterations for count in pt_counter.values()])
    
    mlx_entropy = -np.sum(mlx_probs * np.log(mlx_probs))
    pt_entropy = -np.sum(pt_probs * np.log(pt_probs))
    
    # Evaluate similarity with increasing sample sizes
    sample_sizes = [100, 500, 1000, 2000, 5000]
    sample_sizes = [s for s in sample_sizes if s <= iterations]
    
    similarities = []
    overlaps = []
    
    for size in sample_sizes:
        # Analyze subset of samples
        metrics_subset = analyze_distributions(
            mlx_samples[:size], 
            pt_samples[:size], 
            size, 
            temperature, 
            topk
        )
        
        similarities.append(metrics_subset['distribution_similarity'])
        overlaps.append(metrics_subset['token_overlap'])
    
    # Plot similarity vs sample size
    plt.figure(figsize=(10, 6))
    
    plt.plot(sample_sizes, similarities, 'o-', label='Distribution Similarity')
    plt.plot(sample_sizes, overlaps, 's-', label='Token Overlap')
    
    plt.xlabel('Sample Size')
    plt.ylabel('Percentage (%)')
    plt.title('Similarity Metrics vs Sample Size')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('token_analysis/sample_size_effect.png')
    print("Sample size effect analysis saved to token_analysis/sample_size_effect.png")
    
    # Print additional metrics
    print("\nAdditional Metrics:")
    print(f"MLX token distribution entropy: {mlx_entropy:.4f}")
    print(f"PyTorch token distribution entropy: {pt_entropy:.4f}")
    print(f"Entropy ratio (MLX/PyTorch): {mlx_entropy/pt_entropy:.4f}")
    
    # Return metrics
    return metrics

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test MLX exact sampling implementation")
    parser.add_argument('--mode', type=str, default='all', choices=['quick', 'sweep', 'detailed', 'all'],
                        help="Test mode: quick, sweep, detailed, or all")
    parser.add_argument('--temperature', type=float, default=0.8,
                        help="Temperature for sampling (default: 0.8)")
    parser.add_argument('--topk', type=int, default=100,
                        help="Top-k parameter for sampling (default: 100)")
    parser.add_argument('--iterations', type=int, default=1000,
                        help="Number of iterations for testing (default: 1000)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Create directory for output
    os.makedirs('token_analysis', exist_ok=True)
    
    # Check if MLX is available
    if not MLX_AVAILABLE:
        print("MLX is not available. This test requires MLX.")
        sys.exit(1)
    
    # Print MLX info
    if hasattr(mx, '__version__'):
        print(f"MLX version: {mx.__version__}")
    print(f"MLX default device: {mx.default_device()}")
    
    # Choose mode based on command-line argument
    if args.mode == 'quick' or args.mode == 'all':
        # Run quick test
        print("\n===== QUICK SAMPLING TEST =====")
        metrics, _ = run_sampling_test(
            temperature=args.temperature,
            topk=args.topk,
            iterations=args.iterations,
            seed=args.seed
        )
    
    if args.mode == 'sweep' or args.mode == 'all':
        # Run parameter sweep
        all_results, best_config = run_parameter_sweep(
            iterations=max(500, args.iterations // 2)  # Use fewer iterations for sweep
        )
        
        # Update parameters to best values if in 'all' mode
        if args.mode == 'all':
            args.temperature, args.topk = best_config
    
    if args.mode == 'detailed' or args.mode == 'all':
        # Run detailed analysis
        detailed_metrics = detailed_analysis(
            temperature=args.temperature,
            topk=args.topk,
            iterations=args.iterations * 2 if args.mode == 'detailed' else args.iterations,
            seed=args.seed
        )

if __name__ == "__main__":
    main()