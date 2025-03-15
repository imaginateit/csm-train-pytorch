#!/usr/bin/env python3
"""
Benchmark MLX token sampling implementations.

This script compares the performance of the original vs optimized MLX token sampling
implementations, measuring both speed and accuracy.
"""

import time
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

import mlx.core as mx

def run_benchmark():
    """Run benchmark of original vs optimized sampling."""
    # Import implementations
    from src.csm.cli.mlx_sample_exact import mlx_sample_exact
    try:
        from src.csm.cli.mlx_sample_exact_optimized import mlx_sample_exact_optimized
    except ImportError:
        print("Optimized implementation not found. Please run 'optimize_mlx.sh' first.")
        return []
    
    # Parameters to test
    temperatures = [0.5, 0.8, 1.0, 1.2]
    topk_values = [25, 50, 100, 200]
    iterations = 20
    vocab_size = 2048
    
    # Results storage
    results = []
    
    # Create realistic test data
    key = mx.random.key(42)
    
    # Test each parameter combination
    for temperature in temperatures:
        for topk in topk_values:
            # Generate fresh test data for each parameter set
            seed = int(temperature * 100) + topk
            np.random.seed(seed)
            key = mx.random.key(seed)
            
            # Create test logits with realistic distribution
            logits = mx.random.normal(key=key, shape=(1, vocab_size))
            
            # Add peaks to simulate real token distribution
            logits_list = logits.tolist()
            peaks = np.random.choice(vocab_size, size=10, replace=False)
            for idx in peaks:
                logits_list[0][idx] += 5.0  # Boost certain tokens
            logits = mx.array(logits_list)
            
            # Time original implementation
            original_samples = []
            start_time = time.time()
            for i in range(iterations):
                sample = mlx_sample_exact(logits, topk=topk, temperature=temperature, seed=seed+i)
                original_samples.append(sample.item())
            original_time = time.time() - start_time
            
            # Time optimized implementation
            optimized_samples = []
            start_time = time.time()
            for i in range(iterations):
                sample = mlx_sample_exact_optimized(logits, topk=topk, temperature=temperature, seed=seed+i)
                optimized_samples.append(sample.item())
            optimized_time = time.time() - start_time
            
            # Calculate speedup
            speedup = original_time / optimized_time if optimized_time > 0 else 1.0
            
            # Calculate sample match rate
            matches = sum(1 for a, b in zip(original_samples, optimized_samples) if a == b)
            match_percentage = (matches / iterations) * 100
            
            # Count unique tokens in each
            unique_original = len(set(original_samples))
            unique_optimized = len(set(optimized_samples))
            
            # Print results
            print(f"Temp={temperature}, TopK={topk}:")
            print(f"  Original: {original_time:.4f}s, Optimized: {optimized_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Match rate: {match_percentage:.1f}% ({matches}/{iterations})")
            print(f"  Unique tokens - Original: {unique_original}, Optimized: {unique_optimized}")
            
            # Store results
            results.append({
                'temperature': temperature,
                'topk': topk,
                'original_time': original_time,
                'optimized_time': optimized_time,
                'speedup': speedup,
                'match_percentage': match_percentage,
                'unique_original': unique_original,
                'unique_optimized': unique_optimized
            })
    
    # Calculate average speedup
    avg_speedup = sum(r['speedup'] for r in results) / len(results) if results else 0
    avg_match = sum(r['match_percentage'] for r in results) / len(results) if results else 0
    
    print("\n=== SUMMARY ===")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Average match rate: {avg_match:.1f}%")
    
    if results:
        # Find best and worst cases
        best_speedup = max(results, key=lambda r: r['speedup'])
        worst_speedup = min(results, key=lambda r: r['speedup'])
        
        print(f"Best speedup: {best_speedup['speedup']:.2f}x (temp={best_speedup['temperature']}, topk={best_speedup['topk']})")
        print(f"Worst speedup: {worst_speedup['speedup']:.2f}x (temp={worst_speedup['temperature']}, topk={worst_speedup['topk']})")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX token generation")
    parser.add_argument("--plot", action="store_true", help="Generate plots of results")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations per test")
    args = parser.parse_args()
    
    print("Running MLX token sampling benchmark...")
    results = run_benchmark()
    
    # Create plots if requested and available
    if args.plot and HAS_PLOTTING and results:
        try:
            # Plot speedup by temperature and topk
            plt.figure(figsize=(12, 6))
            
            # Group by temperature
            temps = sorted(set(r['temperature'] for r in results))
            for temp in temps:
                temp_results = [r for r in results if r['temperature'] == temp]
                temp_results.sort(key=lambda r: r['topk'])
                
                plt.plot(
                    [r['topk'] for r in temp_results],
                    [r['speedup'] for r in temp_results],
                    'o-',
                    label=f"Temp={temp}"
                )
                
            plt.xlabel('Top-K Value')
            plt.ylabel('Speedup Factor (x)')
            plt.title('MLX Sampling Optimization Speedup')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig('mlx_speedup.png')
            print("Speedup plot saved to mlx_speedup.png")
            
            # Plot match percentage
            plt.figure(figsize=(12, 6))
            
            for temp in temps:
                temp_results = [r for r in results if r['temperature'] == temp]
                temp_results.sort(key=lambda r: r['topk'])
                
                plt.plot(
                    [r['topk'] for r in temp_results],
                    [r['match_percentage'] for r in temp_results],
                    'o-',
                    label=f"Temp={temp}"
                )
                
            plt.xlabel('Top-K Value')
            plt.ylabel('Match Percentage (%)')
            plt.title('MLX Sampling Implementation Matching')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig('mlx_matching.png')
            print("Matching plot saved to mlx_matching.png")
        except Exception as e:
            print(f"Error creating plots: {e}")
    elif args.plot and not HAS_PLOTTING:
        print("Plotting requested but matplotlib is not available. Install with 'pip install matplotlib'")

if __name__ == "__main__":
    main()