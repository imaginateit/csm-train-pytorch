#!/usr/bin/env python
"""
Test script for comparing token generation between PyTorch and MLX implementations.

This script loads both PyTorch and MLX implementations of the CSM model and compares
their token generation behavior using the token analyzer and exact MLX sampling 
implementation. It helps identify differences in distribution patterns and validates
that the MLX implementation produces similar results to PyTorch.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import our token generation utilities
from csm.cli.token_analyzer import capture_tokens, analyze_distributions, distribution_similarity, save_token_analysis
from csm.cli.mlx_sample_exact import mlx_sample_exact, sample_test

def main():
    parser = argparse.ArgumentParser(description="Test and compare token generation between PyTorch and MLX")
    parser.add_argument('--text', type=str, default="Hello, this is a test of the CSM speech model.",
                        help="Text to generate speech from")
    parser.add_argument('--temperature', type=float, default=0.9,
                        help="Temperature for token sampling")
    parser.add_argument('--topk', type=int, default=50,
                        help="Top-k value for token sampling")
    parser.add_argument('--voice', type=str, default="standard",
                        help="Voice preset to use")
    parser.add_argument('--test-sampling', action='store_true',
                        help="Run direct sampling comparison without loading full models")
    parser.add_argument('--use-exact', action='store_true',
                        help="Use exact MLX sampling implementation for generation")
    parser.add_argument('--iterations', type=int, default=100,
                        help="Number of iterations for sampling test")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to run PyTorch model on (cpu, cuda, mps)")
    parser.add_argument('--save-dir', type=str, default="token_analysis",
                        help="Directory to save analysis results")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug output")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # If we're just testing the sampling function directly
    if args.test_sampling:
        print(f"=== Testing sampling implementations with {args.iterations} iterations ===")
        print(f"Parameters: temperature={args.temperature}, topk={args.topk}")
        
        # Run with different temperatures to test behavior
        for temp in [args.temperature, args.temperature + 0.3]:
            mlx_counts, pt_counts = sample_test(
                temperature=temp, 
                topk=args.topk,
                iterations=args.iterations
            )
            print("\n" + "="*50 + "\n")
        
        # Plot the results
        plt.figure(figsize=(12, 6))
        
        # Top MLX tokens
        plt.subplot(1, 2, 1)
        top_n = min(10, len(mlx_counts))
        plt.bar(
            [str(t) for t, c in mlx_counts[:top_n]], 
            [c for t, c in mlx_counts[:top_n]]
        )
        plt.title(f"Top {top_n} MLX Tokens")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        
        # Top PyTorch tokens
        plt.subplot(1, 2, 2)
        top_n = min(10, len(pt_counts))
        plt.bar(
            [str(t) for t, c in pt_counts[:top_n]], 
            [c for t, c in pt_counts[:top_n]]
        )
        plt.title(f"Top {top_n} PyTorch Tokens")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "sampling_comparison.png"))
        print(f"Sampling comparison plot saved to {args.save_dir}/sampling_comparison.png")
        
        return
    
    # Load the models for token generation comparison
    print("Loading PyTorch and MLX models...")
    
    # Import here to avoid loading everything in sampling-only mode
    from csm.generator import load_csm_1b
    from csm.cli.mlx_components.generator import MLXGenerator
    from csm.cli.mlx_components.config import MLXConfig
    
    # Load PyTorch model
    pt_model = load_csm_1b(device=args.device)
    
    # Load MLX model
    mlx_config = MLXConfig()
    mlx_model = MLXGenerator(model=pt_model._model, tokenizer=pt_model._tokenizer, debug=args.debug)
    
    # If we're using the exact implementation, patch it in
    if args.use_exact:
        print("Using exact MLX sampling implementation")
        
        # Import the mlx sampling functions
        from csm.cli.mlx_embedding import mlx_sample_topk
        import sys
        
        # Backup the original function
        original_sample_topk = mlx_sample_topk
        
        # Patch with our exact implementation
        sys.modules['csm.cli.mlx_embedding'].mlx_sample_topk = mlx_sample_exact
        
        print("MLX sampling function patched with exact implementation")
    
    # Generate speech and analyze tokens
    print(f"\nGenerating speech for: \"{args.text}\"")
    print(f"Parameters: temperature={args.temperature}, topk={args.topk}, voice={args.voice}")
    
    # Capture tokens from both implementations
    results = capture_tokens(pt_model, mlx_model, args.text)
    
    # Save the results
    implementation = "exact" if args.use_exact else "hybrid"
    save_token_analysis(results, f"{implementation}_tokens.pt")
    
    # Additional visualization if tokens were successfully captured
    if ('pytorch' in results and 'tokens' in results['pytorch'] and 
        'mlx' in results and 'tokens' in results['mlx']):
        
        pt_tokens = results['pytorch']['tokens']
        mlx_tokens = results['mlx']['tokens']
        
        if pt_tokens is not None and mlx_tokens is not None:
            try:
                # Flatten tensors for visualization
                pt_flat = pt_tokens.flatten().cpu().tolist()
                mlx_flat = mlx_tokens.flatten().cpu().tolist()
                
                # Create token distribution histograms
                plt.figure(figsize=(12, 10))
                
                # Plot with full range
                plt.subplot(2, 1, 1)
                plt.hist(pt_flat, bins=50, alpha=0.5, label='PyTorch')
                plt.hist(mlx_flat, bins=50, alpha=0.5, label='MLX')
                plt.title('Token Distribution - Full Range')
                plt.xlabel('Token Value')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot zoomed in on active range
                plt.subplot(2, 1, 2)
                
                # Determine active range
                active_min = min(min(pt_flat), min(mlx_flat))
                active_max = max(max(pt_flat), max(mlx_flat))
                
                # Add some margin
                margin = (active_max - active_min) * 0.1
                plot_min = max(0, active_min - margin)
                plot_max = active_max + margin
                
                plt.hist(pt_flat, bins=50, alpha=0.5, label='PyTorch', range=(plot_min, plot_max))
                plt.hist(mlx_flat, bins=50, alpha=0.5, label='MLX', range=(plot_min, plot_max))
                plt.title('Token Distribution - Active Range')
                plt.xlabel('Token Value')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f"{implementation}_distribution.png"))
                print(f"Token distribution plot saved to {args.save_dir}/{implementation}_distribution.png")
                
                # Additional analysis - token value frequency heatmap
                plt.figure(figsize=(15, 5))
                
                # Create bins for token values
                max_token = max(active_max, 2050)
                num_bins = min(100, max_token)
                
                # Count tokens by bin
                pt_counts = np.zeros(num_bins)
                mlx_counts = np.zeros(num_bins)
                
                for val in pt_flat:
                    if 0 <= val < num_bins:
                        pt_counts[val] += 1
                        
                for val in mlx_flat:
                    if 0 <= val < num_bins:
                        mlx_counts[val] += 1
                
                # Plot as a heatmap
                plt.subplot(1, 2, 1)
                plt.bar(range(num_bins), pt_counts)
                plt.title('PyTorch Token Values')
                plt.xlabel('Token Value')
                plt.ylabel('Count')
                
                plt.subplot(1, 2, 2)
                plt.bar(range(num_bins), mlx_counts)
                plt.title('MLX Token Values')
                plt.xlabel('Token Value')
                plt.ylabel('Count')
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f"{implementation}_token_heatmap.png"))
                print(f"Token heatmap saved to {args.save_dir}/{implementation}_token_heatmap.png")
                
            except Exception as e:
                print(f"Error generating visualization: {e}")
    
    # Restore original sampling function if we patched it
    if args.use_exact:
        sys.modules['csm.cli.mlx_embedding'].mlx_sample_topk = original_sample_topk
        print("Original MLX sampling function restored")

if __name__ == "__main__":
    main()