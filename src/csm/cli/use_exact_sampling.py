#!/usr/bin/env python
"""
Utility to patch the MLX module to use the exact sampling implementation.

This script provides a simple way to use the exact PyTorch-matching implementation
for token generation in the CSM system.
"""

import os
import sys
import importlib
from typing import Optional

def patch_mlx_sampling(enable: bool = True, verbose: bool = False) -> Optional[callable]:
    """
    Ensure the MLX sampling implementation is using the optimized exact implementation.
    
    The optimized implementation is now the default since we've simplified the
    codebase to only use this version.
    
    Args:
        enable: Kept for compatibility - no longer needed as exact sampling is always enabled
        verbose: Whether to print debug information
        
    Returns:
        The sampling function for compatibility
    """
    try:
        # Import the MLX embedding module
        from csm.cli import mlx_embedding
        
        # The sampling function is already the optimized implementation
        # This function is kept for compatibility
        if verbose:
            print("Using optimized MLX sampling implementation")
            
        return mlx_embedding.mlx_sample_topk
    except Exception as e:
        if verbose:
            print(f"Error checking MLX sampling: {e}")
        return None

def is_exact_sampling_enabled() -> bool:
    """
    Check if the exact sampling implementation is being used.
    Always returns True since we've simplified the codebase to only use the exact sampling.
    
    Returns:
        True always
    """
    return True

def main():
    """
    Command-line interface for patching MLX sampling.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Check MLX sampling implementation")
    parser.add_argument('--status', action='store_true',
                       help="Check if exact sampling is enabled")
    parser.add_argument('--verbose', action='store_true',
                       help="Print debug information")
    
    args = parser.parse_args()
    
    # Check status if requested
    if args.status:
        if is_exact_sampling_enabled():
            print("Exact sampling is ENABLED")
        else:
            print("Exact sampling is DISABLED")
        return
    
    # Always enabled
    patch_mlx_sampling(True, args.verbose)
    
    # Show final status
    if args.verbose:
        print("Exact sampling is now ENABLED")

if __name__ == "__main__":
    main()