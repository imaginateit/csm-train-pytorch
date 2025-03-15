#!/usr/bin/env python
"""
Utility to patch the MLX module to use the exact sampling implementation.

This script provides a simple way to patch the MLX token generation system
to use the exact PyTorch-matching implementation instead of the hybrid one.
It can be used as a simple flag or environment variable in the CSM system.
"""

import os
import sys
import importlib
from typing import Optional

def patch_mlx_sampling(enable: bool = True, verbose: bool = False) -> Optional[callable]:
    """
    Patch the MLX sampling implementation with the exact PyTorch-matching version.
    
    Args:
        enable: Whether to enable (True) or disable (False) the exact implementation
        verbose: Whether to print debug information
        
    Returns:
        The original sampling function if patching was successful, None otherwise
    """
    try:
        # Import the MLX embedding module that contains the sampling functions
        from csm.cli import mlx_embedding
        
        # Store the original function
        original_func = mlx_embedding.mlx_sample_topk
        
        if enable:
            # Import the exact implementation
            try:
                from csm.cli.mlx_sample_exact import mlx_sample_exact
                
                # Patch the module
                sys.modules['csm.cli.mlx_embedding'].mlx_sample_topk = mlx_sample_exact
                
                if verbose:
                    print("Patched MLX sampling with exact PyTorch-matching implementation")
                
                return original_func
            except ImportError as e:
                if verbose:
                    print(f"Error importing exact sampling implementation: {e}")
                return None
        else:
            # Reset to original implementation if we have the original function
            if original_func is not None:
                sys.modules['csm.cli.mlx_embedding'].mlx_sample_topk = original_func
                
                if verbose:
                    print("Restored original MLX sampling implementation")
                
            return None
    except Exception as e:
        if verbose:
            print(f"Error patching MLX sampling: {e}")
        return None

def is_exact_sampling_enabled() -> bool:
    """
    Check if the exact sampling implementation is being used.
    
    Returns:
        True if exact sampling is enabled, False otherwise
    """
    try:
        # Import both modules
        from csm.cli import mlx_embedding
        from csm.cli import mlx_sample_exact
        
        # Compare the function references
        return (mlx_embedding.mlx_sample_topk is mlx_sample_exact.mlx_sample_exact)
    except Exception:
        return False

def main():
    """
    Command-line interface for patching MLX sampling.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Patch MLX sampling with exact PyTorch-matching implementation")
    parser.add_argument('--enable', action='store_true', default=True,
                       help="Enable exact sampling (default)")
    parser.add_argument('--disable', action='store_true',
                       help="Disable exact sampling and restore original implementation")
    parser.add_argument('--status', action='store_true',
                       help="Check if exact sampling is currently enabled")
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
    
    # Handle enable/disable
    if args.disable:
        patch_mlx_sampling(False, args.verbose)
    else:
        patch_mlx_sampling(True, args.verbose)
    
    # Show final status
    if args.verbose:
        if is_exact_sampling_enabled():
            print("Exact sampling is now ENABLED")
        else:
            print("Exact sampling is now DISABLED")

if __name__ == "__main__":
    main()