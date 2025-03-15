"""
Exact implementation of PyTorch-equivalent token sampling in MLX.

This module imports and re-exports the optimized implementation to ensure
we're always using the most efficient version while maintaining compatibility.
"""

from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.random
import numpy as np
import torch

# Import the optimized implementation
from .mlx_sample_exact_optimized import mlx_sample_exact_optimized

# Re-export the optimized function as the standard interface
mlx_sample_exact = mlx_sample_exact_optimized