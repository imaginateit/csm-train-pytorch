"""
MLX implementations of transformer layers and basic building blocks.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

# MLX Layer implementations
class MLXLayerNorm(nn.Module):
    """Layer normalization for MLX."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.scale = mx.zeros((hidden_size,))
        self.bias = mx.zeros((hidden_size,))

    def __call__(self, x):
        return nn.layer_norm(x, self.scale, self.bias, self.eps)


class MLXAttention(nn.Module):
    """Multi-head attention implementation for MLX."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x, mask=None, cache=None):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Handle Multi-Query Attention if needed
        if self.num_heads > self.num_kv_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k_list = [k] * repeat_factor
            v_list = [v] * repeat_factor
            k = mx.concatenate(k_list, axis=2)
            v = mx.concatenate(v_list, axis=2)
        
        # Compute attention
        attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights + mask
        
        # Apply softmax
        attn_weights = mx.softmax(attn_weights, axis=-1)
        
        # Compute weighted sum
        attn_output = mx.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Project to output space
        attn_output = self.output_proj(attn_output)
        
        return attn_output


class MLPMLP(nn.Module):
    """MLP implementation for MLX."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)

    def __call__(self, x):
        # SwiGLU activation
        swish = self.w1(x) * mx.sigmoid(self.w3(x))
        return self.w2(swish)


class MLXTransformerLayer(nn.Module):
    """Transformer layer implementation for MLX."""

    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        intermediate_size: int, 
        num_kv_heads: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.intermediate_size = intermediate_size
        
        # Self-attention
        self.sa_norm = MLXLayerNorm(hidden_size)
        self.attn = MLXAttention(hidden_size, num_heads, num_kv_heads)
        
        # MLP
        self.mlp_norm = MLXLayerNorm(hidden_size)
        self.mlp = MLPMLP(hidden_size, intermediate_size)

    def __call__(self, x, mask=None, cache=None):
        # Self-attention with residual connection
        x = x + self.attn(self.sa_norm(x), mask=mask, cache=cache)
        
        # MLP with residual connection
        x = x + self.mlp(self.mlp_norm(x))
        
        return x


class MLXTransformer(nn.Module):
    """Transformer implementation for MLX."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: Optional[int] = None,
        embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim or hidden_size
        
        # Layers
        self.layers = [
            MLXTransformerLayer(hidden_size, num_heads, intermediate_size, num_kv_heads)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.norm = MLXLayerNorm(hidden_size)
        
    def forward(self, hidden_states, input_pos=None, mask=None, cache=None):
        """Forward pass with proper error handling for shape issues."""
        
        # Check input shapes
        if len(hidden_states.shape) != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {hidden_states.shape}")
            
        batch_size, seq_len, embed_dim = hidden_states.shape
        if embed_dim != self.hidden_size:
            raise ValueError(f"Input embedding dimension {embed_dim} doesn't match model dimension {self.hidden_size}")
        
        # Process each layer
        for i, layer in enumerate(self.layers):
            try:
                hidden_states = layer(hidden_states, mask=mask)
            except Exception as e:
                raise ValueError(f"Error in layer {i}: {e}")
        
        # Apply final norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


def create_causal_mask(seq_len):
    """Create a causal mask for MLX attention."""
    # Create a mask where each element can only attend to itself and previous elements
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    return mx.array(mask)

def index_causal_mask(mask, positions):
    """Index into a causal mask using provided positions."""
    batch_size = positions.shape[0]
    seq_len = positions.shape[1]
    
    # Create a batch of indexed masks
    indexed_mask = mx.zeros((batch_size, seq_len, seq_len))
    
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                pos_i = positions[b, i]
                pos_j = positions[b, j]
                if pos_i.item() >= mask.shape[0] or pos_j.item() >= mask.shape[1]:
                    # Out of bounds, use default masking
                    indexed_mask = indexed_mask.at[b, i, j].set(-1e9 if pos_i > pos_j else 0.0)
                else:
                    indexed_mask = indexed_mask.at[b, i, j].set(mask[pos_i.item(), pos_j.item()])
    
    return indexed_mask

def rotary_embedding(x, sin, cos, position_ids):
    """Apply rotary embeddings to input tensors using the given sin/cos values and positions."""
    # Extract dimensions
    batch_size, seq_len, n_heads, head_dim = x.shape
    
    # Index into the sin/cos caches based on position_ids
    cos_pos = mx.take(cos, position_ids, axis=0)
    sin_pos = mx.take(sin, position_ids, axis=0)
    
    # Reshape for broadcasting - explicit for MLX compatibility
    cos_pos = cos_pos.reshape(batch_size, seq_len, 1, cos_pos.shape[-1])
    sin_pos = sin_pos.reshape(batch_size, seq_len, 1, sin_pos.shape[-1])
    
    # Split into even and odd dimensions
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    # Rotate even and odd dimensions
    x_rotated_even = x_even * cos_pos - x_odd * sin_pos
    x_rotated_odd = x_even * sin_pos + x_odd * cos_pos
    
    # Interleave the rotated values
    x_interleaved = mx.zeros_like(x)
    x_interleaved = x_interleaved.at[..., 0::2].set(x_rotated_even)
    x_interleaved = x_interleaved.at[..., 1::2].set(x_rotated_odd)
    
    return x_interleaved

def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert a PyTorch tensor to an MLX array."""
    if tensor is None:
        return None
    # Handle BFloat16
    if tensor.dtype == torch.bfloat16:
        return mx.array(tensor.detach().float().cpu().numpy())
    return mx.array(tensor.detach().cpu().numpy())

def mlx_to_torch(arr: mx.array, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert an MLX array to a PyTorch tensor."""
    if arr is None:
        return None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(np.array(arr), device=device)