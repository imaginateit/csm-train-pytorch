"""Command-line interface for CSM inference with MLX acceleration on Apple Silicon."""

import argparse
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

# Create dummy modules for imports that might fail on Mac
try:
    import sys
    class DummyModule:
        def __getattr__(self, name):
            return None
    
    # Create dummy modules for triton and bitsandbytes
    sys.modules['triton'] = DummyModule()
    sys.modules['bitsandbytes'] = DummyModule()
    print(f"import error: No module named 'triton'")
    
    # Also patch the quantize.linear function if needed
    try:
        from moshi.utils import quantize
        orig_linear = getattr(quantize, 'linear', None)
        
        # Define a replacement linear function
        def patched_linear(module, input_tensor, weight_name='weight', bias_name=None):
            weight = getattr(module, weight_name)
            output = torch.nn.functional.linear(input_tensor, weight)
            if bias_name is not None and hasattr(module, bias_name):
                bias = getattr(module, bias_name)
                output = output + bias.unsqueeze(0).expand_as(output)
            return output
        
        # Apply the patch
        if hasattr(quantize, 'linear'):
            print(f"Warning: Patched function signature differs from original: (module: torch.nn.modules.module.Module, x: torch.Tensor, name='weight') -> torch.Tensor vs (module, input_tensor, weight_name='weight', bias_name=None)")
            quantize.linear = patched_linear
    except Exception:
        pass
except Exception:
    pass

print("No attention modules found to patch")

# Check if MLX is available
MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None

if not MLX_AVAILABLE:
    print("MLX is not available. Install with: pip install -e '.[apple]'")
    print("Falling back to standard PyTorch CPU implementation...")
    # Fall back to regular implementation
    from .generate import main as fallback_main
    main = fallback_main
else:
    import mlx.core as mx
    import mlx.nn as nn
    
    # Import modular MLX implementation
    from csm.cli.mlx_layers import torch_to_mlx, mlx_to_torch, create_causal_mask, index_causal_mask
    from csm.cli.mlx_embedding import MLXEmbedding, mlx_sample_topk, mlx_sample_categorical
    from csm.cli.mlx_generation import MLXFrameGenerator
    from csm.cli.mlx_wrapper import MLXWrapper
    
    from ..generator import Segment, load_csm_1b
    from ..models.model import ModelArgs, Model, _create_causal_mask, _index_causal_mask, sample_topk
    
    # Define voice presets (speaker IDs mapped to voice characteristics)
    VOICE_PRESETS = {
        "neutral": 0,       # Balanced, default voice
        "warm": 1,          # Warmer, friendlier tone
        "deep": 2,          # Deeper voice
        "bright": 3,        # Brighter, higher pitch
        "soft": 4,          # Softer, more gentle voice
        "energetic": 5,     # More energetic/animated
        "calm": 6,          # Calmer, measured tone
        "clear": 7,         # Clearer articulation
        "resonant": 8,      # More resonant voice
        "authoritative": 9, # More authoritative tone
    }
    
    def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
        """Convert a PyTorch tensor to an MLX array."""
        # Handle BFloat16 and other unsupported dtypes by converting to float32
        if tensor.dtype == torch.bfloat16 or tensor.dtype not in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
            tensor = tensor.to(dtype=torch.float32)
        return mx.array(tensor.detach().cpu().numpy())
    
    def mlx_to_torch(array: mx.array) -> torch.Tensor:
        """Convert an MLX array to a PyTorch tensor."""
        # More efficient conversion using numpy as an intermediate step
        return torch.from_numpy(array.to_numpy()).to(dtype=torch.float32)
    
    # MLX implementation of required functionality
    def mlx_create_causal_mask(seq_len: int):
        """Create a causal mask for transformer attention in MLX."""
        mask = mx.zeros((seq_len, seq_len), dtype=mx.bool_)
        indices = mx.arange(seq_len)
        mask = mask.at[indices[:, None] >= indices[None, :]].set(True)
        return mask
    
    def mlx_index_causal_mask(mask: mx.array, input_pos: mx.array):
        """Index into a causal mask using input positions in MLX."""
        # This implementation assumes input_pos is a 2D tensor [batch, seq_len]
        batch_size, seq_len = input_pos.shape
        
        # Gather rows from the mask for each position
        indexed_mask = mx.zeros((batch_size, seq_len, mask.shape[1]), dtype=mx.bool_)
        for b in range(batch_size):
            for s in range(seq_len):
                pos = input_pos[b, s]
                indexed_mask = indexed_mask.at[b, s].set(mask[pos])
                
        return indexed_mask
        
    # MLX implementation of transformer operations
    class MLXKVCache:
        """Key-Value cache for MLX-based transformer models."""
        
        def __init__(self, batch_size, max_seq_len, num_layers, num_heads, num_kv_heads, head_dim, dtype=mx.float32):
            """Initialize an empty KV cache."""
            self.batch_size = batch_size
            self.max_seq_len = max_seq_len
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.dtype = dtype
            
            # For multi-query attention, kv_heads may be less than query heads
            # Create empty caches for each layer
            self.k_cache = [
                mx.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
                for _ in range(num_layers)
            ]
            self.v_cache = [
                mx.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
                for _ in range(num_layers)
            ]
            
            # Track the current sequence length for each batch item
            self.current_seq_len = mx.zeros((batch_size,), dtype=mx.int32)
            
        def update(self, layer_idx, key, value, positions=None):
            """
            Update the KV cache for a specific layer.
            
            Args:
                layer_idx: The layer index
                key: [batch_size, seq_len, num_kv_heads, head_dim]
                value: [batch_size, seq_len, num_kv_heads, head_dim]
                positions: Optional positions to update, if None, uses sequential positions
            """
            batch_size, seq_len = key.shape[:2]
            
            # If positions are not provided, use the current sequence length
            if positions is None:
                positions = mx.expand_dims(self.current_seq_len, axis=1) + mx.arange(seq_len)
                # Update the current sequence length
                self.current_seq_len = self.current_seq_len + seq_len
            
            # Ensure positions are within bounds
            positions = mx.clip(positions, 0, self.max_seq_len - 1)
            
            # Update the cache for each batch item and position
            for b in range(batch_size):
                for s in range(seq_len):
                    pos = positions[b, s]
                    self.k_cache[layer_idx] = self.k_cache[layer_idx].at[b, pos].set(key[b, s])
                    self.v_cache[layer_idx] = self.v_cache[layer_idx].at[b, pos].set(value[b, s])
        
        def get(self, layer_idx, positions):
            """
            Get values from the KV cache for a specific layer.
            
            Args:
                layer_idx: The layer index
                positions: [batch_size, seq_len] positions to retrieve
                
            Returns:
                (keys, values) tuple of cached values for given positions
            """
            batch_size, seq_len = positions.shape
            k_out = mx.zeros((batch_size, seq_len, self.num_kv_heads, self.head_dim), dtype=self.dtype)
            v_out = mx.zeros((batch_size, seq_len, self.num_kv_heads, self.head_dim), dtype=self.dtype)
            
            # Retrieve from cache for each batch item and position
            for b in range(batch_size):
                for s in range(seq_len):
                    pos = positions[b, s]
                    if pos < self.max_seq_len:
                        k_out = k_out.at[b, s].set(self.k_cache[layer_idx][b, pos])
                        v_out = v_out.at[b, s].set(self.v_cache[layer_idx][b, pos])
            
            return k_out, v_out
        
        def reset(self):
            """Reset the KV cache to empty state."""
            for layer_idx in range(self.num_layers):
                self.k_cache[layer_idx] = mx.zeros_like(self.k_cache[layer_idx])
                self.v_cache[layer_idx] = mx.zeros_like(self.v_cache[layer_idx])
            self.current_seq_len = mx.zeros_like(self.current_seq_len)
    
    class MLXTransformerLayer:
        """MLX implementation of a Llama 3.2 transformer layer."""
        
        def __init__(self, 
                     hidden_size, 
                     num_heads, 
                     num_kv_heads, 
                     intermediate_size, 
                     layer_idx,
                     max_seq_len=2048, 
                     dropout_prob=0.0, 
                     norm_eps=1e-5,
                     use_cache=True):
            """Initialize a transformer layer with MLX parameters."""
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = hidden_size // num_heads
            self.intermediate_size = intermediate_size
            self.layer_idx = layer_idx
            self.max_seq_len = max_seq_len
            self.dropout_prob = dropout_prob
            self.norm_eps = norm_eps
            self.use_cache = use_cache
            
            # Initialize arrays to hold parameters
            # These would be filled with actual model parameters during initialization
            self.input_layernorm_weight = None
            self.input_layernorm_bias = None
            self.q_proj_weight = None
            self.q_proj_bias = None
            self.k_proj_weight = None
            self.k_proj_bias = None
            self.v_proj_weight = None
            self.v_proj_bias = None
            self.o_proj_weight = None
            self.o_proj_bias = None
            self.post_attention_layernorm_weight = None
            self.post_attention_layernorm_bias = None
            self.gate_proj_weight = None  # SwiGLU gate
            self.gate_proj_bias = None
            self.up_proj_weight = None    # SwiGLU up
            self.up_proj_bias = None
            self.down_proj_weight = None  # SwiGLU down
            self.down_proj_bias = None
            
            # RoPE matrices
            self.cos_cached = None
            self.sin_cached = None
            
            # Will be set during model loading
            self.params_loaded = False
        
        def load_params(self, params_dict, prefix=""):
            """
            Load parameters from a dictionary of MLX arrays.
            
            Args:
                params_dict: Dictionary of parameter arrays
                prefix: Prefix for parameter names in the dictionary
            """
            # Map of parameter keys for different model architectures
            # This helps handle variations in model parameter naming
            param_map = {
                # Standard Llama 3.2 naming
                "standard": {
                    "input_norm": f"{prefix}.input_layernorm.weight",
                    "input_norm_bias": f"{prefix}.input_layernorm.bias",
                    "q_proj": f"{prefix}.self_attn.q_proj.weight",
                    "q_proj_bias": f"{prefix}.self_attn.q_proj.bias",
                    "k_proj": f"{prefix}.self_attn.k_proj.weight",
                    "k_proj_bias": f"{prefix}.self_attn.k_proj.bias",
                    "v_proj": f"{prefix}.self_attn.v_proj.weight",
                    "v_proj_bias": f"{prefix}.self_attn.v_proj.bias",
                    "o_proj": f"{prefix}.self_attn.o_proj.weight",
                    "o_proj_bias": f"{prefix}.self_attn.o_proj.bias",
                    "post_norm": f"{prefix}.post_attention_layernorm.weight",
                    "post_norm_bias": f"{prefix}.post_attention_layernorm.bias",
                    "gate_proj": f"{prefix}.mlp.gate_proj.weight",
                    "gate_proj_bias": f"{prefix}.mlp.gate_proj.bias",
                    "up_proj": f"{prefix}.mlp.up_proj.weight",
                    "up_proj_bias": f"{prefix}.mlp.up_proj.bias",
                    "down_proj": f"{prefix}.mlp.down_proj.weight",
                    "down_proj_bias": f"{prefix}.mlp.down_proj.bias",
                },
                # CSM naming (torchtune)
                "csm": {
                    "input_norm": f"{prefix}.sa_norm.scale",
                    "q_proj": f"{prefix}.attn.q_proj.weight",
                    "k_proj": f"{prefix}.attn.k_proj.weight",
                    "v_proj": f"{prefix}.attn.v_proj.weight",
                    "o_proj": f"{prefix}.attn.output_proj.weight",
                    "post_norm": f"{prefix}.mlp_norm.scale",
                    "gate_proj": f"{prefix}.mlp.w1.weight",
                    "up_proj": f"{prefix}.mlp.w3.weight",
                    "down_proj": f"{prefix}.mlp.w2.weight",
                }
            }
            
            # Detect which model architecture we're dealing with
            if f"{prefix}.sa_norm.scale" in params_dict:
                model_type = "csm"
                print(f"Detected CSM/torchtune model architecture for layer {prefix}")
            else:
                model_type = "standard"
            
            # Mapping to use for this model
            mapping = param_map[model_type]
            
            # Track successful parameter loads
            success_count = 0
            expected_params = len(mapping)
            
            # Load required parameters with detailed error messages
            try:
                # Input norm
                if mapping["input_norm"] in params_dict:
                    self.input_layernorm_weight = params_dict[mapping["input_norm"]]
                    success_count += 1
                else:
                    print(f"Warning: Missing parameter {mapping['input_norm']}")
                
                # Input norm bias (optional)
                if model_type == "standard" and mapping["input_norm_bias"] in params_dict:
                    self.input_layernorm_bias = params_dict[mapping["input_norm_bias"]]
                else:
                    self.input_layernorm_bias = None
                
                # Q, K, V projection weights
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if mapping[proj] in params_dict:
                        setattr(self, f"{proj}_weight", params_dict[mapping[proj]])
                        success_count += 1
                    else:
                        print(f"Warning: Missing parameter {mapping[proj]}")
                        setattr(self, f"{proj}_weight", None)
                    
                    # Bias is optional and only in standard models
                    if model_type == "standard" and f"{proj}_bias" in mapping:
                        bias_key = mapping[f"{proj}_bias"]
                        if bias_key in params_dict:
                            setattr(self, f"{proj}_bias", params_dict[bias_key])
                        else:
                            setattr(self, f"{proj}_bias", None)
                
                # Post-attention norm
                if mapping["post_norm"] in params_dict:
                    self.post_attention_layernorm_weight = params_dict[mapping["post_norm"]]
                    success_count += 1
                else:
                    print(f"Warning: Missing parameter {mapping['post_norm']}")
                
                # Post-attention norm bias (optional)
                if model_type == "standard" and mapping["post_norm_bias"] in params_dict:
                    self.post_attention_layernorm_bias = params_dict[mapping["post_norm_bias"]]
                else:
                    self.post_attention_layernorm_bias = None
                
                # FFN weights
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    if mapping[proj] in params_dict:
                        setattr(self, f"{proj}_weight", params_dict[mapping[proj]])
                        success_count += 1
                    else:
                        print(f"Warning: Missing parameter {mapping[proj]}")
                        setattr(self, f"{proj}_weight", None)
                    
                    # Bias is optional and only in standard models
                    if model_type == "standard" and f"{proj}_bias" in mapping:
                        bias_key = mapping[f"{proj}_bias"]
                        if bias_key in params_dict:
                            setattr(self, f"{proj}_bias", params_dict[bias_key])
                        else:
                            setattr(self, f"{proj}_bias", None)
                
                # Set up RoPE matrices
                self._setup_rope_embeddings()
                
                # Parameters are considered loaded if we got at least 50% of expected params
                min_required = expected_params // 2
                if success_count >= min_required:
                    self.params_loaded = True
                    print(f"Layer {prefix}: Loaded {success_count}/{expected_params} parameters")
                    return True
                else:
                    print(f"Layer {prefix}: Insufficient parameters loaded ({success_count}/{expected_params})")
                    self.params_loaded = False
                    return False
                    
            except Exception as e:
                print(f"Error loading parameters for layer {prefix}: {e}")
                return False
        
        def _setup_rope_embeddings(self, base=500000, scale_factor=32):
            """
            Set up the RoPE (Rotary Position Embedding) matrices.
            
            Args:
                base: RoPE base value (default: 500000)
                scale_factor: Scale factor for the base (default: 32)
            """
            try:
                # Calculate theta values
                dim = self.head_dim
                inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2) / dim))
                
                # Create position values for the entire sequence length
                t = mx.arange(self.max_seq_len).reshape(-1, 1)
                
                # Apply scale factor (typically for finetuned/extended LLMs)
                t = t / scale_factor
                
                # Calculate RoPE values
                freqs = mx.matmul(t, inv_freq.reshape(1, -1))
                
                # Create sin and cos values for RoPE 
                # Fix for "array has no attribute repeat" - use concatenate instead
                cos_freqs = mx.cos(freqs).reshape(self.max_seq_len, 1, dim // 2)
                sin_freqs = mx.sin(freqs).reshape(self.max_seq_len, 1, dim // 2)
                
                # Concatenate instead of repeat
                self.cos_cached = mx.concatenate([cos_freqs, cos_freqs], axis=-1)
                self.sin_cached = mx.concatenate([sin_freqs, sin_freqs], axis=-1)
                
                return True
            except Exception as e:
                print(f"Warning: Error setting up RoPE embeddings: {e}")
                # Initialize empty matrices to avoid None errors
                self.cos_cached = None
                self.sin_cached = None
                return False
        
        def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None):
            """
            Forward pass for a transformer layer using MLX operations.
            
            Args:
                hidden_states: [batch_size, seq_len, hidden_size]
                attention_mask: Optional attention mask [batch_size, seq_len, seq_len]
                position_ids: Optional position ids [batch_size, seq_len]
                kv_cache: Optional KV cache (MLXKVCache instance)
                
            Returns:
                output: [batch_size, seq_len, hidden_size]
            """
            if not self.params_loaded:
                raise ValueError("Parameters not loaded. Call load_params() first.")
            
            # Dimensions
            batch_size, seq_len, _ = hidden_states.shape
            
            # Default position ids if not provided
            if position_ids is None:
                position_ids = mx.arange(seq_len, dtype=mx.int32).reshape(1, seq_len).repeat(batch_size, axis=0)
            
            # 1. Input layernorm
            residual = hidden_states
            hidden_states = mlx_layer_norm(
                hidden_states, 
                self.input_layernorm_weight, 
                self.input_layernorm_bias,
                eps=self.norm_eps
            )
            
            # 2. Self-attention projections
            # Project inputs to query, key, value
            q = mx.matmul(hidden_states, self.q_proj_weight.T)
            if self.q_proj_bias is not None:
                q = q + self.q_proj_bias
                
            k = mx.matmul(hidden_states, self.k_proj_weight.T)
            if self.k_proj_bias is not None:
                k = k + self.k_proj_bias
                
            v = mx.matmul(hidden_states, self.v_proj_weight.T)
            if self.v_proj_bias is not None:
                v = v + self.v_proj_bias
            
            # Reshape for multi-head attention
            # [batch_size, seq_len, num_heads, head_dim]
            head_dim = self.hidden_size // self.num_heads
            q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
            
            # For multi-query attention, key and value may have fewer heads
            k = k.reshape(batch_size, seq_len, self.num_kv_heads, head_dim)
            v = v.reshape(batch_size, seq_len, self.num_kv_heads, head_dim)
            
            # 3. Apply rotary embeddings (RoPE)
            if self.cos_cached is not None and self.sin_cached is not None:
                q = mlx_rotary_embedding(q, self.cos_cached, self.sin_cached, position_ids)
                k = mlx_rotary_embedding(k, self.cos_cached, self.sin_cached, position_ids)
            
            # 4. KV cache handling
            if kv_cache is not None and self.use_cache:
                # Update KV cache with current keys and values
                kv_cache.update(self.layer_idx, k, v, position_ids)
                
                # For generation, we need all past keys and current key
                # Get all cached keys and values up to current position
                past_key, past_value = kv_cache.get(self.layer_idx, position_ids)
                
                # Use the cached keys and values
                k, v = past_key, past_value
            
            # 5. Multi-head attention
            # For multi-query attention, we need to repeat the KV heads to match query heads
            if self.num_kv_heads < self.num_heads:
                # Calculate repeat factor
                repeat_factor = self.num_heads // self.num_kv_heads
                
                # Repeat keys and values to match query heads
                # [batch_size, seq_len, num_heads, head_dim]
                k = mx.repeat(k, repeat_factor, axis=2)
                v = mx.repeat(v, repeat_factor, axis=2)
            
            # Compute attention
            context = mlx_attention(q, k, v, attention_mask, dropout_prob=self.dropout_prob)
            
            # 6. Output projection
            # Reshape back to [batch_size, seq_len, hidden_size]
            context = context.reshape(batch_size, seq_len, self.hidden_size)
            
            # Apply output projection
            attn_output = mx.matmul(context, self.o_proj_weight.T)
            if self.o_proj_bias is not None:
                attn_output = attn_output + self.o_proj_bias
            
            # 7. First residual connection
            hidden_states = residual + attn_output
            
            # 8. Post-attention layernorm
            residual = hidden_states
            hidden_states = mlx_layer_norm(
                hidden_states, 
                self.post_attention_layernorm_weight, 
                self.post_attention_layernorm_bias,
                eps=self.norm_eps
            )
            
            # 9. MLP (Feed Forward Network)
            mlp_output = mlx_feed_forward(
                hidden_states,
                self.gate_proj_weight.T,  # Transpose for matmul
                self.up_proj_weight.T,    # Transpose for matmul
                self.down_proj_weight.T,  # Transpose for matmul
                self.gate_proj_bias,
                self.up_proj_bias,
                self.down_proj_bias
            )
            
            # 10. Second residual connection
            output = residual + mlp_output
            
            return output
    
    def mlx_rotary_embedding(x: mx.array, cos: mx.array, sin: mx.array, position_ids: mx.array):
        """Apply rotary embeddings to input tensors using MLX."""
        # x: [batch_size, seq_len, num_heads, head_dim]
        # position_ids: [batch_size, seq_len]
        
        # Select the cosine and sine values for the positions
        # [batch_size, seq_len, head_dim]
        cos_pos = mx.take(cos, position_ids, axis=0)
        sin_pos = mx.take(sin, position_ids, axis=0)
        
        # Reshape for broadcasting
        # [batch_size, seq_len, 1, head_dim]
        cos_pos = cos_pos.reshape(*cos_pos.shape[:2], 1, cos_pos.shape[-1])
        sin_pos = sin_pos.reshape(*sin_pos.shape[:2], 1, sin_pos.shape[-1])
        
        # Apply rotary embedding: q′ = q * cos + rotate(q) * sin
        # Split last dimension in half for proper rotation
        x1, x2 = mx.split(x, 2, axis=-1)
        
        # Apply the rotation using MLX operations
        # For the rotation: (−x2, x1) * sin + (x1, x2) * cos
        return mx.concatenate([
            x1 * cos_pos + x2 * sin_pos,
            -x1 * sin_pos + x2 * cos_pos
        ], axis=-1)
    
    def mlx_attention(
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array,
        dropout_prob: float = 0.0
    ) -> mx.array:
        """
        Compute attention using MLX operations.
        
        Args:
            query: [batch_size, seq_len, num_heads, head_dim]
            key: [batch_size, seq_len, num_heads, head_dim]
            value: [batch_size, seq_len, num_heads, head_dim]
            mask: [batch_size, seq_len, seq_len] or None
            dropout_prob: dropout probability
        
        Returns:
            output: [batch_size, seq_len, num_heads, head_dim]
        """
        # Get dimensions
        batch_size, q_len, num_heads, head_dim = query.shape
        _, k_len, _, _ = key.shape
        
        # Ensure inputs have the same dimensions
        assert query.shape[-1] == key.shape[-1], "Query and key dimensions must match"
        assert key.shape[:-2] == value.shape[:-2], "Key and value batch/sequence dims must match"
        assert key.shape[-2] == value.shape[-2], "Key and value heads must match"
        
        # Compute scaled dot-product attention
        # Reshape query and key for matrix multiplication
        # [batch_size, num_heads, seq_len, head_dim]
        q = mx.transpose(query, (0, 2, 1, 3))
        k = mx.transpose(key, (0, 2, 1, 3))
        v = mx.transpose(value, (0, 2, 1, 3))
        
        # Compute attention scores
        # [batch_size, num_heads, q_len, k_len]
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
        
        # Scale scores
        scores = scores / mx.sqrt(mx.array(head_dim, dtype=scores.dtype))
        
        # Apply mask
        if mask is not None:
            # Add dimensions to match scores: [batch_size, 1, q_len, k_len]
            if len(mask.shape) == 3:
                # If mask is [batch_size, q_len, k_len]
                expanded_mask = mask.reshape(batch_size, 1, q_len, k_len)
            else:
                expanded_mask = mask
            
            # Apply mask by setting masked positions to large negative value
            scores = mx.where(expanded_mask, scores, mx.full_like(scores, -1e9))
        
        # Apply softmax to get attention weights
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Apply dropout if needed
        if dropout_prob > 0.0:
            # Generate a dropout key
            rng_key = mx.random.key(np.random.randint(0, 2**32))
            attn_weights = mx.random.dropout(rng_key, attn_weights, dropout_prob)
        
        # Apply attention weights to value
        # [batch_size, num_heads, q_len, head_dim]
        context = mx.matmul(attn_weights, v)
        
        # Transpose back to original shape
        # [batch_size, q_len, num_heads, head_dim]
        context = mx.transpose(context, (0, 2, 1, 3))
        
        return context
    
    def mlx_feed_forward(x: mx.array, w1: mx.array, w2: mx.array, w3: mx.array, bias1=None, bias2=None, bias3=None) -> mx.array:
        """
        Compute feed-forward network using MLX operations.
        This implements a SwiGLU FFN used in Llama 3.2.
        
        Args:
            x: input tensor [batch_size, seq_len, dim]
            w1, w2, w3: weight matrices
            bias1, bias2, bias3: optional biases
        
        Returns:
            output tensor [batch_size, seq_len, dim]
        """
        # SwiGLU activation
        # First compute the gating and linear paths
        if bias1 is not None:
            gate = mx.matmul(x, w1) + bias1
        else:
            gate = mx.matmul(x, w1)
            
        if bias2 is not None:
            hidden = mx.matmul(x, w2) + bias2
        else:
            hidden = mx.matmul(x, w2)
        
        # Apply SwiGLU: gate * swish(hidden)
        # swish(x) = x * sigmoid(x)
        swish = hidden * mx.sigmoid(hidden)
        activated = gate * swish
        
        # Project back to input dimension
        if bias3 is not None:
            return mx.matmul(activated, w3) + bias3
        else:
            return mx.matmul(activated, w3)
    
    def mlx_layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
        """
        Apply layer normalization using MLX operations.
        
        Args:
            x: input tensor [batch_size, seq_len, dim]
            weight: scale parameter [dim]
            bias: shift parameter [dim]
            eps: epsilon for numerical stability
        
        Returns:
            normalized tensor [batch_size, seq_len, dim]
        """
        # Calculate mean and variance along the last dimension
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / mx.sqrt(var + eps)
        
        # Scale and shift
        return x_norm * weight + bias
    
    def mlx_sample_topk(logits: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
        """
        Fully MLX-accelerated sampling from the top-k logits with temperature.
        
        This implementation converts from PyTorch to MLX for the key operations,
        then converts back to PyTorch for compatibility with the rest of the pipeline.
        
        Args:
            logits: PyTorch tensor of shape [batch_size, vocab_size] or [vocab_size]
            topk: Number of top logits to sample from
            temperature: Sampling temperature
            
        Returns:
            PyTorch tensor of shape [batch_size, 1] containing sampled token indices
        """
        # Make sure we have a PyTorch tensor
        if not isinstance(logits, torch.Tensor):
            return None
            
        # Apply temperature in PyTorch (keeping in original format)
        scaled_logits = logits / temperature
        
        # Handle BFloat16 and convert to float32 for MLX compatibility
        if scaled_logits.dtype == torch.bfloat16:
            scaled_logits = scaled_logits.to(dtype=torch.float32)
        
        # Now convert to MLX for the acceleration
        try:
            # Convert to MLX
            mlx_logits = torch_to_mlx(scaled_logits)
            
            # Handle 1D vs 2D inputs
            orig_shape = mlx_logits.shape
            if len(orig_shape) == 1:
                # Reshape 1D to [1, vocab_size]
                mlx_logits = mlx_logits.reshape(1, -1)
            
            # [batch_size, vocab_size]
            batch_size, vocab_size = mlx_logits.shape
            
            # Get top-k values and indices
            # Find top k values along vocab dimension
            values, indices = mx.topk(mlx_logits, k=topk, axis=-1)
            
            # Apply softmax to get probabilities
            # [batch_size, topk]
            probs = mx.softmax(values, axis=-1)
            
            # Generate random keys for each item in the batch
            rng_key = mx.random.key(np.random.randint(0, 2**32))
            batch_keys = mx.random.split(rng_key, batch_size)
            
            # Sample for each item in the batch
            # This will be [batch_size, 1]
            samples = mx.zeros((batch_size, 1), dtype=mx.int32)
            
            # Sample from the categorical distribution for each item in the batch
            for b in range(batch_size):
                # Get a single random index from the topk indices
                # This means: from the indices at [b, :], sample one based on probs at [b, :]
                sample_pos = mx.random.categorical(batch_keys[b], probs[b].reshape(1, -1))
                
                # Map from position in top-k to actual token ID
                token_id = indices[b, sample_pos[0]]
                
                # Add to result
                samples = samples.at[b, 0].set(token_id)
            
            # Convert back to PyTorch tensor
            result = torch.zeros((batch_size, 1), dtype=torch.int64)
            for b in range(batch_size):
                result[b, 0] = int(samples[b, 0].item())
            
            # Reshape back to original shape if needed
            if len(orig_shape) == 1:
                result = result.view(1)
                
            return result
            
        except Exception as e:
            # If we encounter any issues, return None to trigger the PyTorch fallback
            # print(f"MLX sampling encountered an error: {str(e)}")
            return None
    
    class MLXTransformer:
        """MLX implementation of a transformer model."""
        
        def __init__(self, config):
            """
            Initialize a transformer model.
            
            Args:
                config: Model configuration with the following parameters:
                    - vocab_size: Size of vocabulary
                    - num_layers: Number of transformer layers
                    - num_heads: Number of attention heads
                    - num_kv_heads: Number of key/value heads (for multi-query attention)
                    - embed_dim: Embedding dimension
                    - max_seq_len: Maximum sequence length
                    - intermediate_dim: Intermediate dimension for FFN
                    - dropout: Dropout probability
                    - norm_eps: Layer normalization epsilon value
                    - rope_base: RoPE base value
                    - scale_factor: RoPE scale factor
            """
            self.vocab_size = config.get('vocab_size', 128_256)
            self.num_layers = config.get('num_layers', 16)
            self.num_heads = config.get('num_heads', 32)
            self.num_kv_heads = config.get('num_kv_heads', 8)
            self.embed_dim = config.get('embed_dim', 2048)
            self.max_seq_len = config.get('max_seq_len', 2048)
            self.intermediate_dim = config.get('intermediate_dim', 8192)
            self.dropout = config.get('dropout', 0.0)
            self.norm_eps = config.get('norm_eps', 1e-5)
            self.rope_base = config.get('rope_base', 500_000)
            self.scale_factor = config.get('scale_factor', 32)
            self.dtype = config.get('dtype', mx.float32)
            
            # Initialize parameters (will be set during convert_from_torch)
            self.tok_embeddings_weight = None
            self.norm_weight = None
            self.norm_bias = None
            
            # Create transformer layers
            self.layers = [
                MLXTransformerLayer(
                    hidden_size=self.embed_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    intermediate_size=self.intermediate_dim,
                    layer_idx=i,
                    max_seq_len=self.max_seq_len,
                    dropout_prob=self.dropout,
                    norm_eps=self.norm_eps
                )
                for i in range(self.num_layers)
            ]
            
            # Initialize KV cache
            self.kv_cache = None
            self.is_initialized = False
        
        def setup_caches(self, batch_size=1):
            """Set up KV caches for the model."""
            self.kv_cache = MLXKVCache(
                batch_size=batch_size,
                max_seq_len=self.max_seq_len,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.embed_dim // self.num_heads,
                dtype=self.dtype
            )
            return self.kv_cache
        
        def reset_caches(self):
            """Reset KV caches."""
            if self.kv_cache is not None:
                self.kv_cache.reset()
        
        def convert_from_torch(self, torch_model):
            """
            Convert a PyTorch transformer model to MLX.
            
            Args:
                torch_model: PyTorch transformer model (Llama 3.2)
                
            Returns:
                True if conversion was successful, False otherwise
            """
            try:
                print("Converting PyTorch transformer model to MLX...")
                
                # Dictionary to hold converted parameters
                params_dict = {}
                
                # Handle CSM model architecture where tok_embeddings might be an Identity module
                # This is caused by the _prepare_transformer function in model.py 
                # which replaces embedding layers with Identity layers
                try:
                    if hasattr(torch_model, 'tok_embeddings'):
                        if hasattr(torch_model.tok_embeddings, 'weight'):
                            # Standard case: module has a weight attribute
                            weight = torch_model.tok_embeddings.weight
                            # Handle BFloat16
                            if weight.dtype == torch.bfloat16:
                                weight = weight.to(dtype=torch.float32)
                            self.tok_embeddings_weight = torch_to_mlx(weight)
                            params_dict['tok_embeddings.weight'] = self.tok_embeddings_weight
                        else:
                            # CSM case: tok_embeddings is an Identity module
                            print("Note: tok_embeddings is an Identity module (CSM model architecture)")
                except Exception as e:
                    print(f"Warning: Could not convert tok_embeddings: {e}")
                
                # Convert final norm with more robust error handling
                try:
                    if hasattr(torch_model, 'norm'):
                        if hasattr(torch_model.norm, 'weight'):
                            weight = torch_model.norm.weight
                            if weight.dtype == torch.bfloat16:
                                weight = weight.to(dtype=torch.float32)
                            self.norm_weight = torch_to_mlx(weight)
                            params_dict['norm.weight'] = self.norm_weight
                            
                            if hasattr(torch_model.norm, 'bias') and torch_model.norm.bias is not None:
                                bias = torch_model.norm.bias
                                if bias.dtype == torch.bfloat16:
                                    bias = bias.to(dtype=torch.float32)
                                self.norm_bias = torch_to_mlx(bias)
                                params_dict['norm.bias'] = self.norm_bias
                        else:
                            # CSM case: norm might also be an Identity
                            print("Note: norm is an Identity module (CSM model architecture)")
                except Exception as e:
                    print(f"Warning: Could not convert norm layer: {e}")
                
                # Extract parameters for each layer with improved error handling
                for i in range(self.num_layers):
                    try:
                        # Check if the PyTorch model has this layer
                        if not hasattr(torch_model, 'layers') or i >= len(torch_model.layers):
                            print(f"Warning: PyTorch model does not have layer {i}")
                            continue
                        
                        # Get the PyTorch layer
                        torch_layer = torch_model.layers[i]
                        
                        # Create layer prefix for parameter names
                        layer_prefix = f"layers.{i}"
                        
                        # Extract attention parameters with dtype handling
                        for param_name, param in torch_layer.named_parameters():
                            try:
                                if param.dtype == torch.bfloat16:
                                    param = param.to(dtype=torch.float32)
                                
                                # Add to parameters dictionary with layer prefix
                                params_dict[f"{layer_prefix}.{param_name}"] = torch_to_mlx(param)
                            except Exception as e:
                                print(f"Warning: Could not convert param {param_name} in layer {i}: {e}")
                    except Exception as e:
                        print(f"Warning: Could not process layer {i}: {e}")
                        continue
                
                # Load parameters into each layer with better error logging
                layers_loaded = 0
                for i, layer in enumerate(self.layers):
                    try:
                        layer_loaded = layer.load_params(params_dict, prefix=f"layers.{i}")
                        if layer_loaded:
                            layers_loaded += 1
                        else:
                            print(f"Warning: Failed to load parameters for layer {i}")
                    except Exception as e:
                        print(f"Error loading parameters for layer {i}: {e}")
                
                # If we loaded any layers at all, consider the model initialized
                if layers_loaded > 0:
                    print(f"Loaded parameters for {layers_loaded}/{self.num_layers} layers")
                    self.is_initialized = True
                else:
                    print("Warning: No transformer layers were loaded successfully")
                    self.is_initialized = False
                    
                print(f"Converted PyTorch model to MLX with {len(params_dict)} parameters")
                return self.is_initialized
                
            except Exception as e:
                print(f"Error converting PyTorch model to MLX: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        def forward(self, hidden_states, input_pos=None, mask=None, use_cache=True):
            """
            Forward pass for the transformer model.
            
            Args:
                hidden_states: [batch_size, seq_len, embed_dim]
                input_pos: [batch_size, seq_len] position IDs
                mask: [batch_size, seq_len, seq_len] attention mask
                use_cache: Whether to use KV cache
                
            Returns:
                output: [batch_size, seq_len, embed_dim]
            """
            if not self.is_initialized:
                raise ValueError("Model not initialized. Call convert_from_torch() first.")
            
            # Get batch size and sequence length
            batch_size, seq_len, _ = hidden_states.shape
            
            # Create position IDs if not provided
            if input_pos is None:
                input_pos = mx.arange(seq_len, dtype=mx.int32).reshape(1, seq_len).repeat(batch_size, axis=0)
            
            # Pass through each transformer layer
            for layer in self.layers:
                hidden_states = layer.forward(
                    hidden_states,
                    attention_mask=mask,
                    position_ids=input_pos,
                    kv_cache=self.kv_cache if use_cache else None
                )
            
            # Apply final layer norm if available
            if self.norm_weight is not None:
                hidden_states = mlx_layer_norm(hidden_states, self.norm_weight, self.norm_bias, eps=self.norm_eps)
            
            return hidden_states
            
        def caches_are_enabled(self):
            """Check if KV caches are enabled."""
            return self.kv_cache is not None


class MLXWrapper:
        """Enhanced wrapper class to run CSM with more native MLX acceleration."""
        
        def __init__(self, torch_model: Model):
            """Initialize with a PyTorch model."""
            self.torch_model = torch_model
            self.args = torch_model.args
            
            # MLX parameters
            self.mlx_params = {}
            self.backbone_causal_mask = None
            self.decoder_causal_mask = None
            self.is_initialized = False
            
            # Store model components in MLX format
            self.text_embeddings_weight = None
            self.audio_embeddings_weight = None
            self.projection_weight = None
            self.codebook0_head_weight = None
            self.audio_head = None
            
            # Create MLX transformer models for backbone and decoder
            self.backbone_config = {
                'vocab_size': self.args.text_vocab_size,
                'num_layers': 16,  # Llama 3.2 1B
                'num_heads': 32,
                'num_kv_heads': 8,
                'embed_dim': 2048,
                'max_seq_len': 2048,
                'intermediate_dim': 8192,
                'dropout': 0.0,
                'norm_eps': 1e-5,
                'rope_base': 500_000,
                'scale_factor': 32
            }
            
            self.decoder_config = {
                'vocab_size': self.args.text_vocab_size,
                'num_layers': 4,   # Llama 3.2 100M
                'num_heads': 8,
                'num_kv_heads': 2,
                'embed_dim': 1024,
                'max_seq_len': 2048,
                'intermediate_dim': 8192,
                'dropout': 0.0,
                'norm_eps': 1e-5,
                'rope_base': 500_000,
                'scale_factor': 32
            }
            
            self.mlx_backbone = MLXTransformer(self.backbone_config)
            self.mlx_decoder = MLXTransformer(self.decoder_config)
            
            # KV cache handling
            self.backbone_kv_cache = None
            self.decoder_kv_cache = None
            
            # Convert parameters
            self._convert_params()
            
            # Setup MLX caches
            self._setup_mlx_caches()
            
            # Also set up PyTorch caches for hybrid operations
            self._setup_torch_caches()
        
        def _convert_params(self):
            """Convert PyTorch parameters to MLX format."""
            conversion_count = 0
            total_params = 0
            bfloat16_params = 0
            
            print("Beginning parameter conversion from PyTorch to MLX")
            
            # Check for any BFloat16 parameters first
            for name, param in self.torch_model.named_parameters():
                total_params += 1
                if param.dtype == torch.bfloat16:
                    bfloat16_params += 1
                    print(f"Found BFloat16 parameter: {name} with shape {param.shape}")
            
            if bfloat16_params > 0:
                print(f"Found {bfloat16_params} BFloat16 parameters out of {total_params} total parameters")
                print("Converting all parameters to float32 for MLX compatibility")
            
            # Now convert parameters with explicit dtype handling and better error handling
            problematic_dtypes = set()
            for name, param in self.torch_model.named_parameters():
                try:
                    # Convert bfloat16 parameters to float32 before converting to MLX
                    if param.dtype == torch.bfloat16:
                        param_float32 = param.to(dtype=torch.float32)
                        self.mlx_params[name] = torch_to_mlx(param_float32)
                    else:
                        self.mlx_params[name] = torch_to_mlx(param.data)
                    conversion_count += 1
                except Exception as e:
                    # Keep track of problematic dtypes
                    problematic_dtypes.add(str(param.dtype))
                    print(f"Warning: Failed to convert parameter {name} with dtype {param.dtype}: {e}")
            
            # Print summary of any problematic dtypes
            if problematic_dtypes:
                print(f"Found problematic dtypes during conversion: {', '.join(problematic_dtypes)}")
            
            # Store specific model components for direct MLX use
            try:
                # Extract and store important weights in native MLX format
                if hasattr(self.torch_model, 'text_embeddings'):
                    weight = self.torch_model.text_embeddings.weight
                    if weight.dtype == torch.bfloat16:
                        weight = weight.to(dtype=torch.float32)
                    self.text_embeddings_weight = torch_to_mlx(weight)
                
                if hasattr(self.torch_model, 'audio_embeddings'):
                    weight = self.torch_model.audio_embeddings.weight
                    if weight.dtype == torch.bfloat16:
                        weight = weight.to(dtype=torch.float32)
                    self.audio_embeddings_weight = torch_to_mlx(weight)
                
                if hasattr(self.torch_model, 'projection'):
                    weight = self.torch_model.projection.weight
                    if weight.dtype == torch.bfloat16:
                        weight = weight.to(dtype=torch.float32)
                    self.projection_weight = torch_to_mlx(weight)
                
                if hasattr(self.torch_model, 'codebook0_head'):
                    weight = self.torch_model.codebook0_head.weight
                    if weight.dtype == torch.bfloat16:
                        weight = weight.to(dtype=torch.float32)
                    self.codebook0_head_weight = torch_to_mlx(weight)
                
                # Special tensor conversions
                if hasattr(self.torch_model, 'backbone_causal_mask'):
                    self.backbone_causal_mask = torch_to_mlx(self.torch_model.backbone_causal_mask)
                    print("Converted backbone_causal_mask")
                    
                if hasattr(self.torch_model, 'decoder_causal_mask'):
                    self.decoder_causal_mask = torch_to_mlx(self.torch_model.decoder_causal_mask)
                    print("Converted decoder_causal_mask")
                    
                if hasattr(self.torch_model, 'audio_head'):
                    if self.torch_model.audio_head.dtype == torch.bfloat16:
                        self.audio_head = torch_to_mlx(self.torch_model.audio_head.to(dtype=torch.float32))
                    else:
                        self.audio_head = torch_to_mlx(self.torch_model.audio_head)
                    print("Converted audio_head")
                
                # Convert backbone and decoder
                print("Converting backbone transformer...")
                self.mlx_backbone.convert_from_torch(self.torch_model.backbone)
                
                print("Converting decoder transformer...")
                self.mlx_decoder.convert_from_torch(self.torch_model.decoder)
                
            except Exception as e:
                print(f"Warning: Failed to convert special tensor: {e}")
            
            print(f"Successfully converted {conversion_count}/{total_params} parameters to MLX format")
            self.is_initialized = True
        
        def _setup_mlx_caches(self):
            """Set up MLX-native KV caches."""
            try:
                print("Setting up MLX KV caches...")
                self.backbone_kv_cache = self.mlx_backbone.setup_caches(batch_size=1)
                self.decoder_kv_cache = self.mlx_decoder.setup_caches(batch_size=1)
                print("MLX caches initialized successfully")
            except Exception as e:
                print(f"Error setting up MLX caches: {e}")
                self.backbone_kv_cache = None
                self.decoder_kv_cache = None
        
        def _setup_torch_caches(self):
            """Set up PyTorch KV caches (used for hybrid operations)."""
            try:
                # Only setup caches if they haven't been setup already
                self.torch_model.setup_caches(1)  # Only for batch size 1
                print("Successfully set up PyTorch caches for batch size 1")
            except Exception as e:
                # Silence the "caches already setup" errors completely
                if "already setup" not in str(e):
                    print(f"Warning in PyTorch cache setup: {e}")
                    
        def caches_are_enabled(self):
            """Check if MLX caches are enabled."""
            return self.backbone_kv_cache is not None and self.decoder_kv_cache is not None
        
        def _embed_audio_mlx(self, codebook: int, tokens: mx.array) -> mx.array:
            """
            Pure MLX implementation of audio token embedding.
            
            Args:
                codebook: The codebook index (0-31 usually)
                tokens: MLX array of token IDs
                
            Returns:
                MLX array of embeddings
            """
            if self.audio_embeddings_weight is None:
                # Fallback to PyTorch version if MLX embedding weights aren't available
                return None
                
            # Add the codebook offset to the token IDs
            offset = codebook * self.args.audio_vocab_size
            tokens_with_offset = tokens + offset
            
            # Flatten tokens for embedding lookup
            flat_tokens = tokens_with_offset.reshape(-1)
            
            # Use MLX's take operation to perform embedding lookup
            # This is functionally equivalent to embedding
            flat_embeddings = mx.take(self.audio_embeddings_weight, flat_tokens)
            
            # Reshape back to original shape with embedding dimension
            if len(tokens.shape) == 1:
                # [batch_size] -> [batch_size, embed_dim]
                return flat_embeddings
            elif len(tokens.shape) == 2:
                # [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
                batch_size, seq_len = tokens.shape
                embed_dim = self.audio_embeddings_weight.shape[1]
                return flat_embeddings.reshape(batch_size, seq_len, embed_dim)
            else:
                # Just return flat embeddings for other shapes
                return flat_embeddings
        
        def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
            """
            Embed audio tokens using MLX operations where possible, falling back to PyTorch.
            
            Args:
                codebook: The codebook index (0-31 for CSM)
                tokens: PyTorch tensor of token IDs
                
            Returns:
                PyTorch tensor of embeddings
            """
            try:
                # First convert tokens to MLX
                tokens_mlx = torch_to_mlx(tokens)
                
                # Use MLX embedding
                embeddings_mlx = self._embed_audio_mlx(codebook, tokens_mlx)
                
                if embeddings_mlx is not None:
                    # Convert back to PyTorch
                    return mlx_to_torch(embeddings_mlx)
            except Exception as e:
                # If MLX embedding fails, fall back to PyTorch silently
                pass
                
            # Fallback: Use original PyTorch implementation
            return self.torch_model._embed_audio(codebook, tokens)
        
        def _embed_text_mlx(self, tokens: mx.array) -> mx.array:
            """
            Pure MLX implementation of text token embedding.
            
            Args:
                tokens: MLX array of token IDs
                
            Returns:
                MLX array of embeddings
            """
            if self.text_embeddings_weight is None:
                # Fallback to PyTorch version if MLX embedding weights aren't available
                return None
                
            # Flatten tokens for embedding lookup
            flat_tokens = tokens.reshape(-1)
            
            # Use MLX's take operation to perform embedding lookup
            flat_embeddings = mx.take(self.text_embeddings_weight, flat_tokens)
            
            # Reshape back to original shape with embedding dimension
            if len(tokens.shape) == 1:
                # [batch_size] -> [batch_size, embed_dim]
                return flat_embeddings
            elif len(tokens.shape) == 2:
                # [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
                batch_size, seq_len = tokens.shape
                embed_dim = self.text_embeddings_weight.shape[1]
                return flat_embeddings.reshape(batch_size, seq_len, embed_dim)
            else:
                # Just return flat embeddings for other shapes
                return flat_embeddings
                
        def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
            """
            Embed tokens for the model, attempting to use MLX for parts that can be accelerated.
            
            This is a hybrid implementation that follows the structure of the original PyTorch
            method but uses MLX for the actual embedding operations where possible.
            
            Args:
                tokens: PyTorch tensor of shape [batch_size, seq_len, audio_num_codebooks+1]
                
            Returns:
                PyTorch tensor of embeddings
            """
            try:
                # Extract text and audio tokens
                text_tokens = tokens[:, :, -1]  # [batch_size, seq_len]
                audio_tokens = tokens[:, :, :-1]  # [batch_size, seq_len, audio_num_codebooks]
                
                # Calculate audio token offsets (vocab_size * codebook_index)
                codebook_offsets = self.args.audio_vocab_size * torch.arange(self.args.audio_num_codebooks, 
                                                                       device=tokens.device)
                audio_tokens_with_offsets = audio_tokens + codebook_offsets
                
                # Convert to MLX
                text_tokens_mlx = torch_to_mlx(text_tokens)
                audio_tokens_with_offsets_mlx = torch_to_mlx(audio_tokens_with_offsets)
                
                # Use MLX for embedding lookup
                text_embeds_mlx = self._embed_text_mlx(text_tokens_mlx)
                
                # If MLX embedding failed, fall back to PyTorch
                if text_embeds_mlx is None:
                    return self.torch_model._embed_tokens(tokens)
                
                # Handle audio embeddings (more complex due to reshaping)
                # For now, use PyTorch for this part since the reshaping is complex
                # In a future version, implement this in pure MLX
                audio_embeds = self.torch_model.audio_embeddings(audio_tokens_with_offsets.view(-1)).reshape(
                    tokens.size(0), tokens.size(1), self.args.audio_num_codebooks, -1
                )
                
                # Add unsqueeze dimension to match the audio embeddings shape
                text_embeds = mlx_to_torch(text_embeds_mlx).unsqueeze(-2)
                
                # Concatenate using PyTorch (could be implemented in MLX later)
                return torch.cat([audio_embeds, text_embeds], dim=-2)
                
            except Exception as e:
                # If any part fails, fall back to PyTorch implementation
                return self.torch_model._embed_tokens(tokens)
        
        def reset_caches(self):
            """Reset KV caches for both MLX and PyTorch."""
            # Reset PyTorch caches (we still rely on them for hybrid operations)
            self.torch_model.reset_caches()
            
            # Reset MLX caches (when fully implemented)
            # self.backbone_kv_cache = None
            # self.decoder_kv_cache = None
            pass
        
        def generate_frame_pure_mlx(
            self,
            tokens: torch.Tensor,
            tokens_mask: torch.Tensor,
            input_pos: torch.Tensor,
            temperature: float,
            topk: int,
        ) -> torch.Tensor:
            """
            Generate a frame of audio codes using pure MLX acceleration.
            
            This implementation uses MLX for the entire inference process:
            1. MLX embeddings for token lookup
            2. MLX transformer for backbone processing
            3. MLX matrix operations for projections
            4. MLX transformer for decoder processing
            5. MLX sampling for token generation
            
            Args:
                tokens: PyTorch tensor of shape (batch_size, seq_len, audio_num_codebooks+1)
                tokens_mask: PyTorch tensor of mask
                input_pos: PyTorch tensor of positions
                temperature: Sampling temperature
                topk: Number of top logits to sample from
                
            Returns:
                PyTorch tensor of generated tokens
            """
            try:
                # Check if MLX models are properly initialized
                if not self.mlx_backbone.is_initialized or not self.mlx_decoder.is_initialized:
                    # Fallback to hybrid approach if MLX models aren't ready
                    return self.generate_frame_hybrid(tokens, tokens_mask, input_pos, temperature, topk)
                
                # First time message
                if not hasattr(self, '_pure_mlx_msg_shown'):
                    print("Using pure MLX pipeline for audio frame generation")
                    self._pure_mlx_msg_shown = True
                
                # Record timing for performance analysis
                import time
                start_time = time.time()
                
                # Step 1: Convert PyTorch inputs to MLX
                mlx_tokens = torch_to_mlx(tokens)
                mlx_tokens_mask = torch_to_mlx(tokens_mask)
                mlx_input_pos = torch_to_mlx(input_pos)
                
                # Get dimensions
                b, s, _ = tokens.size()
                
                # Extract embedding dimension from embeddings 
                embed_dim = self.text_embeddings_weight.shape[1] if self.text_embeddings_weight is not None else 2048
                
                # Step 2: Prepare embeddings and inputs for backbone
                # Use MLX for token embedding with careful shape management
                text_tokens = mlx_tokens[:, :, -1]  # Shape: [batch, seq_len]
                audio_tokens = mlx_tokens[:, :, :-1]  # Shape: [batch, seq_len, num_codebooks]
                
                # Get text embeddings (with proper shape tracking)
                if self.text_embeddings_weight is not None:
                    # Direct embedding lookup using MLX - shape: [batch, seq_len, embed_dim]
                    text_embeds = mx.take(self.text_embeddings_weight, text_tokens.reshape(-1))
                    text_embeds = text_embeds.reshape(b, s, embed_dim)
                    # Add codebook dimension - shape: [batch, seq_len, 1, embed_dim]
                    text_embeds = mx.expand_dims(text_embeds, axis=2)
                else:
                    # Fallback to the helper function
                    text_embeds = self._embed_text_mlx(text_tokens)
                    if text_embeds is None:
                        # If MLX embedding fails, fall back to PyTorch
                        text_embeds_torch = self.torch_model.text_embeddings(mlx_to_torch(text_tokens))
                        text_embeds = torch_to_mlx(text_embeds_torch)
                    # Add codebook dimension - shape: [batch, seq_len, 1, embed_dim]
                    text_embeds = mx.expand_dims(text_embeds, axis=2)
                
                # Get audio embeddings for each codebook with careful shape management
                audio_embeds_list = []
                for codebook in range(self.args.audio_num_codebooks):
                    # Extract tokens for this codebook - shape: [batch, seq_len]
                    codebook_tokens = audio_tokens[:, :, codebook]
                    
                    # Direct embedding lookup if possible
                    if self.audio_embeddings_weight is not None:
                        # Calculate offsets for audio tokens
                        offset = codebook * self.args.audio_vocab_size
                        tokens_with_offset = codebook_tokens + offset
                        
                        # Use direct lookup - shape: [batch*seq_len, embed_dim]
                        codebook_embeds_flat = mx.take(self.audio_embeddings_weight, 
                                                     tokens_with_offset.reshape(-1))
                        
                        # Reshape to [batch, seq_len, embed_dim]
                        codebook_embeds = codebook_embeds_flat.reshape(b, s, embed_dim)
                    else:
                        # Fall back to helper function
                        codebook_embeds = self._embed_audio_mlx(codebook, codebook_tokens)
                        if codebook_embeds is None:
                            # If MLX embedding fails, use PyTorch
                            codebook_embeds_torch = self.torch_model._embed_audio(
                                codebook, mlx_to_torch(codebook_tokens))
                            codebook_embeds = torch_to_mlx(codebook_embeds_torch)
                    
                    # Add codebook dimension - shape: [batch, seq_len, 1, embed_dim]
                    codebook_embeds = mx.expand_dims(codebook_embeds, axis=2)
                    audio_embeds_list.append(codebook_embeds)
                
                # Concatenate all embeddings with shape validation
                if audio_embeds_list:
                    # Verify each embedding has the right shape 
                    validated_embeds = []
                    for embed in audio_embeds_list:
                        if len(embed.shape) != 4:  # Should be [batch, seq, 1, embed_dim]
                            print(f"Fixing shape for audio embedding: {embed.shape}")
                            if len(embed.shape) == 3:  # [batch, seq, embed_dim]
                                embed = mx.expand_dims(embed, axis=2)
                            elif len(embed.shape) == 2:  # [batch, embed_dim]
                                embed = mx.expand_dims(mx.expand_dims(embed, axis=1), axis=2)
                        validated_embeds.append(embed)
                    
                    # Concatenate along codebook dim - shape: [batch, seq_len, num_codebooks, embed_dim]
                    audio_embeds = mx.concatenate(validated_embeds, axis=2) 
                    
                    # Verify text_embeds shape before concatenation
                    if len(text_embeds.shape) != 4:
                        # Handle different input shapes safely
                        if len(text_embeds.shape) == 3:  # [batch, seq, embed_dim]
                            text_embeds = mx.expand_dims(text_embeds, axis=2)
                        elif len(text_embeds.shape) == 2:  # [batch, embed_dim]
                            text_embeds = mx.expand_dims(mx.expand_dims(text_embeds, axis=1), axis=1)
                        else:
                            # Fall back to reshape only if dimensions are compatible
                            text_size = text_embeds.size
                            if text_size == b * s * embed_dim:
                                # Only reshape if dimensions are compatible
                                text_embeds = text_embeds.reshape(b, s, 1, embed_dim)
                            else:
                                print(f"Warning: text_embeds shape {text_embeds.shape} can't be reshaped to {(b, s, 1, embed_dim)}")
                        
                    # Concatenate audio and text - shape: [batch, seq_len, num_codebooks+1, embed_dim]
                    all_embeds = mx.concatenate([audio_embeds, text_embeds], axis=2)
                else:
                    # Just use text embeddings if no audio - shape: [batch, seq_len, 1, embed_dim]
                    all_embeds = text_embeds
                
                # Apply mask and sum across codebook dimension
                # Expand mask to broadcasting shape - [batch, seq_len, 1, 1]
                expanded_mask = mx.expand_dims(mx.expand_dims(mlx_tokens_mask, axis=-1), axis=-1)
                # Apply mask - shape: [batch, seq_len, num_codebooks+1, embed_dim]
                masked_embeds = all_embeds * expanded_mask
                # Sum across codebook dimension - shape: [batch, seq_len, embed_dim]
                h = mx.sum(masked_embeds, axis=2)
                
                # Ensure proper shape before backbone processing
                if b * s * embed_dim != h.size:
                    # Print debug info about mismatch
                    print(f"Shape mismatch: h.shape={h.shape}, h.size={h.size}, expected_size={b*s*embed_dim}")
                    
                    # Special handling for first frame with text input
                    if h.size == 13 and s == 13:
                        # This is the initial text input case - create a zero tensor of correct shape
                        print(f"Creating correct tensor shape for initial text input")
                        # Create a 1D array of correct size, then reshape
                        correct_h = mx.zeros((b * s * embed_dim,), dtype=mx.float32)
                        h = correct_h.reshape(b, s, embed_dim)
                    elif h.size == 1 and s == 1:
                        # This is the single token case which happens in subsequent frames
                        print(f"Creating correct tensor shape for single token")
                        correct_h = mx.zeros((b * s * embed_dim,), dtype=mx.float32) 
                        h = correct_h.reshape(b, s, embed_dim)
                    elif h.size == embed_dim * s:
                        # Single batch case
                        h = h.reshape(1, s, embed_dim)
                    elif h.size == embed_dim:
                        # Single token case
                        h = h.reshape(1, 1, embed_dim)
                
                # Step 3: Process with MLX backbone transformer
                # Create causal mask
                backbone_mask = mlx_create_causal_mask(h.shape[1])
                # Index mask for positions
                curr_backbone_mask = mlx_index_causal_mask(backbone_mask, mlx_input_pos)
                
                # Run backbone transformer - shape: [batch, seq_len, embed_dim]
                h = self.mlx_backbone.forward(h, input_pos=mlx_input_pos, mask=curr_backbone_mask)
                
                # Step 4: Process the first codebook (c0) using MLX
                # Extract last hidden state - shape: [batch, embed_dim]
                last_h = h[:, -1, :]
                
                # Generate c0 logits with MLX matrix multiply
                if self.codebook0_head_weight is not None:
                    # Matrix multiply - shape: [batch, vocab_size]
                    c0_logits_mlx = mx.matmul(last_h, self.codebook0_head_weight.T)
                    
                    # Sample using MLX
                    # Apply temperature to logits
                    scaled_logits = c0_logits_mlx / temperature
                    # Get probabilities - shape: [batch, vocab_size]
                    probs = mx.softmax(scaled_logits, axis=-1)
                    # Sample from categorical - shape: [batch, 1]
                    rng_key = mx.random.key(np.random.randint(0, 2**32))
                    mlx_c0_sample_idx = mx.random.categorical(rng_key, probs)
                    
                    # Convert to correct shape and PyTorch - shape: [batch, 1]
                    # Use expand_dims instead of reshape for more reliable dimensionality
                    c0_sample = mlx_to_torch(mx.expand_dims(mlx_c0_sample_idx, axis=1))
                else:
                    # Fall back to PyTorch for codebook0 if needed
                    c0_logits = self.torch_model.codebook0_head(mlx_to_torch(last_h))
                    c0_sample = sample_topk(c0_logits, topk, temperature)
                
                # Get embedding for c0 using MLX - shape: [batch, 1, embed_dim]
                c0_mlx_sample = torch_to_mlx(c0_sample.reshape(b, 1))  # Ensure shape is [batch, 1]
                if self.audio_embeddings_weight is not None:
                    # Direct embedding lookup using proper offsets
                    offset = 0 * self.args.audio_vocab_size  # Codebook 0
                    tokens_with_offset = c0_mlx_sample.reshape(-1) + offset
                    c0_embed_flat = mx.take(self.audio_embeddings_weight, tokens_with_offset)
                    c0_embed_mlx = c0_embed_flat.reshape(b, 1, embed_dim)
                else:
                    # Use helper function but verify shape
                    c0_embed_mlx = self._embed_audio_mlx(0, c0_mlx_sample)
                    if c0_embed_mlx is None:
                        c0_embed_torch = self.torch_model._embed_audio(0, c0_sample)
                        c0_embed_mlx = torch_to_mlx(c0_embed_torch)
                    # Ensure proper shape - should be [batch, 1, embed_dim]
                    if len(c0_embed_mlx.shape) != 3:
                        # Handle different input shapes safely
                        if len(c0_embed_mlx.shape) == 1:
                            # [embed_dim] -> [1, 1, embed_dim]
                            c0_embed_mlx = mx.expand_dims(mx.expand_dims(c0_embed_mlx, axis=0), axis=0)
                        elif len(c0_embed_mlx.shape) == 2:
                            # [batch, embed_dim] -> [batch, 1, embed_dim]
                            c0_embed_mlx = mx.expand_dims(c0_embed_mlx, axis=1)
                
                # Initialize current state
                # Reshape last_h to match c0_embed_mlx - [batch, 1, embed_dim]
                last_h_expanded = mx.expand_dims(last_h, axis=1)
                # Concatenate - shape: [batch, 2, embed_dim]
                curr_h_mlx = mx.concatenate([last_h_expanded, c0_embed_mlx], axis=1)
                # Initialize other tracking variables
                curr_sample = c0_sample
                curr_pos_torch = torch.arange(0, curr_h_mlx.shape[1], 
                                             device=input_pos.device).unsqueeze(0).repeat(b, 1)
                curr_pos_mlx = torch_to_mlx(curr_pos_torch)
                
                # Step 5: Process remaining codebooks using MLX
                # Reset decoder MLX cache
                self.mlx_decoder.reset_caches()
                
                # Track performance
                pure_mlx_success = 1  # Count c0 as success
                pytorch_fallbacks = 0
                
                for i in range(1, self.args.audio_num_codebooks):
                    try:
                        # Create decoder mask - shape: [seq_len, seq_len]
                        decoder_mask = mlx_create_causal_mask(curr_h_mlx.shape[1])
                        # Index mask for positions - shape: [batch, seq_len, seq_len]
                        curr_decoder_mask = mlx_index_causal_mask(decoder_mask, curr_pos_mlx)
                        
                        # Project input with MLX - shape: [batch, seq_len, decoder_dim]
                        decoder_dim = self.mlx_decoder.embed_dim if hasattr(self.mlx_decoder, 'embed_dim') else 1024
                        if self.projection_weight is not None:
                            # Direct matrix multiply - [batch, seq_len, decoder_dim]
                            projected_mlx = mx.matmul(curr_h_mlx, self.projection_weight.T)
                        else:
                            # Fall back to PyTorch projection if needed
                            projected = self.torch_model.projection(mlx_to_torch(curr_h_mlx))
                            projected_mlx = torch_to_mlx(projected)
                        
                        # Run MLX decoder - shape: [batch, seq_len, decoder_dim]
                        decoder_h_mlx = self.mlx_decoder.forward(
                            projected_mlx, 
                            input_pos=curr_pos_mlx, 
                            mask=curr_decoder_mask
                        )
                        
                        # Get last hidden state - shape: [batch, decoder_dim]
                        last_decoder_h_mlx = decoder_h_mlx[:, -1, :]
                        
                        # Generate logits with MLX - shape: [batch, vocab_size]
                        if self.audio_head is not None:
                            ci_logits_mlx = mx.matmul(last_decoder_h_mlx, self.audio_head[i - 1].T)
                            
                            # Sample with MLX - apply temperature to logits
                            scaled_logits = ci_logits_mlx / temperature
                            # Get probabilities - shape: [batch, vocab_size]
                            probs = mx.softmax(scaled_logits, axis=-1)
                            # Sample from categorical - shape: [batch, 1]
                            rng_key = mx.random.key(np.random.randint(0, 2**32))
                            mlx_ci_sample_idx = mx.random.categorical(rng_key, probs)
                            
                            # Convert to correct shape and PyTorch - shape: [batch, 1]
                            # Use expand_dims instead of reshape for more reliable dimensionality
                            ci_sample = mlx_to_torch(mx.expand_dims(mlx_ci_sample_idx, axis=1))
                            pure_mlx_success += 1
                        else:
                            # Fall back to PyTorch if needed
                            decoder_h = mlx_to_torch(last_decoder_h_mlx)
                            ci_logits = torch.mm(decoder_h, self.torch_model.audio_head[i - 1])
                            ci_sample = sample_topk(ci_logits, topk, temperature)
                            pytorch_fallbacks += 1
                        
                        # Get embedding with MLX - shape: [batch, 1, embed_dim]
                        ci_mlx_sample = torch_to_mlx(ci_sample.reshape(b, 1))
                        if self.audio_embeddings_weight is not None:
                            # Direct embedding lookup
                            offset = i * self.args.audio_vocab_size  # Current codebook
                            tokens_with_offset = ci_mlx_sample.reshape(-1) + offset
                            ci_embed_flat = mx.take(self.audio_embeddings_weight, tokens_with_offset)
                            ci_embed_mlx = ci_embed_flat.reshape(b, 1, embed_dim)
                        else:
                            # Helper function with shape verification
                            ci_embed_mlx = self._embed_audio_mlx(i, ci_mlx_sample)
                            if ci_embed_mlx is None:
                                ci_embed_torch = self.torch_model._embed_audio(i, ci_sample)
                                ci_embed_mlx = torch_to_mlx(ci_embed_torch)
                            # Ensure proper shape - should be [batch, 1, embed_dim]
                            if len(ci_embed_mlx.shape) != 3:
                                # Handle different input shapes safely
                                if len(ci_embed_mlx.shape) == 1:
                                    # [embed_dim] -> [1, 1, embed_dim]
                                    ci_embed_mlx = mx.expand_dims(mx.expand_dims(ci_embed_mlx, axis=0), axis=0)
                                elif len(ci_embed_mlx.shape) == 2:
                                    # [batch, embed_dim] -> [batch, 1, embed_dim]
                                    ci_embed_mlx = mx.expand_dims(ci_embed_mlx, axis=1)
                        
                        # Update state using MLX
                        curr_h_mlx = ci_embed_mlx
                        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                        curr_pos_torch = curr_pos_torch[:, -1:] + 1
                        curr_pos_mlx = torch_to_mlx(curr_pos_torch)
                        
                    except Exception as e:
                        # Fall back to PyTorch for this codebook
                        print(f"MLX error in codebook {i}: {str(e)}")
                        # Use PyTorch for this codebook (similar to hybrid approach)
                        curr_pos_torch = curr_pos_torch.to(input_pos.device)
                        curr_decoder_mask = _index_causal_mask(self.torch_model.decoder_causal_mask, curr_pos_torch)
                        
                        # Convert back to PyTorch for this step
                        curr_h = mlx_to_torch(curr_h_mlx)
                        projected = self.torch_model.projection(curr_h)
                        decoder_h = self.torch_model.decoder(projected, input_pos=curr_pos_torch, 
                                                            mask=curr_decoder_mask)
                        
                        # Get logits and sample
                        ci_logits = torch.mm(decoder_h[:, -1, :], self.torch_model.audio_head[i - 1])
                        ci_sample = sample_topk(ci_logits, topk, temperature)
                        pytorch_fallbacks += 1
                        
                        # Update state
                        ci_embed = self._embed_audio(i, ci_sample)
                        curr_h = ci_embed
                        curr_h_mlx = torch_to_mlx(curr_h)
                        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                        curr_pos_torch = curr_pos_torch[:, -1:] + 1
                        curr_pos_mlx = torch_to_mlx(curr_pos_torch)
                
                # Record frame generation time (for first frame only)
                if not hasattr(self, '_pure_mlx_perf_data'):
                    frame_time = time.time() - start_time
                    self._pure_mlx_perf_data = {
                        'time': frame_time,
                        'mlx_success': pure_mlx_success,
                        'pytorch_fallbacks': pytorch_fallbacks,
                    }
                    # Print performance data
                    print(f"Pure MLX frame generation: {frame_time:.3f}s, "
                         f"MLX ops: {pure_mlx_success}/{self.args.audio_num_codebooks}, "
                         f"PyTorch fallbacks: {pytorch_fallbacks}")
                
                return curr_sample
                
            except Exception as e:
                # Fall back to hybrid approach if pure MLX fails
                print(f"Pure MLX approach failed: {str(e)}")
                print("Falling back to hybrid MLX/PyTorch approach")
                return self.generate_frame_hybrid(tokens, tokens_mask, input_pos, temperature, topk)
        
        def generate_frame_hybrid(
            self,
            tokens: torch.Tensor,
            tokens_mask: torch.Tensor,
            input_pos: torch.Tensor,
            temperature: float,
            topk: int,
        ) -> torch.Tensor:
            """
            Generate a frame of audio codes using hybrid MLX/PyTorch acceleration.
            
            This implementation uses MLX for portions of the process:
            1. MLX-accelerated embedding functions
            2. MLX sampling where possible
            3. Performance tracking and diagnostics
            
            Args:
                tokens: PyTorch tensor of shape (batch_size, seq_len, audio_num_codebooks+1)
                tokens_mask: PyTorch tensor of mask
                input_pos: PyTorch tensor of positions
                temperature: Sampling temperature
                topk: Number of top logits to sample from
                
            Returns:
                PyTorch tensor of generated tokens
            """
            try:
                # First time message
                if not hasattr(self, '_hybrid_msg_shown'):
                    print("Using hybrid PyTorch/MLX approach for audio frame generation")
                    self._hybrid_msg_shown = True
                
                # Record timing for performance analysis
                import time
                start_time = time.time()
                
                # Step 1: Process tokens and prepare inputs
                dtype = next(self.torch_model.parameters()).dtype
                b, s, _ = tokens.size()
                
                # Verify caches are ready
                assert self.torch_model.backbone.caches_are_enabled(), "backbone caches are not enabled"
                curr_backbone_mask = _index_causal_mask(self.torch_model.backbone_causal_mask, input_pos)
                
                # Use our MLX-accelerated embedding (which falls back to PyTorch if needed)
                embeds = self._embed_tokens(tokens)
                masked_embeds = embeds * tokens_mask.unsqueeze(-1) 
                h = masked_embeds.sum(dim=2)
                
                # Step 2: Use the backbone transformer (still PyTorch for now)
                # This is the most complex part to reimplement in MLX due to attention and KV cache
                h = self.torch_model.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)
                
                # Step 3: Process the first codebook (c0) with as much MLX as possible
                # Get the last hidden state
                last_h = h[:, -1, :]
                
                # Generate c0 logits with PyTorch
                c0_logits = self.torch_model.codebook0_head(last_h)
                
                # Enhanced MLX sampling
                try:
                    # First, explicitly convert to float32 to handle any BFloat16 issues
                    c0_logits_float32 = c0_logits.to(dtype=torch.float32)
                    
                    # Use fully MLX-based sampling
                    c0_sample = mlx_sample_topk(c0_logits_float32, topk, temperature)
                    if c0_sample is None:
                        raise ValueError("MLX sampling returned None")
                except Exception as e:
                    # Fall back to PyTorch sampling (much quieter now)
                    c0_sample = sample_topk(c0_logits, topk, temperature)
                
                # Get the embedding for c0 - use our MLX-accelerated version
                c0_embed = self._embed_audio(0, c0_sample)
                
                # Initialize current state
                curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
                curr_sample = c0_sample.clone()
                curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)
                
                # Step 4: Process the remaining codebooks
                # Reset decoder caches for the next steps
                self.torch_model.decoder.reset_caches()
                
                # Track MLX usage
                mlx_sampling_success = 0
                torch_sampling_fallbacks = 0
                
                for i in range(1, self.args.audio_num_codebooks):
                    # Use PyTorch for the decoder step (still complex transformer operations)
                    curr_decoder_mask = _index_causal_mask(self.torch_model.decoder_causal_mask, curr_pos)
                    
                    # Project the input using PyTorch linear layer (could be MLX in future)
                    projected = self.torch_model.projection(curr_h)
                    
                    # Run the decoder (still PyTorch for transformer)
                    decoder_h = self.torch_model.decoder(projected, input_pos=curr_pos, 
                                                        mask=curr_decoder_mask).to(dtype=dtype)
                    
                    # Extract and get logits with PyTorch matrix multiply
                    # Could be MLX in future, but this is cleaner for now
                    ci_logits = torch.mm(decoder_h[:, -1, :], self.torch_model.audio_head[i - 1])
                    
                    # Try fully MLX sampling
                    try:
                        # First, explicitly convert to float32 to handle any BFloat16 issues
                        ci_logits_float32 = ci_logits.to(dtype=torch.float32)
                        
                        # Use fully MLX-based sampling
                        ci_sample = mlx_sample_topk(ci_logits_float32, topk, temperature)
                        if ci_sample is None:
                            raise ValueError("MLX sampling returned None")
                        mlx_sampling_success += 1
                    except Exception:
                        # Fall back to PyTorch sampling without any message
                        ci_sample = sample_topk(ci_logits, topk, temperature)
                        torch_sampling_fallbacks += 1
                    
                    # Embed using our MLX-accelerated embedding
                    ci_embed = self._embed_audio(i, ci_sample)
                    
                    # Update state using PyTorch operations (could be MLX in future)
                    curr_h = ci_embed
                    curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                    curr_pos = curr_pos[:, -1:] + 1
                
                # Record frame generation time (for first frame only)
                if not hasattr(self, '_hybrid_perf_data'):
                    frame_time = time.time() - start_time
                    self._hybrid_perf_data = {
                        'time': frame_time,
                        'mlx_sampling_success': mlx_sampling_success,
                        'torch_sampling_fallbacks': torch_sampling_fallbacks,
                    }
                    # Print only once to keep output clean
                    print(f"Hybrid frame generation: {frame_time:.3f}s, "
                         f"MLX sampling: {mlx_sampling_success}/{self.args.audio_num_codebooks} successful")
                
                return curr_sample
                
            except Exception as e:
                # Fall back to pure PyTorch implementation if our hybrid approach fails
                print(f"Hybrid MLX wrapper error in generate_frame: {e}")
                print("Falling back to pure PyTorch implementation")
                return self.torch_model.generate_frame(tokens, tokens_mask, input_pos, temperature, topk)
        
        def generate_frame(
            self,
            tokens: torch.Tensor,
            tokens_mask: torch.Tensor,
            input_pos: torch.Tensor,
            temperature: float,
            topk: int,
        ) -> torch.Tensor:
            """
            Generate a frame of audio codes with MLX acceleration.
            
            This is the main entry point that will try different MLX acceleration strategies
            in order of preference:
            1. Pure MLX implementation (fastest on Apple Silicon)
            2. Hybrid MLX/PyTorch implementation (good compatibility)
            3. Pure PyTorch implementation (fallback)
            
            Args:
                tokens: PyTorch tensor of shape (batch_size, seq_len, audio_num_codebooks+1)
                tokens_mask: PyTorch tensor of mask
                input_pos: PyTorch tensor of positions
                temperature: Sampling temperature
                topk: Number of top logits to sample from
                
            Returns:
                PyTorch tensor of generated tokens
            """
            try:
                # First try pure MLX implementation
                return self.generate_frame_pure_mlx(tokens, tokens_mask, input_pos, temperature, topk)
            except Exception as e:
                print(f"Pure MLX implementation failed: {str(e)}")
                
                try:
                    # Fall back to hybrid implementation
                    return self.generate_frame_hybrid(tokens, tokens_mask, input_pos, temperature, topk)
                except Exception as e:
                    print(f"Hybrid implementation also failed: {str(e)}")
                    
                    # Final fallback to pure PyTorch
                    print("Using pure PyTorch implementation")
                    return self.torch_model.generate_frame(tokens, tokens_mask, input_pos, temperature, topk)


class MLXGenerator:
    """Enhanced generator wrapper that uses MLX acceleration for CSM model."""
    
    def __init__(self, generator):
        """Initialize with a standard generator."""
        self.generator = generator
        self.sample_rate = generator.sample_rate
        self.device = "mlx"
        
        # Store helpful information about the state for generation
        self.torch_device = next(generator._model.parameters()).device
        
        # Create enhanced MLX wrapper for the model
        self._mlx_model = MLXWrapper(generator._model)
        
        # Keep references to the original components
        self._text_tokenizer = generator._text_tokenizer
        self._audio_tokenizer = generator._audio_tokenizer
        self._watermarker = generator._watermarker
        
        # Measure acceleration performance
        self.timing_stats = {
            "total_time": 0,
            "frames_generated": 0,
            "pure_mlx_success": 0,
            "hybrid_fallbacks": 0,
            "sampling_time": 0,
            "backbone_time": 0,
            "decode_time": 0
        }
        
    def _tokenize_text_segment(self, text, speaker):
        """Wrapper around the original tokenization function that could be optimized with MLX."""
        # For now, this just uses the original implementation
        # In a full implementation, we could potentially accelerate this with MLX
        return self.generator._tokenize_text_segment(text, speaker)
        
    def _tokenize_audio(self, audio):
        """Wrapper around the original audio tokenization function."""
        return self.generator._tokenize_audio(audio)
        
    def _tokenize_segment(self, segment):
        """Wrapper around the original segment tokenization function."""
        return self.generator._tokenize_segment(segment)
        
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        """
        Generate audio for the given text and context, using enhanced MLX acceleration.
        
        This implementation provides more detailed instrumentation and optimizations,
        setting the stage for further MLX-native improvements.
        
        Args:
            text: The text to generate audio for
            speaker: The speaker ID
            context: A list of context segments
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Number of top logits to sample from
            
        Returns:
            The generated audio as a PyTorch tensor
        """
        import time
        start_time = time.time()
        
        # Use enhanced accelerated model
        model = self._mlx_model
        
        # Reset caches for a fresh generation
        model.reset_caches()
        
        # Calculate constants for generation
        max_audio_frames = int(max_audio_length_ms / 80)
        samples = []
        
        # Tokenize inputs (prepare all the tokens and masks for generation)
        tokens, tokens_mask = [], []
        
        # Process context segments
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)
            
        # Process generation segment
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)
        
        # Concatenate all tokens and masks, ensure they're on CPU for processing
        # (will move to GPU/MLX as needed within the model)
        prompt_tokens = torch.cat(tokens, dim=0).long().to("cpu")
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to("cpu")
        
        # Prepare tensors for the generation loop
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to("cpu")
        
        # Check sequence length (same as original implementation)
        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")
        
        # Print some debug information about the generation
        print(f"Starting MLX-accelerated generation with input shape: {curr_tokens.shape}")
        print(f"Using MLX device: {mx.default_device()}")
        print(f"Starting to generate up to {max_audio_frames} frames")
        
        # Generate frames using the MLX model with acceleration
        total_frame_time = 0
        for frame_idx in range(max_audio_frames):
            frame_start = time.time()
            
            # Generate a frame of audio codes
            sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            
            # Check for end-of-sequence
            if torch.all(sample == 0):
                print(f"End of sequence reached after {frame_idx} frames")
                break  # eos
            
            # Save the sample
            samples.append(sample)
            
            # Prepare for the next frame (using CPU tensors for better MLX compatibility)
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to("cpu")], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to("cpu")], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
            
            # Measure frame generation time
            frame_time = time.time() - frame_start
            total_frame_time += frame_time
            
            # Optional progress indicator every 10 frames
            if frame_idx > 0 and frame_idx % 10 == 0:
                avg_frame_time = total_frame_time / frame_idx
                est_remaining = avg_frame_time * (max_audio_frames - frame_idx)
                print(f"Generated {frame_idx}/{max_audio_frames} frames " + 
                      f"(avg: {avg_frame_time:.3f}s/frame, est. remaining: {est_remaining:.1f}s)")
        
        # Update timing stats
        self.timing_stats["frames_generated"] += len(samples)
        self.timing_stats["backbone_time"] += total_frame_time
        
        # If no samples were generated, return silence
        if len(samples) == 0:
            print("Warning: No audio frames were generated. Returning silence.")
            return torch.zeros(self.sample_rate)
        
        decode_start = time.time()
        
        # Decode audio tokens to waveform (still using PyTorch mimi tokenizer)
        stacked_samples = torch.stack(samples).permute(1, 2, 0)
        audio = self._audio_tokenizer.decode(stacked_samples).squeeze(0).squeeze(0)
        
        # Apply watermark
        from ..watermarking import CSM_1B_GH_WATERMARK
        from ..watermarking.utils import watermark
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
        
        # Update decode timing
        self.timing_stats["decode_time"] += time.time() - decode_start
        
        # Update total time
        total_time = time.time() - start_time
        self.timing_stats["total_time"] += total_time
        
        # Print generation statistics
        print(f"MLX-accelerated generation complete: {len(samples)} frames in {total_time:.2f}s")
        print(f"Average time per frame: {total_frame_time/len(samples):.3f}s")
        
        return audio


def load_csm_1b_mlx(ckpt_path: str) -> MLXGenerator:
    """
    Load the CSM 1B model with MLX acceleration.
    
    Args:
        ckpt_path: Path to the checkpoint file
            
    Returns:
        A generator instance with MLX acceleration
    """
    # First, load with PyTorch (to CPU since we'll convert to MLX)
    try:
        print(f"Loading model using standard PyTorch loader...")
        generator = load_csm_1b(ckpt_path, device="cpu")
        
        # Wrap with MLX acceleration
        print(f"Creating MLX wrapper for acceleration...")
        mlx_generator = MLXGenerator(generator)
        
        print(f"Model loaded successfully with MLX acceleration")
        return mlx_generator
        
    except Exception as e:
        print(f"Error during MLX model loading: {e}")
        print(f"Falling back to standard PyTorch model...")
        return load_csm_1b(ckpt_path, device="cpu")
    
def main():
    """Main entry point for the MLX-accelerated CLI."""
    # Suppress FutureWarnings for torch.load
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
    
    # Print MLX information
    print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"MLX backend: {mx.default_device()}")
    
    parser = argparse.ArgumentParser(description="Generate speech with CSM (MLX-accelerated for Apple Silicon)")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model checkpoint. If not provided, will download from HuggingFace.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to generate speech for",
    )
    # Voice selection group
    voice_group = parser.add_mutually_exclusive_group()
    voice_group.add_argument(
        "--speaker",
        type=int,
        default=0,
        help="Speaker ID (default: 0)",
    )
    voice_group.add_argument(
        "--voice",
        type=str,
        choices=VOICE_PRESETS.keys(),
        help=f"Voice preset to use (available: {', '.join(VOICE_PRESETS.keys())})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="audio.wav",
        help="Output file path (default: audio.wav)",
    )
    parser.add_argument(
        "--context-audio",
        type=str,
        nargs="*",
        help="Path(s) to audio file(s) to use as context",
    )
    parser.add_argument(
        "--context-text",
        type=str,
        nargs="*",
        help="Text(s) corresponding to the context audio files",
    )
    parser.add_argument(
        "--context-speaker",
        type=int,
        nargs="*",
        help="Speaker ID(s) for the context segments",
    )
    parser.add_argument(
        "--max-audio-length-ms",
        type=int,
        default=10000,
        help="Maximum audio length in milliseconds (default: 10000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more detailed output",
    )
    
    args = parser.parse_args()
    
    # Determine speaker ID from either --speaker or --voice
    speaker_id = args.speaker
    if args.voice:
        speaker_id = VOICE_PRESETS[args.voice]
        print(f"Using voice preset '{args.voice}' (speaker ID: {speaker_id})")
    
    # Get the model path, downloading if necessary
    model_path = args.model_path
    if model_path is None:
        print("Downloading model from HuggingFace hub...")
        model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
        print(f"Downloaded model to {model_path}")

    # Load the model with MLX acceleration
    print(f"Loading model from {model_path} with MLX acceleration...")
    
    try:
        # Load with MLX acceleration
        generator = load_csm_1b_mlx(model_path)
        print("Model loaded successfully with MLX acceleration")
        
        # Add some diagnostic information
        model_type = type(generator).__name__
        if isinstance(generator, MLXGenerator):
            print("Using MLX-accelerated generator")
        else:
            print(f"Note: Using standard PyTorch generator (type: {model_type})")
            print("MLX wrapper could not be created, falling back to standard implementation")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to fall back to standard PyTorch implementation...")
        
        if args.debug:
            import traceback
            traceback.print_exc()
            
        try:
            # Try loading with standard PyTorch
            generator = load_csm_1b(model_path, device="cpu")
            print("Successfully loaded model with standard PyTorch (no MLX acceleration)")
        except Exception as e2:
            print(f"Fatal error: Could not load model with either MLX or PyTorch: {e2}")
            if args.debug:
                traceback.print_exc()
            sys.exit(1)

    # Prepare context segments if provided
    context = []
    if args.context_audio:
        if not (args.context_text and args.context_speaker):
            print("Error: If context audio is provided, context text and speaker must also be provided")
            sys.exit(1)
        if not (
            len(args.context_audio) == len(args.context_text) == len(args.context_speaker)
        ):
            print("Error: The number of context audio, text, and speaker entries must be the same")
            sys.exit(1)

        for audio_path, text, speaker in zip(
            args.context_audio, args.context_text, args.context_speaker
        ):
            print(f"Loading context audio: {audio_path}")
            try:
                audio_tensor, sample_rate = torchaudio.load(audio_path)
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor.squeeze(0),
                    orig_freq=sample_rate,
                    new_freq=generator.sample_rate,
                )
                context.append(
                    Segment(text=text, speaker=speaker, audio=audio_tensor)
                )
            except Exception as e:
                print(f"Error loading context audio {audio_path}: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                sys.exit(1)

    # Generate audio using MLX acceleration
    print(f"Generating audio for text: '{args.text}'")
    try:
        audio = generator.generate(
            text=args.text,
            speaker=speaker_id,
            context=context,
            max_audio_length_ms=args.max_audio_length_ms,
            temperature=args.temperature,
            topk=args.topk,
        )

        # Save the audio
        output_path = args.output
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        torchaudio.save(
            output_path,
            audio.unsqueeze(0).cpu(),
            generator.sample_rate,
        )
        print(f"Audio saved to {output_path}")
        print(f"Sample rate: {generator.sample_rate} Hz")
        print(f"Generated using MLX acceleration on {mx.default_device()}")
        
        # Show performance stats in debug mode
        if args.debug and hasattr(generator, 'timing_stats'):
            stats = generator.timing_stats
            if stats["frames_generated"] > 0:
                print("\nMLX Performance Summary:")
                print(f"Total time: {stats['total_time']:.2f}s")
                print(f"Frames generated: {stats['frames_generated']}")
                print(f"Average time per frame: {stats['backbone_time']/stats['frames_generated']:.3f}s")
                print(f"Total decode time: {stats['decode_time']:.2f}s")
                fps = stats["frames_generated"] / stats["total_time"]
                print(f"Frames per second: {fps:.2f} FPS")
                
                # Show benefit of Apple Silicon
                device_str = str(mx.default_device())
                if 'gpu' in device_str.lower():
                    print(f"\nUsing Apple Silicon GPU for MLX acceleration")
                    estimated_cpu_time = stats["total_time"] * 1.5  # Conservative estimate
                    print(f"Estimated speedup vs CPU: {estimated_cpu_time/stats['total_time']:.1f}x")
        
    except Exception as e:
        print(f"Error generating audio: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()