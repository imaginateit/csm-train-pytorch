#!/usr/bin/env python
"""
Layer-by-layer comparison tool for CSM PyTorch vs MLX implementations.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Add parent directory to path to import csm modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from csm.generator import Generator
from csm.cli.mlx_components.config import MLXConfig
from csm.cli.mlx_wrapper import load_mlx_model

# Check if MLX is available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

def compare_model_outputs(text="Hello, this is a test"):
    """Compare PyTorch and MLX model outputs for each layer."""
    print("Loading PyTorch model...")
    pt_model = Generator()
    
    if not MLX_AVAILABLE:
        print("MLX is not available. Cannot perform comparison.")
        return
    
    print("Loading MLX model...")
    mlx_config = MLXConfig()
    mlx_config.debug = True
    mlx_wrapper = load_mlx_model(None, mlx_config)
    mlx_model = mlx_wrapper.model
    
    # Tokenize the text
    print(f"Tokenizing text: '{text}'")
    pt_tokens = pt_model.tokenize(text)
    
    # Get speaker token
    speaker_id = 1  # Using "warm" voice
    speaker_token = torch.tensor([[65632 + speaker_id]])
    
    # Create input context for PyTorch model
    pt_context = torch.cat([pt_tokens, speaker_token], dim=1)
    
    # Create input for MLX model
    # Convert PyTorch tensor to MLX array
    mlx_tokens = mx.array(pt_tokens.cpu().numpy())
    
    print(f"Input shapes - PyTorch: {pt_context.shape}, MLX: {mlx_tokens.shape}")
    
    # Create a detailed comparison of layer outputs
    print("\n=== Layer-by-Layer Comparison ===")
    
    # Extract the internal model from PyTorch Generator
    pt_internal = pt_model._model
    
    # 1. Compare backbone embedding outputs
    print("\n=== Text Embedding Comparison ===")
    
    # PyTorch embedding
    with torch.no_grad():
        # Get embedding for text tokens only (exclude speaker)
        pt_embed = pt_internal.text_embeddings(pt_tokens)
    
    # MLX embedding 
    mlx_embed = mlx_model.text_embeddings(mlx_tokens)
    
    # Convert MLX to PyTorch for comparison
    mlx_embed_pt = torch.tensor(mlx_embed.tolist())
    
    print(f"PyTorch embed shape: {pt_embed.shape}, MLX embed shape: {mlx_embed.shape}")
    if pt_embed.shape == mlx_embed_pt.shape:
        # Calculate statistics for comparison
        abs_diff = torch.abs(pt_embed - mlx_embed_pt)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()
        
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        
        # Check correlation
        pt_flat = pt_embed.reshape(-1)
        mlx_flat = mlx_embed_pt.reshape(-1)
        correlation = torch.corrcoef(torch.stack([pt_flat, mlx_flat]))[0, 1].item()
        print(f"Correlation: {correlation:.6f}")
    
    # 2. Compare backbone layer outputs
    print("\n=== Backbone Layer Comparison ===")
    
    # PyTorch forward pass with hooks
    layer_outputs_pt = {}
    hooks = []
    
    # Add hooks to capture layer outputs
    def get_hook(name):
        def hook(module, input, output):
            layer_outputs_pt[name] = output
        return hook
    
    # Attach hooks to backbone layers
    for i, layer in enumerate(pt_internal.backbone.layers):
        hook = layer.register_forward_hook(get_hook(f"backbone.layer{i}"))
        hooks.append(hook)
    
    # Run forward pass to collect outputs
    with torch.no_grad():
        pt_internal(pt_context)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Now collect MLX layer outputs by running each layer individually
    # This is more complicated as we need to manually run the MLX layers
    
    # Get the MLX backbone
    mlx_backbone = mlx_model.backbone
    
    # Apply embedding first to get initial hidden states
    speaker_mlx = mx.array([[65632 + speaker_id]])
    mlx_context = mx.concatenate([mlx_tokens, speaker_mlx], axis=1)
    
    # Get embeddings for the full context
    mlx_embed_full = mlx_model.text_embeddings(mlx_context)
    
    # Apply positional encoding if needed
    # Note: CSM doesn't use explicit positional encoding, using rotary attention instead
    
    # Compare first few layers only to avoid too much output
    for i in range(min(3, len(mlx_backbone.layers))):
        print(f"\nLayer {i}:")
        
        # For first layer, use embeddings as input
        if i == 0:
            prev_output = mlx_embed_full
        else:
            # For subsequent layers, use previous layer's output
            prev_layer = mlx_backbone.layers[i-1]
            # We need to manually apply the mlx layer
            # (This is simplified - actual model might have more complex forward pass)
            prev_output = mlx_backbone.layers[i-1](prev_output)
        
        # Apply current layer
        mlx_layer_output = mlx_backbone.layers[i](prev_output)
        
        # Convert to PyTorch tensor for comparison
        mlx_layer_pt = torch.tensor(mlx_layer_output.tolist())
        
        # Get corresponding PyTorch output
        pt_layer_output = layer_outputs_pt[f"backbone.layer{i}"]
        
        print(f"  PyTorch shape: {pt_layer_output.shape}, MLX shape: {mlx_layer_output.shape}")
        
        if pt_layer_output.shape == mlx_layer_pt.shape:
            # Calculate statistics
            abs_diff = torch.abs(pt_layer_output - mlx_layer_pt)
            max_diff = torch.max(abs_diff).item()
            mean_diff = torch.mean(abs_diff).item()
            
            print(f"  Max absolute difference: {max_diff:.6f}")
            print(f"  Mean absolute difference: {mean_diff:.6f}")
            
            # Check correlation
            pt_flat = pt_layer_output.reshape(-1)
            mlx_flat = mlx_layer_pt.reshape(-1)
            if pt_flat.shape == mlx_flat.shape:
                correlation = torch.corrcoef(torch.stack([pt_flat, mlx_flat]))[0, 1].item()
                print(f"  Correlation: {correlation:.6f}")
    
    # 3. Compare audio head outputs (token prediction)
    print("\n=== Audio Head (Token Prediction) Comparison ===")
    
    # PyTorch audio head prediction
    with torch.no_grad():
        logits = pt_internal(pt_context)
        pt_audio_head = pt_internal.audio_head(logits[:, -1:])
    
    # Run MLX model fully through backbone
    mlx_backbone_output = mlx_backbone(mlx_embed_full)
    
    # Apply MLX audio head
    mlx_audio_head = mlx_model.audio_head(mlx_backbone_output[:, -1:])
    
    # Convert to PyTorch for comparison
    mlx_audio_head_pt = torch.tensor(mlx_audio_head.tolist())
    
    print(f"PyTorch audio head shape: {pt_audio_head.shape}, MLX shape: {mlx_audio_head.shape}")
    
    if pt_audio_head.shape == mlx_audio_head_pt.shape:
        # Calculate statistics
        abs_diff = torch.abs(pt_audio_head - mlx_audio_head_pt)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()
        
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        
        # Check correlation for the first codebook only (to avoid excessive output)
        pt_flat = pt_audio_head[:, 0, :].reshape(-1)
        mlx_flat = mlx_audio_head_pt[:, 0, :].reshape(-1)
        correlation = torch.corrcoef(torch.stack([pt_flat, mlx_flat]))[0, 1].item()
        print(f"Correlation (first codebook): {correlation:.6f}")
        
        # Compare top tokens predicted
        top_k = 5
        pt_top_values, pt_top_indices = torch.topk(pt_audio_head[0, 0], top_k)
        mlx_top_values, mlx_top_indices = torch.topk(mlx_audio_head_pt[0, 0], top_k)
        
        print(f"\nTop {top_k} tokens for first codebook (PyTorch):")
        for i in range(top_k):
            print(f"  Token {pt_top_indices[i].item()}: {pt_top_values[i].item():.4f}")
            
        print(f"\nTop {top_k} tokens for first codebook (MLX):")
        for i in range(top_k):
            print(f"  Token {mlx_top_indices[i].item()}: {mlx_top_values[i].item():.4f}")
    
    print("\n=== Comparison Complete ===")


def main():
    """Main function for debugging layer outputs."""
    compare_model_outputs()


if __name__ == "__main__":
    main()