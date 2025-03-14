"""
MLX wrapper for PyTorch CSM model that converts model parameters and handles execution.
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from csm.models.model import sample_topk

from csm.cli.mlx_layers import (
    MLXTransformer, torch_to_mlx, mlx_to_torch, create_causal_mask, index_causal_mask
)
from csm.cli.mlx_embedding import MLXEmbedding, mlx_sample_topk, mlx_sample_categorical
from csm.cli.mlx_generation import MLXFrameGenerator


class MLXWrapper:
    """
    MLX wrapper for PyTorch CSM model that converts model parameters and handles execution.
    """
    
    def __init__(self, torch_model, args=None):
        """Initialize the MLX wrapper."""
        self.torch_model = torch_model
        
        # Create default args if not provided
        if args is None:
            import argparse
            args = argparse.Namespace()
            if hasattr(torch_model, 'args'):
                model_args = torch_model.args
                args.audio_vocab_size = model_args.audio_vocab_size
                args.audio_num_codebooks = model_args.audio_num_codebooks
            else:
                # Default values
                args.audio_vocab_size = 2051
                args.audio_num_codebooks = 32
            args.debug = True  # Enable debug output
            
        self.args = args
        self.mlx_backbone = None
        self.mlx_decoder = None
        self.text_embeddings_weight = None
        self.audio_embeddings_weight = None
        self.codebook0_head_weight = None
        self.audio_head = None
        self.projection_weight = None
        self.embedding = None
        self.frame_generator = None
        self.max_seq_len = 2048
        
        # Convert PyTorch model to MLX
        self._convert_from_torch()
        
        # Set up RoPE embeddings
        self._setup_rope_embeddings()
        
        # Setup MLX KV caches
        self._setup_mlx_kv_caches()
        
    def _convert_from_torch(self):
        """Convert PyTorch model to MLX format."""
        print("Beginning parameter conversion from PyTorch to MLX")
        
        # Count parameters
        total_params = 0
        bfloat16_count = 0
        
        # Check parameter types
        for name, param in self.torch_model.named_parameters():
            total_params += 1
            if param.dtype == torch.bfloat16:
                bfloat16_count += 1
                print(f"Found BFloat16 parameter: {name} with shape {param.shape}")
        
        print(f"Found {bfloat16_count} BFloat16 parameters out of {total_params} total parameters")
        print("Converting all parameters to float32 for MLX compatibility")
        
        # Convert backbone_causal_mask
        self.backbone_causal_mask = torch_to_mlx(self.torch_model.backbone_causal_mask)
        print("Converted backbone_causal_mask")
        
        # Convert decoder_causal_mask
        self.decoder_causal_mask = torch_to_mlx(self.torch_model.decoder_causal_mask)
        print("Converted decoder_causal_mask")
        
        # Convert audio_head
        if hasattr(self.torch_model, 'audio_head'):
            self.audio_head = []
            for i, head in enumerate(self.torch_model.audio_head):
                self.audio_head.append(torch_to_mlx(head.weight))
            print("Converted audio_head")
        
        # Convert backbone transformer
        print("Converting backbone transformer...")
        self.mlx_backbone = self._convert_transformer(self.torch_model.backbone, "backbone")
        
        # Convert decoder transformer
        print("Converting decoder transformer...")
        self.mlx_decoder = self._convert_transformer(self.torch_model.decoder, "decoder")
        
        # Convert embedding weights
        if hasattr(self.torch_model, 'text_embeddings') and hasattr(self.torch_model.text_embeddings, 'weight'):
            self.text_embeddings_weight = torch_to_mlx(self.torch_model.text_embeddings.weight)
            
        if hasattr(self.torch_model, 'audio_embeddings') and hasattr(self.torch_model.audio_embeddings, 'weight'):
            self.audio_embeddings_weight = torch_to_mlx(self.torch_model.audio_embeddings.weight)
            
        # Convert projection weights
        if hasattr(self.torch_model, 'projection') and hasattr(self.torch_model.projection, 'weight'):
            self.projection_weight = torch_to_mlx(self.torch_model.projection.weight)
            
        # Convert codebook0_head weights
        if hasattr(self.torch_model, 'codebook0_head') and hasattr(self.torch_model.codebook0_head, 'weight'):
            self.codebook0_head_weight = torch_to_mlx(self.torch_model.codebook0_head.weight)
        
        # Set up MLX embedding helper
        self.embedding = MLXEmbedding(
            text_embeddings=self.text_embeddings_weight,
            audio_embeddings=self.audio_embeddings_weight,
            audio_vocab_size=self.args.audio_vocab_size,
            audio_num_codebooks=self.args.audio_num_codebooks,
            embed_dim=self.mlx_backbone.hidden_size,
            debug=self.args.debug
        )
        
        # Setup frame generator
        self.frame_generator = MLXFrameGenerator(
            backbone=self.mlx_backbone,
            decoder=self.mlx_decoder,
            embedding=self.embedding,
            projection_weight=self.projection_weight,
            codebook0_head_weight=self.codebook0_head_weight,
            audio_head_weights=self.audio_head,
            audio_vocab_size=self.args.audio_vocab_size,
            audio_num_codebooks=self.args.audio_num_codebooks,
            debug=self.args.debug,
            fallback_fn=self._fallback_generate
        )
        
        # Print success message
        print(f"Successfully converted {total_params} parameters to MLX format")

    def _convert_transformer(self, torch_model, name="model"):
        """Convert a PyTorch transformer model to MLX."""
        print(f"Converting PyTorch transformer model to MLX...")
        
        # Determine model architecture
        is_llama_style = False
        is_gpt_style = False
        is_torchtune = False
        
        # Check for Identity modules indicating CSM/torchtune architecture
        if hasattr(torch_model, 'tok_embeddings') and isinstance(torch_model.tok_embeddings, torch.nn.Identity):
            print("Note: tok_embeddings is an Identity module (CSM model architecture)")
            is_torchtune = True
            
        if hasattr(torch_model, 'output') and isinstance(torch_model.output, torch.nn.Identity):
            print("Note: output is an Identity module (CSM model architecture)")
            is_torchtune = True
            
        # Count and configure for layers
        num_layers = 0
        hidden_size = 0
        num_heads = 0
        intermediate_size = 0
        num_kv_heads = None
        
        # Check for layers attribute
        if hasattr(torch_model, 'layers'):
            num_layers = len(torch_model.layers)
            
            # Get configuration from first layer
            if num_layers > 0:
                first_layer = torch_model.layers[0]
                
                # Detect CSM/torchtune architecture
                if hasattr(first_layer, 'attn') and hasattr(first_layer, 'mlp'):
                    if hasattr(first_layer.attn, 'q_proj'):
                        # Get parameters from the layer
                        hidden_size = first_layer.attn.q_proj.weight.shape[0]
                        intermediate_size = first_layer.mlp.w1.weight.shape[0]
                        
                        # Calculate heads based on dimensions
                        if hasattr(first_layer.attn, 'n_heads'):
                            num_heads = first_layer.attn.n_heads
                        else:
                            # Estimate based on dimensions (typical head size is 64 or 128)
                            num_heads = hidden_size // 64
                            
                        # Check for MQA/GQA
                        if hasattr(first_layer.attn, 'k_proj'):
                            k_proj_shape = first_layer.attn.k_proj.weight.shape[0]
                            if k_proj_shape != hidden_size:
                                num_kv_heads = k_proj_shape // (hidden_size // num_heads)
                        
                        is_torchtune = True
        
        # Create MLX transformer
        if not is_torchtune:
            raise ValueError("Only CSM/torchtune architecture is supported")
            
        # Create MLX transformer
        mlx_model = MLXTransformer(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_kv_heads=num_kv_heads
        )
        
        # Load parameters for each layer
        loaded_layers = 0
        for i, layer in enumerate(torch_model.layers):
            try:
                print(f"Detected CSM/torchtune model architecture for layer layers.{i}")
                
                # Self-attention parameters
                mlx_model.layers[i].attn.q_proj.weight = torch_to_mlx(layer.attn.q_proj.weight)
                mlx_model.layers[i].attn.k_proj.weight = torch_to_mlx(layer.attn.k_proj.weight)
                mlx_model.layers[i].attn.v_proj.weight = torch_to_mlx(layer.attn.v_proj.weight)
                mlx_model.layers[i].attn.output_proj.weight = torch_to_mlx(layer.attn.output_proj.weight)
                
                # MLP parameters
                mlx_model.layers[i].mlp.w1.weight = torch_to_mlx(layer.mlp.w1.weight)
                mlx_model.layers[i].mlp.w2.weight = torch_to_mlx(layer.mlp.w2.weight)
                mlx_model.layers[i].mlp.w3.weight = torch_to_mlx(layer.mlp.w3.weight)
                
                # Layer norm parameters
                mlx_model.layers[i].sa_norm.scale = torch_to_mlx(layer.sa_norm.scale)
                mlx_model.layers[i].mlp_norm.scale = torch_to_mlx(layer.mlp_norm.scale)
                
                print(f"Layer layers.{i}: Loaded 9/9 parameters")
                loaded_layers += 1
            except Exception as e:
                print(f"Error loading layer {i}: {e}")
                
        # Load final layer norm if it exists
        if hasattr(torch_model, 'norm'):
            mlx_model.norm.scale = torch_to_mlx(torch_model.norm.scale)
        
        print(f"Loaded parameters for {loaded_layers}/{num_layers} layers")
        print(f"Converted PyTorch model to MLX with {loaded_layers * 9} parameters")
        
        return mlx_model
    
    def _setup_rope_embeddings(self):
        """Set up rotary positional embeddings."""
        # Get dimensions
        if hasattr(self.mlx_backbone, 'layers') and len(self.mlx_backbone.layers) > 0:
            head_dim = self.mlx_backbone.layers[0].attn.head_dim
        else:
            # Fallback
            head_dim = 64
            
        # Set max sequence length and theta
        self.max_seq_len = 2048
        theta = 10000.0
        
        # Create position indices
        position = mx.arange(0, self.max_seq_len)
        
        # Create frequency matrix
        dim = head_dim // 2
        freqs = 1.0 / (theta ** (mx.arange(0, dim) / dim))
        freqs = mx.reshape(freqs, (1, -1)) * mx.reshape(position, (-1, 1))
        
        # Create sin/cos embeddings
        cos_freqs = mx.cos(freqs).reshape(self.max_seq_len, 1, dim)
        sin_freqs = mx.sin(freqs).reshape(self.max_seq_len, 1, dim)
        
        # Use concatenation instead of repeat for compatibility
        self.cos_cached = mx.concatenate([cos_freqs] * 2, axis=-1)
        self.sin_cached = mx.concatenate([sin_freqs] * 2, axis=-1)
    
    def _setup_mlx_kv_caches(self):
        """Set up MLX KV caches for inference."""
        print("Setting up MLX KV caches...")
        # Setup is handled by each specific module
        print("MLX caches initialized successfully")
    
    def _fallback_generate(self, i=None, curr_sample=None):
        """Fallback method for generation that uses PyTorch."""
        if i is not None and curr_sample is not None:
            # Codebook fallback
            ci_sample, _ = self.torch_model._generate_codebook(
                i, mlx_to_torch(curr_sample), curr_sample.shape[1]
            )
            return ci_sample
        else:
            # Full fallback
            raise ValueError("Complete fallback not implemented yet")
            
    def generate_frame(self, tokens, input_pos, frame_idx, topk=5, temperature=1.0):
        """Generate a frame using the MLX-powered frame generator."""
        try:
            # Try to use the pure MLX implementation
            if self.frame_generator is not None:
                print("Using pure MLX pipeline for audio frame generation")
                return self.frame_generator.generate_frame(
                    tokens, input_pos, topk, temperature
                )
            else:
                # Fallback to hybrid approach
                raise ValueError("Pure MLX generator not available")
        except Exception as e:
            if self.args.debug:
                print(f"Pure MLX approach failed: {e}")
                print("Falling back to hybrid MLX/PyTorch approach")
            
            # Fall back to hybrid approach
            return self.generate_frame_hybrid(tokens, input_pos, frame_idx, topk, temperature)
    
    def generate_frame_hybrid(self, tokens, input_pos, frame_idx, topk=5, temperature=1.0):
        """Generate a frame using the hybrid MLX/PyTorch approach."""
        print("Using hybrid PyTorch/MLX approach for audio frame generation")
        with torch.no_grad():
            start_time = time.time()
            b, s, total_codebooks = tokens.size()
            
            # Prepare inputs
            tokens_mask = torch.ones_like(tokens, dtype=torch.float)
            text_tokens = tokens[:, :, -1]
            audio_tokens = tokens[:, :, :-1]
            
            # Process text and audio tokens with PyTorch
            # Use the original model for backbone and initial processing
            h, _, _ = self.torch_model.forward_first_stage(
                tokens, tokens_mask, input_pos
            )
            
            # From this point on, we'll try to use MLX where possible
            # Initial codebook with MLX
            try:
                # Get last hidden state
                last_h_mlx = torch_to_mlx(h[:, -1, :])
                
                # Generate c0 logits with MLX
                mlx_success = 0
                if self.codebook0_head_weight is not None:
                    # Matrix multiply
                    c0_logits_mlx = mx.matmul(last_h_mlx, self.codebook0_head_weight.T)
                    
                    # Sample using MLX
                    c0_sample_mlx = mlx_sample_categorical(c0_logits_mlx, temperature)
                    c0_sample = mlx_to_torch(c0_sample_mlx)
                    mlx_success += 1
                else:
                    # Fall back to PyTorch
                    c0_logits = self.torch_model.codebook0_head(mlx_to_torch(last_h_mlx, self.torch_device))
                    c0_sample = sample_topk(c0_logits, topk, temperature)
                
                # Process codebooks sequentially
                curr_sample = c0_sample
                
                # Process remaining codebooks with PyTorch
                for i in range(1, self.args.audio_num_codebooks):
                    # Fall back to PyTorch for remaining codebooks
                    ci_sample, _ = self.torch_model._generate_codebook(
                        i, mlx_to_torch(curr_sample, self.torch_device), curr_sample.shape[1]
                    )
                    curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                
                hybrid_time = time.time() - start_time
                if self.args.debug:
                    print(f"Hybrid frame generation: {hybrid_time:.3f}s, MLX sampling: {mlx_success}/{self.args.audio_num_codebooks} successful")
                
                return curr_sample
            except Exception as e:
                if self.args.debug:
                    print(f"Hybrid MLX approach failed: {e}, falling back to PyTorch")
                
                # Fall back to completely PyTorch approach
                return self.torch_model.generate_frame(tokens, tokens_mask, input_pos, temperature, topk)