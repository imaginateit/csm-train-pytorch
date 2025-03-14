"""
MLX implementation of embedding and sampling operations for CSM.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import torch

from csm.cli.mlx_layers import torch_to_mlx, mlx_to_torch

class MLXEmbedding:
    """
    MLX implementation of embedding operations with robust shape handling.
    """
    
    def __init__(
        self, 
        text_embeddings: Optional[mx.array] = None,
        audio_embeddings: Optional[mx.array] = None,
        audio_vocab_size: int = 2048,
        audio_num_codebooks: int = 32,
        embed_dim: int = 2048,
        debug: bool = False
    ):
        self.text_embeddings = text_embeddings
        self.audio_embeddings = audio_embeddings
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks
        self.embed_dim = embed_dim
        self.debug = debug
        
    def embed_text(self, tokens: mx.array) -> mx.array:
        """
        Embed text tokens with careful shape handling.
        
        Args:
            tokens: Text tokens with shape [batch_size, seq_len]
            
        Returns:
            Text embeddings with shape [batch_size, seq_len, embed_dim]
        """
        if self.text_embeddings is None:
            raise ValueError("Text embeddings not available")
            
        # Handle input shape carefully
        original_shape = tokens.shape
        
        # Get dimensions
        if len(original_shape) == 0:  # Scalar
            batch_size, seq_len = 1, 1
            tokens = mx.array([[tokens.item() if hasattr(tokens, 'item') else tokens]])
        elif len(original_shape) == 1:  # Vector [seq_len]
            batch_size, seq_len = 1, original_shape[0]
            tokens = mx.expand_dims(tokens, axis=0)  # [1, seq_len]
        elif len(original_shape) == 2:  # Matrix [batch_size, seq_len]
            batch_size, seq_len = original_shape
        elif len(original_shape) == 3 and original_shape[2] == 1:  # 3D with final dim 1
            batch_size, seq_len = original_shape[0], original_shape[1]
            tokens = tokens[:, :, 0]  # Remove final dimension
        else:
            # Unexpected shape, try best effort reshape
            if self.debug:
                print(f"Unexpected token shape: {original_shape}")
            total_elements = np.prod(original_shape)
            batch_size, seq_len = 1, total_elements
            tokens = tokens.reshape(batch_size, seq_len)
            
        try:
            # Process each token individually to avoid reshape issues
            embeddings = mx.zeros((batch_size, seq_len, self.embed_dim))
            
            for b in range(batch_size):
                for s in range(seq_len):
                    try:
                        # Get the token ID, handling potential shape issues
                        if batch_size == 1 and seq_len == 1 and len(original_shape) == 0:
                            # Scalar case
                            token_id = tokens.item() if hasattr(tokens, 'item') else int(tokens)
                        else:
                            # Normal case
                            token_id = tokens[b, s].item() if hasattr(tokens[b, s], 'item') else int(tokens[b, s])
                        
                        # Look up embedding for this token, handle out of bounds
                        if 0 <= token_id < self.text_embeddings.shape[0]:
                            token_embedding = self.text_embeddings[token_id]
                            # Copy embedding values to avoid reshape issues
                            for i in range(self.embed_dim):
                                if i < len(token_embedding):
                                    embeddings = embeddings.at[b, s, i].set(token_embedding[i])
                    except Exception as e:
                        if self.debug:
                            print(f"Error embedding token at position ({b},{s}): {e}")
            
            return embeddings
        except Exception as e:
            if self.debug:
                print(f"MLX text embedding error: {e}")
                print(f"Input shape: {tokens.shape}, Embeddings shape: {self.text_embeddings.shape}")
            
            # Create zeros as fallback
            return mx.zeros((batch_size, seq_len, self.embed_dim))
    
    def embed_audio(self, tokens: mx.array, codebook: int) -> mx.array:
        """
        Embed audio tokens for a specific codebook with careful shape handling.
        
        Args:
            tokens: Audio tokens with shape [batch_size, seq_len]
            codebook: Codebook index
            
        Returns:
            Audio embeddings with shape [batch_size, seq_len, embed_dim]
        """
        if self.audio_embeddings is None:
            raise ValueError("Audio embeddings not available")
            
        # Handle input shape carefully
        original_shape = tokens.shape
        
        # Get dimensions
        if len(original_shape) == 0:  # Scalar
            batch_size, seq_len = 1, 1
            tokens = mx.array([[tokens.item() if hasattr(tokens, 'item') else tokens]])
        elif len(original_shape) == 1:  # Vector [seq_len]
            batch_size, seq_len = 1, original_shape[0]
            tokens = mx.expand_dims(tokens, axis=0)  # [1, seq_len]
        elif len(original_shape) == 2:  # Matrix [batch_size, seq_len]
            batch_size, seq_len = original_shape
        elif len(original_shape) == 3 and original_shape[2] == 1:  # 3D with final dim 1
            batch_size, seq_len = original_shape[0], original_shape[1]
            tokens = tokens[:, :, 0]  # Remove final dimension
        else:
            # Unexpected shape, try best effort reshape
            if self.debug:
                print(f"Unexpected token shape for audio codebook {codebook}: {original_shape}")
            total_elements = np.prod(original_shape)
            batch_size, seq_len = 1, total_elements
            tokens = tokens.reshape(batch_size, seq_len)
        
        # Calculate offset based on codebook
        offset = codebook * self.audio_vocab_size
            
        try:
            # Process each token individually to avoid reshape issues
            embeddings = mx.zeros((batch_size, seq_len, self.embed_dim))
            
            for b in range(batch_size):
                for s in range(seq_len):
                    try:
                        # Get the token ID, handling potential shape issues
                        if batch_size == 1 and seq_len == 1 and len(original_shape) == 0:
                            # Scalar case
                            token_id = tokens.item() if hasattr(tokens, 'item') else int(tokens)
                        else:
                            # Normal case
                            token_id = tokens[b, s].item() if hasattr(tokens[b, s], 'item') else int(tokens[b, s])
                        
                        # Apply offset
                        token_id_with_offset = token_id + offset
                        
                        # Look up embedding for this token, handle out of bounds
                        if 0 <= token_id_with_offset < self.audio_embeddings.shape[0]:
                            token_embedding = self.audio_embeddings[token_id_with_offset]
                            # Copy embedding values to avoid reshape issues
                            for i in range(self.embed_dim):
                                if i < len(token_embedding):
                                    embeddings = embeddings.at[b, s, i].set(token_embedding[i])
                    except Exception as e:
                        if self.debug:
                            print(f"Error embedding audio token at position ({b},{s}) for codebook {codebook}: {e}")
            
            return embeddings
        except Exception as e:
            if self.debug:
                print(f"MLX audio embedding error for codebook {codebook}: {e}")
                print(f"Input shape: {tokens.shape}, Embeddings shape: {self.audio_embeddings.shape}")
            
            # Create zeros as fallback
            return mx.zeros((batch_size, seq_len, self.embed_dim))


def mlx_sample_topk(logits: mx.array, topk: int = 5, temperature: float = 1.0) -> mx.array:
    """
    Sample from logits using MLX with top-k sampling and robust shape handling.
    
    Args:
        logits: Raw logits with shape [batch_size, vocab_size]
        topk: Number of top candidates to consider
        temperature: Temperature for sampling
        
    Returns:
        Sampled tokens with shape [batch_size, 1]
    """
    # Ensure proper input shape
    if len(logits.shape) == 1:
        logits = logits.reshape(1, -1)
        
    batch_size, vocab_size = logits.shape
    
    # Apply temperature
    scaled_logits = logits / max(temperature, 1e-5)
    
    # Find top-k values and indices
    values, indices = mx.topk(scaled_logits, min(topk, vocab_size))
    
    # Convert to probabilities
    probs = mx.softmax(values, axis=-1)
    
    # Sample from the categorical distribution
    samples = mx.zeros((batch_size, 1), dtype=mx.int32)
    
    # Create separate random keys for each batch item
    batch_keys = [mx.random.key(np.random.randint(0, 2**32)) for _ in range(batch_size)]
    
    # Sample for each batch
    for b in range(batch_size):
        # Create distribution for this batch item
        current_probs = probs[b].reshape(1, -1)
        
        # Sample from the distribution
        sample_pos = mx.random.categorical(batch_keys[b], current_probs)
        
        # Get the sampled index within our top-k set
        sample_idx = sample_pos.item() if hasattr(sample_pos, 'item') else sample_pos.reshape(-1)[0]
        
        # Map back to original vocabulary
        token_id = indices[b, sample_idx]
        
        # Store the result
        samples = samples.at[b, 0].set(token_id)
    
    return samples


def mlx_sample_categorical(logits: mx.array, temperature: float = 1.0) -> mx.array:
    """
    Sample from logits using MLX with categorical sampling and robust shape handling.
    
    Args:
        logits: Raw logits with shape [batch_size, vocab_size]
        temperature: Temperature for sampling
        
    Returns:
        Sampled tokens with shape [batch_size, 1]
    """
    # Ensure proper input shape
    if len(logits.shape) == 1:
        logits = logits.reshape(1, -1)
        
    batch_size, vocab_size = logits.shape
    
    # Apply temperature
    scaled_logits = logits / max(temperature, 1e-5)
    
    # Convert to probabilities
    probs = mx.softmax(scaled_logits, axis=-1)
    
    # Sample using a separate key for each batch item
    samples = mx.zeros((batch_size, 1), dtype=mx.int32)
    
    # Create separate random keys for each batch item
    for b in range(batch_size):
        # Create key and sample
        key = mx.random.key(np.random.randint(0, 2**32))
        
        # Get only this batch item's probabilities
        batch_probs = probs[b:b+1]
        
        # Sample from categorical distribution
        sample = mx.random.categorical(key, batch_probs)
        
        # Store the result with proper shape handling
        samples = samples.at[b, 0].set(sample.item())
    
    return samples