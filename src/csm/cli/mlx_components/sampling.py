"""
Sampling functions for MLX acceleration.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch

# Import MLX if available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    # Create a dummy module
    class DummyMX:
        def __getattr__(self, name):
            raise ImportError("MLX is not available")
    mx = DummyMX()

def mlx_topk_sampling(
    logits: mx.array, 
    k: int = 5, 
    temperature: float = 1.0,
    seed: Optional[int] = None
) -> mx.array:
    """
    Sample from logits using top-k sampling with MLX.
    
    Args:
        logits: Logits to sample from [batch_size, vocab_size]
        k: Number of top candidates to sample from
        temperature: Temperature for sampling
        seed: Random seed for reproducibility
        
    Returns:
        Sampled indices with shape [batch_size, 1]
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX is not available for sampling")
    
    # Apply temperature
    scaled_logits = logits / max(temperature, 1e-5)
    
    # Get dimensions
    if len(scaled_logits.shape) == 1:
        # [vocab_size] -> [1, vocab_size]
        batch_size = 1
        vocab_size = scaled_logits.shape[0]
        scaled_logits = mx.expand_dims(scaled_logits, axis=0)
    else:
        batch_size, vocab_size = scaled_logits.shape
    
    # Keep track of whether our sampling succeeded
    success = False
    
    # Try using MLX operations
    try:
        # Get top-k values and indices
        top_values, top_indices = mx.topk(scaled_logits, min(k, vocab_size))
        
        # Apply softmax to get probabilities
        top_probs = mx.softmax(top_values, axis=-1)
        
        # Sample from categorical distribution
        if seed is not None:
            rng_key = mx.random.key(seed)
        else:
            rng_key = mx.random.key(np.random.randint(0, 2**32))
            
        # Sample indices from top-k
        sample_indices = mx.random.categorical(rng_key, top_probs)
        
        # Gather the actual token indices
        batch_indices = mx.arange(batch_size)
        samples = top_indices[batch_indices, sample_indices]
        
        # Reshape to [batch_size, 1]
        samples = samples.reshape(batch_size, 1)
        success = True
        
    except Exception as e:
        print(f"MLX top-k sampling failed: {e}")
        # Fall back to NumPy sampling
        try:
            samples = np.zeros((batch_size, 1), dtype=np.int32)
            
            # Process each batch item
            for b in range(batch_size):
                # Get logits for this batch
                batch_logits = scaled_logits[b].to_numpy()
                
                # Get top-k indices
                top_indices = np.argsort(batch_logits)[-k:]
                
                # Get top-k values
                top_values = batch_logits[top_indices]
                
                # Apply softmax
                top_probs = np.exp(top_values) / np.sum(np.exp(top_values))
                
                # Sample
                if seed is not None:
                    np.random.seed(seed)
                    
                sample_idx = np.random.choice(len(top_indices), p=top_probs)
                samples[b, 0] = top_indices[sample_idx]
                
            # Convert back to MLX
            samples = mx.array(samples)
            success = True
            
        except Exception as numpy_e:
            print(f"NumPy fallback sampling failed: {numpy_e}")
            # Return zeros as last resort
            samples = mx.zeros((batch_size, 1), dtype=mx.int32)
    
    return samples, success

def mlx_categorical_sampling(
    logits: mx.array, 
    temperature: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[mx.array, bool]:
    """
    Sample from logits using categorical sampling with MLX.
    
    Args:
        logits: Logits to sample from [batch_size, vocab_size]
        temperature: Temperature for sampling
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled indices with shape [batch_size, 1], success flag)
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX is not available for sampling")
    
    # Apply temperature
    scaled_logits = logits / max(temperature, 1e-5)
    
    # Get dimensions
    if len(scaled_logits.shape) == 1:
        # [vocab_size] -> [1, vocab_size]
        batch_size = 1
        vocab_size = scaled_logits.shape[0]
        scaled_logits = mx.expand_dims(scaled_logits, axis=0)
    else:
        batch_size, vocab_size = scaled_logits.shape
    
    # Convert to probabilities
    probs = mx.softmax(scaled_logits, axis=-1)
    
    # Keep track of whether our sampling succeeded
    success = False
    
    # Try using MLX operations
    try:
        # Create random key
        if seed is not None:
            rng_key = mx.random.key(seed)
        else:
            rng_key = mx.random.key(np.random.randint(0, 2**32))
            
        # Sample from categorical distribution
        samples = mx.random.categorical(rng_key, probs)
        
        # Reshape to [batch_size, 1]
        samples = samples.reshape(batch_size, 1)
        success = True
        
    except Exception as e:
        print(f"MLX categorical sampling failed: {e}")
        print(f"!!!!! DEBUG: probs.shape={probs.shape}")
        
        # Fall back to NumPy sampling
        try:
            # Convert to NumPy for sampling
            np_probs = probs.to_numpy()
            samples = np.zeros((batch_size, 1), dtype=np.int32)
            
            # Set random seed if provided
            if seed is not None:
                np.random.seed(seed)
                
            # Sample for each batch item
            for b in range(batch_size):
                # Get probabilities for this batch
                batch_probs = np_probs[b]
                
                # Normalize to ensure it sums to 1
                batch_probs = batch_probs / np.sum(batch_probs)
                
                # Sample
                sample = np.random.choice(vocab_size, p=batch_probs)
                samples[b, 0] = sample
                
            # Convert back to MLX
            samples = mx.array(samples)
            success = True
            
        except Exception as numpy_e:
            print(f"NumPy fallback sampling failed: {numpy_e}")
            
            # Final fallback - just pick highest probability
            try:
                samples = mx.argmax(probs, axis=-1).reshape(batch_size, 1)
                success = True
            except:
                # Return zeros as last resort
                samples = mx.zeros((batch_size, 1), dtype=mx.int32)
    
    return samples, success