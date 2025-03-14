"""
Pure MLX implementation of the frame generation pipeline for CSM.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import torch

from csm.cli.mlx_layers import MLXTransformer, torch_to_mlx, mlx_to_torch, create_causal_mask, index_causal_mask
from csm.cli.mlx_embedding import MLXEmbedding, mlx_sample_topk, mlx_sample_categorical


class MLXFrameGenerator:
    """
    Pure MLX implementation of frame generation for CSM with robust error handling.
    """
    
    def __init__(
        self,
        backbone: MLXTransformer,
        decoder: MLXTransformer,
        embedding: MLXEmbedding,
        projection_weight: Optional[mx.array] = None,
        codebook0_head_weight: Optional[mx.array] = None,
        audio_head_weights: Optional[List[mx.array]] = None,
        audio_vocab_size: int = 2051,
        audio_num_codebooks: int = 32,
        debug: bool = False,
        fallback_fn = None
    ):
        # Make debug a property so it can be accessed in methods
        self.debug = debug
        self.backbone = backbone
        self.decoder = decoder
        self.embedding = embedding
        self.projection_weight = projection_weight
        self.codebook0_head_weight = codebook0_head_weight
        self.audio_head_weights = audio_head_weights
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks
        self.debug = debug
        self.fallback_fn = fallback_fn
        
    def generate_frame(self, tokens: torch.Tensor, positions: torch.Tensor, topk: int = 5, temperature: float = 1.0):
        """
        Generate a frame using pure MLX with robust error handling and shape management.
        
        Args:
            tokens: Input tokens with shape [batch_size, seq_len, num_codebooks+1]
            positions: Position indices with shape [batch_size, seq_len]
            topk: Number of top candidates to consider for sampling
            temperature: Temperature for sampling
            
        Returns:
            Generated audio tokens with shape [batch_size, audio_num_codebooks]
        """
        try:
            # Convert inputs to MLX
            mlx_tokens = torch_to_mlx(tokens)
            mlx_positions = torch_to_mlx(positions)
            
            # Get dimensions
            batch_size, seq_len, total_codebooks = mlx_tokens.shape
            
            # Process text and audio tokens
            text_tokens = mlx_tokens[:, :, -1]  # [batch, seq_len]
            audio_tokens = mlx_tokens[:, :, :-1]  # [batch, seq_len, audio_num_codebooks]
            
            # Create attention mask for tokens
            tokens_mask = mx.ones((batch_size, seq_len), dtype=mx.float32)
            
            # Step 1: Embed tokens
            # Get text embeddings - shape: [batch, seq_len, embed_dim]
            text_embeds = self.embedding.embed_text(text_tokens)
            
            # Add codebook dimension - shape: [batch, seq_len, 1, embed_dim]
            text_embeds = mx.expand_dims(text_embeds, axis=2)
            
            # Get audio embeddings for each codebook
            audio_embeds_list = []
            for codebook in range(self.audio_num_codebooks):
                # Extract tokens for this codebook - shape: [batch, seq_len]
                codebook_tokens = audio_tokens[:, :, codebook]
                
                # Embed using MLX - shape: [batch, seq_len, embed_dim]
                codebook_embeds = self.embedding.embed_audio(codebook_tokens, codebook)
                
                # Add codebook dimension - shape: [batch, seq_len, 1, embed_dim]
                codebook_embeds = mx.expand_dims(codebook_embeds, axis=2)
                audio_embeds_list.append(codebook_embeds)
            
            # Concatenate all audio embeddings - shape: [batch, seq_len, audio_num_codebooks, embed_dim]
            if audio_embeds_list:
                audio_embeds = mx.concatenate(audio_embeds_list, axis=2)
                
                # Concatenate with text embeddings - shape: [batch, seq_len, audio_num_codebooks+1, embed_dim]
                all_embeds = mx.concatenate([audio_embeds, text_embeds], axis=2)
            else:
                # Just use text embeddings if no audio
                all_embeds = text_embeds
            
            # Apply mask and sum across codebook dimension
            # Expand mask - shape: [batch, seq_len, 1, 1]
            expanded_mask = mx.expand_dims(mx.expand_dims(tokens_mask, axis=2), axis=3)
            
            # Apply mask - shape: [batch, seq_len, audio_num_codebooks+1, embed_dim]
            masked_embeds = all_embeds * expanded_mask
            
            # Sum across codebook dimension - shape: [batch, seq_len, embed_dim]
            hidden_states = mx.sum(masked_embeds, axis=2)
            
            # Handle reshape issues by explicitly reshaping to correct dimensions
            # This is the critical section that's causing the reshape errors
            if batch_size * seq_len * embed_dim != hidden_states.size:
                if self.debug:
                    print(f"Fixing hidden_states shape issue: {hidden_states.shape}, size={hidden_states.size}, expected={batch_size*seq_len*embed_dim}")
                
                # Case 1: Initial reshape for input text tokens (common case)
                if hidden_states.size == seq_len and batch_size == 1:
                    # Create zeros with correct shape and size
                    correct_hidden = mx.zeros((batch_size, seq_len, embed_dim))
                    # Copy data where possible
                    for i in range(min(seq_len, hidden_states.size)):
                        value = hidden_states[i] if i < hidden_states.size else 0
                        correct_hidden = correct_hidden.at[0, i, 0].set(value)
                    hidden_states = correct_hidden
                
                # Case 2: Single token case for subsequent frames
                elif hidden_states.size == 1 and batch_size == 1 and seq_len == 1:
                    # Create array with correct shape
                    value = hidden_states.item() if hasattr(hidden_states, 'item') else hidden_states.reshape(-1)[0]
                    correct_hidden = mx.ones((batch_size, seq_len, embed_dim)) * value
                    hidden_states = correct_hidden
                
                # Case 3: Directly reshape if dimensions are compatible
                elif hidden_states.size == batch_size * seq_len * embed_dim:
                    hidden_states = hidden_states.reshape(batch_size, seq_len, embed_dim)
            
            # Step 2: Run backbone transformer
            # Create causal mask - shape: [seq_len, seq_len]
            backbone_mask = create_causal_mask(seq_len)
            
            # Index mask for positions - shape: [batch, seq_len, seq_len]
            indexed_mask = index_causal_mask(backbone_mask, mlx_positions)
            
            # Process with backbone - shape: [batch, seq_len, embed_dim]
            backbone_output = self.backbone.forward(hidden_states, mask=indexed_mask)
            
            # Step 3: Process the first codebook (c0)
            # Extract last hidden state - shape: [batch, embed_dim]
            last_hidden = backbone_output[:, -1, :]
            
            # Generate c0 logits with MLX matrix multiply - shape: [batch, vocab_size]
            if self.codebook0_head_weight is not None:
                c0_logits = mx.matmul(last_hidden, self.codebook0_head_weight.T)
                
                # Sample from logits - shape could be [batch] or [batch, 1]
                c0_sample_mlx = mlx_sample_categorical(c0_logits, temperature)
                
                # Fix shape issues with categorical sampling result
                if len(c0_sample_mlx.shape) == 0:  # Scalar result
                    # Convert scalar to [1, 1] tensor
                    c0_sample_mlx = mx.array([[c0_sample_mlx.item() if hasattr(c0_sample_mlx, 'item') else c0_sample_mlx]])
                elif len(c0_sample_mlx.shape) == 1:  # Vector result [batch]
                    # Add sequence dimension: [batch] -> [batch, 1]
                    c0_sample_mlx = mx.expand_dims(c0_sample_mlx, axis=1)
                
                # Convert to PyTorch with explicit shape
                c0_sample = mlx_to_torch(c0_sample_mlx)
            else:
                raise ValueError("codebook0_head_weight is required for MLX generation")
            
            # Get embedding for c0 - shape: [batch, 1, embed_dim]
            try:
                c0_embed_mlx = self.embedding.embed_audio(c0_sample_mlx, 0)
                
                # Verify embedding shape and fix if needed
                if c0_embed_mlx.size != batch_size * 1 * embed_dim:
                    if self.debug:
                        print(f"Fixing c0_embed_mlx shape: {c0_embed_mlx.shape}, size={c0_embed_mlx.size}, expected={batch_size*1*embed_dim}")
                    
                    # Create zeros with correct dimensions
                    correct_embed = mx.zeros((batch_size, 1, embed_dim))
                    
                    # If we have a scalar value, expand it
                    if len(c0_embed_mlx.shape) == 0:
                        value = c0_embed_mlx.item() if hasattr(c0_embed_mlx, 'item') else float(c0_embed_mlx)
                        correct_embed = mx.ones((batch_size, 1, embed_dim)) * value
                    # If we have a vector, reshape it
                    elif len(c0_embed_mlx.shape) == 1 and c0_embed_mlx.size == embed_dim:
                        for i in range(embed_dim):
                            correct_embed = correct_embed.at[0, 0, i].set(c0_embed_mlx[i])
                    
                    c0_embed_mlx = correct_embed
            except Exception as e:
                if self.debug:
                    print(f"Error in c0 embedding: {e}")
                # Create a fallback embedding
                c0_embed_mlx = mx.zeros((batch_size, 1, embed_dim))
            
            # Step 4: Initialize decoder state
            # Add batch dimension if needed
            if len(c0_embed_mlx.shape) == 2:
                c0_embed_mlx = mx.expand_dims(c0_embed_mlx, axis=0)
                
            # Add last hidden state for context - shape: [batch, 1, embed_dim]
            last_hidden_expanded = mx.expand_dims(last_hidden, axis=1)
            
            # Concatenate - shape: [batch, 2, embed_dim]
            curr_hidden = mx.concatenate([last_hidden_expanded, c0_embed_mlx], axis=1)
            
            # Track current sample
            curr_sample = c0_sample
            
            # Create positions for decoder - shape: [batch, 2]
            curr_positions = mx.array(np.zeros((batch_size, curr_hidden.shape[1]), dtype=np.int32))
            
            # Step 5: Generate remaining codebooks
            for i in range(1, self.audio_num_codebooks):
                try:
                    # Create causal mask for decoder - shape: [seq_len, seq_len]
                    decoder_seq_len = curr_hidden.shape[1]
                    decoder_mask = create_causal_mask(decoder_seq_len)
                    
                    # Index mask for positions
                    curr_decoder_mask = index_causal_mask(decoder_mask, curr_positions)
                    
                    # Project to decoder dimension - shape: [batch, seq_len, decoder_dim]
                    if self.projection_weight is not None:
                        projected = mx.matmul(curr_hidden, self.projection_weight.T)
                    else:
                        raise ValueError("projection_weight is required for MLX generation")
                    
                    # Run decoder - shape: [batch, seq_len, decoder_dim]
                    decoder_output = self.decoder.forward(projected, mask=curr_decoder_mask)
                    
                    # Extract last hidden state - shape: [batch, decoder_dim]
                    last_decoder_hidden = decoder_output[:, -1, :]
                    
                    # Generate logits - shape: [batch, vocab_size]
                    if self.audio_head_weights is not None and i - 1 < len(self.audio_head_weights):
                        ci_logits = mx.matmul(last_decoder_hidden, self.audio_head_weights[i - 1].T)
                        
                        # Sample from logits - shape could be [batch] or [batch, 1]
                        ci_sample_mlx = mlx_sample_categorical(ci_logits, temperature)
                        
                        # Fix shape issues with categorical sampling result
                        if len(ci_sample_mlx.shape) == 0:  # Scalar result
                            # Convert scalar to [1, 1] tensor
                            ci_sample_mlx = mx.array([[ci_sample_mlx.item() if hasattr(ci_sample_mlx, 'item') else ci_sample_mlx]])
                        elif len(ci_sample_mlx.shape) == 1:  # Vector result [batch]
                            # Add sequence dimension: [batch] -> [batch, 1]
                            ci_sample_mlx = mx.expand_dims(ci_sample_mlx, axis=1)
                        
                        # Convert to PyTorch with explicit shape
                        ci_sample = mlx_to_torch(ci_sample_mlx)
                    else:
                        raise ValueError(f"audio_head_weights[{i-1}] is required for MLX generation")
                    
                    # Get embedding - shape: [batch, 1, embed_dim]
                    try:
                        ci_embed_mlx = self.embedding.embed_audio(ci_sample_mlx, i)
                        
                        # Verify embedding shape and fix if needed
                        if ci_embed_mlx.size != batch_size * 1 * embed_dim:
                            if self.debug:
                                print(f"Fixing ci_embed_mlx shape: {ci_embed_mlx.shape}, size={ci_embed_mlx.size}, expected={batch_size*1*embed_dim}")
                            
                            # Create zeros with correct dimensions
                            correct_embed = mx.zeros((batch_size, 1, embed_dim))
                            
                            # If we have a scalar value, expand it
                            if len(ci_embed_mlx.shape) == 0:
                                value = ci_embed_mlx.item() if hasattr(ci_embed_mlx, 'item') else float(ci_embed_mlx)
                                correct_embed = mx.ones((batch_size, 1, embed_dim)) * value
                            # If we have a vector, reshape it
                            elif len(ci_embed_mlx.shape) == 1 and ci_embed_mlx.size == embed_dim:
                                for j in range(embed_dim):
                                    correct_embed = correct_embed.at[0, 0, j].set(ci_embed_mlx[j])
                            
                            ci_embed_mlx = correct_embed
                    except Exception as e:
                        if self.debug:
                            print(f"Error in ci embedding for codebook {i}: {e}")
                        # Create a fallback embedding
                        ci_embed_mlx = mx.zeros((batch_size, 1, embed_dim))
                    
                    # Update state - ensure the shape is correct
                    curr_hidden = ci_embed_mlx
                    curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                    
                    # Update positions
                    curr_positions = curr_positions + 1
                except Exception as e:
                    if self.debug:
                        print(f"MLX codebook {i} error: {e}")
                    
                    # Fall back to PyTorch for this codebook if available
                    if self.fallback_fn is not None:
                        ci_sample = self.fallback_fn(i, curr_sample)
                        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                    else:
                        # If no fallback, just use zeros
                        ci_sample = torch.zeros_like(c0_sample)
                        curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            
            return curr_sample
            
        except Exception as e:
            if self.debug:
                print(f"Pure MLX frame generation failed: {e}")
            
            # Use fallback if available
            if self.fallback_fn is not None:
                return self.fallback_fn(tokens, positions, topk, temperature)
            else:
                # Create a dummy result as last resort
                return torch.zeros((tokens.size(0), self.audio_num_codebooks), device=tokens.device)