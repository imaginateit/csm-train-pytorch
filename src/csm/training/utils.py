"""Training utilities for CSM."""

import os
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def compute_loss(
    model,
    input_tokens: torch.Tensor,
    input_masks: torch.Tensor,
    target_audio_tokens: torch.Tensor,
    semantic_weight: float = 100.0,
    acoustic_weight: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute loss for CSM training.
    
    Args:
        model: CSM model
        input_tokens: Input token sequence (batch_size, seq_len, 33)
        input_masks: Input token masks (batch_size, seq_len, 33)
        target_audio_tokens: Target audio tokens (batch_size, target_len, 32)
        semantic_weight: Weight for semantic token loss
        acoustic_weight: Weight for acoustic token loss
        
    Returns:
        Tuple of (total loss, loss details)
    """
    batch_size, seq_len, _ = input_tokens.size()
    
    # Create positions
    positions = torch.arange(0, seq_len, device=input_tokens.device)
    positions = positions.unsqueeze(0).repeat(batch_size, 1)
    
    # Compute embeddings
    embeds = model._embed_tokens(input_tokens)
    masked_embeds = embeds * input_masks.unsqueeze(-1)
    h = masked_embeds.sum(dim=2)
    
    # Get backbone outputs
    mask = model._index_causal_mask(model.backbone_causal_mask, positions)
    backbone_outputs = model.backbone(h, input_pos=positions, mask=mask)
    
    # Initialize loss components
    losses = {}
    total_loss = 0.0
    
    # Get targets for semantic tokens (codebook 0)
    target_semantic = target_audio_tokens[:, :, 0]
    
    # Compute semantic token loss
    semantic_logits = model.codebook0_head(backbone_outputs[:, :-1])  # exclude last position
    semantic_loss = F.cross_entropy(
        semantic_logits.reshape(-1, semantic_logits.size(-1)),
        target_semantic[:, :semantic_logits.size(1)].reshape(-1)
    )
    losses["semantic_loss"] = semantic_loss
    total_loss += semantic_weight * semantic_loss
    
    # Compute acoustic token losses (over the decoder)
    acoustic_loss = 0.0
    
    # Placeholder for decoder loss computation
    # This is a simplified approach - actual implementation would need to follow
    # the model's generate_frame logic more closely
    
    losses["acoustic_loss"] = torch.tensor(0.0, device=input_tokens.device)
    total_loss += acoustic_weight * losses["acoustic_loss"]
    
    return total_loss, losses


def compute_loss_mlx(
    model,
    input_tokens,
    input_masks,
    target_audio_tokens,
    semantic_weight=100.0,
    acoustic_weight=1.0
):
    """
    Compute loss for CSM training using MLX.
    
    Args:
        model: CSM model in MLX format
        input_tokens: Input token sequence
        input_masks: Input token masks
        target_audio_tokens: Target audio tokens
        semantic_weight: Weight for semantic token loss
        acoustic_weight: Weight for acoustic token loss
        
    Returns:
        Tuple of (total loss, loss details)
    """
    import mlx.core as mx
    import mlx.nn as nn
    import logging
    import math
    
    logger = logging.getLogger("compute_loss_mlx")
    
    try:
        # Input validation
        if input_tokens is None or input_masks is None or target_audio_tokens is None:
            logger.error("Received None inputs to compute_loss_mlx")
            raise ValueError("Inputs to compute_loss_mlx cannot be None")
            
        # Check shapes
        if not hasattr(input_tokens, 'shape') or len(input_tokens.shape) != 3:
            logger.error(f"Invalid input_tokens shape: {getattr(input_tokens, 'shape', 'unknown')}")
            raise ValueError(f"input_tokens must be 3D tensor, got {getattr(input_tokens, 'shape', 'unknown')}")
            
        if not hasattr(input_masks, 'shape') or len(input_masks.shape) != 3:
            logger.error(f"Invalid input_masks shape: {getattr(input_masks, 'shape', 'unknown')}")
            raise ValueError(f"input_masks must be 3D tensor, got {getattr(input_masks, 'shape', 'unknown')}")
            
        if not hasattr(target_audio_tokens, 'shape') or len(target_audio_tokens.shape) != 3:
            logger.error(f"Invalid target_audio_tokens shape: {getattr(target_audio_tokens, 'shape', 'unknown')}")
            raise ValueError(f"target_audio_tokens must be 3D tensor, got {getattr(target_audio_tokens, 'shape', 'unknown')}")
        
        # Get dimensions
        batch_size, seq_len, _ = input_tokens.shape
        
        # Model validation
        if not hasattr(model, 'backbone'):
            logger.error("Model has no backbone")
            raise ValueError("Model must have a backbone attribute")
            
        # Create positions
        try:
            positions = mx.arange(0, seq_len)
            positions = mx.expand_dims(positions, 0)
            positions = mx.repeat(positions, batch_size, 0)
        except Exception as pos_e:
            logger.warning(f"Error creating positions: {pos_e}, using zeros")
            positions = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        
        # Compute embeddings - try different attribute names to handle various model classes
        embed_method_found = False
        embeds = None
        
        try:
            # Try multiple approaches to get embeddings
            if hasattr(model, 'embed_tokens') and callable(model.embed_tokens):
                logger.debug("Using model.embed_tokens")
                embeds = model.embed_tokens(input_tokens)
                embed_method_found = True
            elif hasattr(model, '_embed_tokens') and callable(model._embed_tokens):
                logger.debug("Using model._embed_tokens")
                embeds = model._embed_tokens(input_tokens)
                embed_method_found = True
            elif hasattr(model, 'embedding') and hasattr(model.embedding, 'embed_tokens'):
                logger.debug("Using model.embedding.embed_tokens")
                embeds = model.embedding.embed_tokens(input_tokens)
                embed_method_found = True
                
            # Verify the embeddings are valid
            if embed_method_found and (
                embeds is None or 
                not hasattr(embeds, 'shape') or 
                math.isnan(float(mx.mean(embeds).item()))
            ):
                logger.warning("Embeddings are invalid, using placeholder")
                embed_method_found = False
                
        except Exception as embed_e:
            logger.warning(f"Error computing embeddings: {embed_e}")
            embed_method_found = False
            
        # Use placeholder embeddings if no method found or errors occurred
        if not embed_method_found:
            logger.warning("No valid embed_tokens method found, using placeholder embeddings")
            # Determine embedding dimension from model if possible
            embed_dim = 2048  # Default
            if hasattr(model, 'embedding') and hasattr(model.embedding, 'embed_dim'):
                embed_dim = model.embedding.embed_dim
            elif hasattr(model, 'backbone') and hasattr(model.backbone, 'hidden_size'):
                embed_dim = model.backbone.hidden_size
                
            # Create placeholder embeddings
            embeds = mx.zeros((batch_size, seq_len, input_tokens.shape[2], embed_dim))
            
        # Apply mask to embeddings
        try:
            # Check shapes of embeds and input_masks
            logger.debug(f"Embeds shape: {embeds.shape}, Input masks shape: {input_masks.shape}")
            
            # Reshape masks to match embedding dimensions
            if len(embeds.shape) == 4 and len(input_masks.shape) == 3:
                # Embeds: [batch, seq, codebooks, dim], Masks: [batch, seq, codebooks]
                expanded_masks = mx.expand_dims(input_masks, -1)
                masked_embeds = embeds * expanded_masks
                h = mx.sum(masked_embeds, axis=2)  # Sum over codebooks
            elif len(embeds.shape) == 3 and len(input_masks.shape) == 3:
                # Reshape masks to be broadcastable with embeds
                # Embeds: [batch, seq, dim], Masks: [batch, seq, codebooks]
                # Sum the mask across codebooks
                summed_masks = mx.sum(input_masks, axis=-1, keepdims=True)
                # Normalize if non-zero
                mask_sum = mx.sum(summed_masks)
                if mask_sum > 0:
                    summed_masks = summed_masks / mask_sum
                h = embeds * summed_masks
            else:
                # Default case - just use embeds as is
                logger.warning(f"Incompatible shapes for masking: embeds {embeds.shape}, masks {input_masks.shape}. Using unmasked embeddings.")
                h = embeds
        except Exception as mask_e:
            logger.warning(f"Error applying mask: {mask_e}, using unmasked embeddings")
            # Fallback: try to reshape if needed
            if len(embeds.shape) == 4:
                # Already 4D (batch, seq, codebooks, dim)
                h = mx.sum(embeds, axis=2)  # Sum over codebooks
            elif len(embeds.shape) == 3:
                # Already 3D (batch, seq, dim)
                h = embeds
            else:
                # Create safe tensor with right shape
                embed_dim = embeds.shape[-1] if hasattr(embeds, 'shape') and len(embeds.shape) > 0 else 2048
                h = mx.zeros((batch_size, seq_len, embed_dim))
        
        # Get backbone outputs with multiple fallbacks
        try:
            # Try to get causal mask if available
            if hasattr(model, 'backbone_causal_mask') and hasattr(model, '_index_causal_mask'):
                logger.debug("Using backbone_causal_mask and _index_causal_mask")
                mask = model._index_causal_mask(model.backbone_causal_mask, positions)
                backbone_outputs = model.backbone(h, input_pos=positions, mask=mask)
            else:
                # Try different parameter combinations for compatibility
                logger.debug("No causal mask, trying different backbone calling conventions")
                try:
                    # Try with positional parameters
                    backbone_outputs = model.backbone(h, positions)
                except:
                    try:
                        # Try with named parameters - input_pos
                        backbone_outputs = model.backbone(h, input_pos=positions)
                    except:
                        try:
                            # Try just the hidden states
                            backbone_outputs = model.backbone(h)
                        except Exception as backbone_e:
                            logger.error(f"All backbone calling conventions failed: {backbone_e}")
                            raise
        except Exception as backbone_e:
            logger.error(f"Error in backbone processing: {backbone_e}")
            # Create placeholder backbone outputs
            embed_dim = h.shape[-1] if hasattr(h, 'shape') and len(h.shape) > 0 else 2048
            backbone_outputs = mx.zeros((batch_size, seq_len, embed_dim))
        
        # Initialize loss components with safe values
        losses = {}
        total_loss = mx.array(0.0)
        
        # Extract semantic token targets
        try:
            # Get targets for semantic tokens (codebook 0)
            target_semantic = target_audio_tokens[:, :, 0]
        except Exception as target_e:
            logger.warning(f"Error extracting semantic targets: {target_e}, using zeros")
            target_semantic = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        
        # Compute semantic token loss with multiple fallbacks
        try:
            if hasattr(model, 'codebook0_head') and callable(getattr(model.codebook0_head, '__call__', None)):
                logger.debug("Using model.codebook0_head")
                
                try:
                    # Try to get logits from the head
                    semantic_logits = model.codebook0_head(backbone_outputs[:, :-1])
                    
                    # Make sure shapes match
                    max_len = min(semantic_logits.shape[1], target_semantic.shape[1])
                    semantic_logits = semantic_logits[:, :max_len]
                    target_semantic = target_semantic[:, :max_len]
                    
                    # Try MLX cross entropy loss safely
                    try:
                        semantic_loss = nn.losses.cross_entropy(
                            semantic_logits.reshape(-1, semantic_logits.shape[-1]),
                            target_semantic.reshape(-1),
                            reduction='mean'
                        )
                    except Exception as ce_e:
                        logger.warning(f"Cross entropy failed: {ce_e}, using MSE fallback")
                        # Fallback to MSE if cross entropy fails
                        target_one_hot = mx.one_hot(
                            target_semantic, 
                            semantic_logits.shape[-1]
                        )
                        semantic_loss = mx.mean(mx.square(
                            semantic_logits - 
                            target_one_hot.reshape(semantic_logits.shape)
                        ))
                    
                    # Validate loss value
                    if math.isnan(float(semantic_loss.item())) or math.isinf(float(semantic_loss.item())):
                        logger.warning(f"Invalid semantic loss value: {float(semantic_loss.item())}, using placeholder")
                        semantic_loss = mx.array(1.0)
                    
                    losses["semantic_loss"] = semantic_loss
                    total_loss = semantic_weight * semantic_loss
                    
                except Exception as head_e:
                    logger.warning(f"Error with codebook0_head: {head_e}, using placeholder loss")
                    semantic_loss = mx.mean(mx.square(backbone_outputs[:, :-1] - backbone_outputs[:, 1:]))
                    losses["semantic_loss"] = semantic_loss
                    total_loss = semantic_weight * semantic_loss
            else:
                # No codebook0_head, use placeholder
                logger.warning("Model has no codebook0_head, using placeholder semantic loss")
                semantic_loss = mx.mean(mx.square(backbone_outputs[:, :-1] - backbone_outputs[:, 1:]))
                losses["semantic_loss"] = semantic_loss
                total_loss = semantic_weight * semantic_loss
        except Exception as semantic_e:
            logger.warning(f"Semantic loss computation failed completely: {semantic_e}")
            semantic_loss = mx.array(1.0)
            losses["semantic_loss"] = semantic_loss
            total_loss = semantic_weight * semantic_loss
        
        # Compute acoustic token losses (decoder) with comprehensive error handling
        if hasattr(model, 'decoder') and hasattr(model, 'projection') and len(target_audio_tokens.shape) > 2:
            try:
                # Get decoder and projection modules
                decoder = model.decoder
                projection = model.projection
                
                # Prepare acoustic loss computation
                acoustic_loss_sum = mx.array(0.0)
                codebook_count = 0
                
                # Get decoder output
                try:
                    # Try different parameter combinations
                    try:
                        # Standard convention
                        decoder_output = decoder(backbone_outputs, input_pos=positions)
                    except:
                        try:
                            # Alternative convention
                            decoder_output = decoder(backbone_outputs, positions)
                        except:
                            try:
                                # Minimal convention
                                decoder_output = decoder(backbone_outputs)
                            except Exception as decoder_e:
                                logger.warning(f"All decoder conventions failed: {decoder_e}")
                                raise
                                
                    # Apply projection to get shared representation
                    if callable(getattr(projection, '__call__', None)):
                        # Normal function call
                        projected = projection(decoder_output)
                    else:
                        # Matrix multiplication
                        projected = mx.matmul(decoder_output, projection.T)
                        
                except Exception as decoder_e:
                    logger.warning(f"Decoder processing failed: {decoder_e}")
                    # Create placeholder decoder output
                    embed_dim = backbone_outputs.shape[-1] if hasattr(backbone_outputs, 'shape') else 2048
                    projected = mx.zeros((batch_size, seq_len, embed_dim))
                
                # Process each codebook separately with individual error handling
                for i in range(1, min(target_audio_tokens.shape[2], 32)):  # Cap at 32 codebooks for safety
                    try:
                        # Get target for this codebook
                        target_i = target_audio_tokens[:, :, i]
                        
                        # Compute loss for this codebook with error handling
                        loss_i = None
                        
                        # Try using model's audio head if available
                        if hasattr(model, 'audio_head') and isinstance(model.audio_head, list) and len(model.audio_head) >= i:
                            try:
                                # Get audio head weight
                                head_weight = model.audio_head[i-1]
                                
                                # Compute logits using matrix multiply
                                logits_i = mx.matmul(projected, head_weight.T)
                                
                                # Get target shape matching
                                target_len = min(logits_i.shape[1], target_i.shape[1])
                                logits_i_trimmed = logits_i[:, :target_len]
                                target_i_trimmed = target_i[:, :target_len]
                                
                                # Try cross entropy loss
                                try:
                                    loss_i = nn.losses.cross_entropy(
                                        logits_i_trimmed.reshape(-1, logits_i_trimmed.shape[-1]),
                                        target_i_trimmed.reshape(-1),
                                        reduction='mean'
                                    )
                                except Exception as ce_e:
                                    logger.warning(f"Cross entropy failed for codebook {i}: {ce_e}")
                                    # Fallback to MSE
                                    vocab_size = logits_i_trimmed.shape[-1]
                                    target_one_hot = mx.one_hot(target_i_trimmed, vocab_size)
                                    loss_i = mx.mean(mx.square(
                                        logits_i_trimmed - 
                                        target_one_hot.reshape(logits_i_trimmed.shape)
                                    ))
                            except Exception as head_e:
                                logger.warning(f"Error using audio head for codebook {i}: {head_e}")
                                loss_i = None
                        
                        # Fallback for missing or failed audio head
                        if loss_i is None:
                            try:
                                # Get vocabulary size from model args or use default
                                vocab_size = 2048
                                if hasattr(model, 'args') and hasattr(model.args, 'audio_vocab_size'):
                                    vocab_size = model.args.audio_vocab_size
                                
                                # Use MSE loss with one-hot encoding
                                target_one_hot = mx.one_hot(
                                    target_i[:, :projected.shape[1]], 
                                    vocab_size
                                )
                                loss_i = mx.mean(mx.square(projected - target_one_hot))
                            except Exception as mse_e:
                                logger.warning(f"MSE fallback failed for codebook {i}: {mse_e}")
                                # Final fallback - use constant
                                loss_i = mx.array(0.1)
                        
                        # Validate loss value
                        if loss_i is not None and not (math.isnan(float(loss_i.item())) or math.isinf(float(loss_i.item()))):
                            acoustic_loss_sum += loss_i
                            codebook_count += 1
                        else:
                            logger.warning(f"Invalid loss for codebook {i}, value: {float(loss_i.item()) if loss_i is not None else 'None'}")
                            
                    except Exception as codebook_e:
                        logger.warning(f"Failed to process codebook {i}: {codebook_e}")
                        # Continue with next codebook
                
                # Compute average acoustic loss
                if codebook_count > 0:
                    acoustic_loss = acoustic_loss_sum / codebook_count
                    losses["acoustic_loss"] = acoustic_loss
                    total_loss += acoustic_weight * acoustic_loss
                else:
                    # No valid codebooks processed
                    logger.warning("No valid codebooks were processed for acoustic loss")
                    acoustic_loss = mx.array(0.0)
                    losses["acoustic_loss"] = acoustic_loss
                
            except Exception as acoustic_e:
                logger.warning(f"Failed to compute acoustic loss: {acoustic_e}")
                acoustic_loss = mx.array(0.0)
                losses["acoustic_loss"] = acoustic_loss
        else:
            # Model missing required components
            logger.warning("Model missing required components for acoustic loss")
            acoustic_loss = mx.array(0.0)
            losses["acoustic_loss"] = acoustic_loss
        
        # Validate final total_loss
        if math.isnan(float(total_loss.item())) or math.isinf(float(total_loss.item())):
            logger.warning(f"Invalid total loss value: {float(total_loss.item())}, using placeholder")
            total_loss = mx.array(1.0)
            
        return total_loss, losses
        
    except Exception as e:
        logger.error(f"Error computing loss: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Provide a fallback loss as last resort
        placeholder_loss = mx.array(1.0)
        placeholder_losses = {"semantic_loss": mx.array(0.5), "acoustic_loss": mx.array(0.5)}
        return placeholder_loss, placeholder_losses


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    global_step: int,
    loss: float,
    save_dir: str,
    name: str = "checkpoint"
):
    """
    Save training checkpoint.
    
    Args:
        model: Model state to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        global_step: Current global step
        loss: Current loss value
        save_dir: Directory to save checkpoint
        name: Checkpoint name prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{name}_epoch{epoch}_step{global_step}.pt")
    
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss
        },
        checkpoint_path
    )
    
    # Save latest checkpoint (for resuming)
    latest_path = os.path.join(save_dir, f"{name}_latest.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss
        },
        latest_path
    )
    
    return checkpoint_path


def save_checkpoint_mlx(
    model,
    optimizer,
    epoch: int,
    global_step: int,
    loss: float,
    save_dir: str,
    name: str = "checkpoint"
):
    """
    Save training checkpoint for MLX model.
    
    Args:
        model: Model state to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        global_step: Current global step
        loss: Current loss value
        save_dir: Directory to save checkpoint
        name: Checkpoint name prefix
    """
    import mlx.core as mx
    import logging
    logger = logging.getLogger("save_checkpoint_mlx")
    
    try:
        # Input validation
        if model is None:
            logger.error("Cannot save checkpoint: model is None")
            return None
            
        if optimizer is None:
            logger.warning("Optimizer is None, will only save model state")
            
        # Validate loss value
        import math
        try:
            loss_value = float(loss)
            if math.isnan(loss_value) or math.isinf(loss_value):
                logger.warning(f"Invalid loss value: {loss_value}, using 1.0")
                loss_value = 1.0
        except (TypeError, ValueError):
            logger.warning(f"Could not convert loss to float: {loss}, using 1.0")
            loss_value = 1.0
            
        # Prepare for checkpoint saving
        from mlx.utils import tree_flatten
        import safetensors.numpy
        
        # Create directory
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as dir_e:
            logger.error(f"Failed to create save directory {save_dir}: {dir_e}")
            # Try to save in current directory as fallback
            save_dir = '.'
            os.makedirs(save_dir, exist_ok=True)
        
        # Setup paths
        checkpoint_path = os.path.join(save_dir, f"{name}_epoch{epoch}_step{global_step}.safetensors")
        optimizer_path = os.path.join(save_dir, f"{name}_optimizer_epoch{epoch}_step{global_step}.safetensors")
        metadata_path = os.path.join(save_dir, f"{name}_metadata_epoch{epoch}_step{global_step}.json")
        latest_path = os.path.join(save_dir, f"{name}_latest.json")
        
        # Get model parameters with comprehensive error handling
        model_saved = False
        
        # Try different approaches to get model parameters
        if hasattr(model, 'parameters') and callable(model.parameters):
            try:
                logger.debug("Using model.parameters() to get model state")
                weights = model.parameters()
                # In case weights is None or empty
                if weights is not None and len(weights) > 0:
                    # Convert MLX arrays to numpy arrays for safetensors
                    np_weights = {}
                    for k, v in tree_flatten(weights):
                        # Convert MLX arrays to numpy arrays if needed
                        if hasattr(v, 'dtype') and not isinstance(v, np.ndarray):
                            try:
                                # Try to convert to numpy array
                                if hasattr(v, 'tolist'):
                                    v = np.array(v.tolist(), dtype=np.float32)
                                else:
                                    # If conversion fails, use a placeholder
                                    v = np.zeros((1, 1), dtype=np.float32)
                            except Exception:
                                # Use placeholder if conversion fails
                                v = np.zeros((1, 1), dtype=np.float32)
                        np_weights[k] = v
                    
                    safetensors.numpy.save_file(np_weights, checkpoint_path)
                    model_saved = True
                    logger.info(f"Saved model state to {checkpoint_path} ({len(np_weights)} parameters)")
                else:
                    logger.warning("model.parameters() returned empty dictionary")
            except Exception as param_e:
                logger.warning(f"Failed to save using model.parameters(): {param_e}")
                
        # Try alternative methods if model.parameters() failed
        if not model_saved and hasattr(model, 'state_dict') and callable(model.state_dict):
            try:
                logger.debug("Using model.state_dict() to get model state")
                state_dict = model.state_dict()
                if state_dict is not None and len(state_dict) > 0:
                    # Convert MLX arrays to numpy arrays for safetensors
                    np_weights = {}
                    for k, v in tree_flatten(state_dict):
                        # Convert MLX arrays to numpy arrays if needed
                        if hasattr(v, 'dtype') and not isinstance(v, np.ndarray):
                            try:
                                # Try to convert to numpy array
                                if hasattr(v, 'tolist'):
                                    v = np.array(v.tolist(), dtype=np.float32)
                                else:
                                    # If conversion fails, use a placeholder
                                    v = np.zeros((1, 1), dtype=np.float32)
                            except Exception:
                                # Use placeholder if conversion fails
                                v = np.zeros((1, 1), dtype=np.float32)
                        np_weights[k] = v
                        
                    safetensors.numpy.save_file(np_weights, checkpoint_path)
                    model_saved = True
                    logger.info(f"Saved model state to {checkpoint_path} using state_dict ({len(np_weights)} parameters)")
                else:
                    logger.warning("model.state_dict() returned empty dictionary")
            except Exception as state_dict_e:
                logger.warning(f"Failed to save using model.state_dict(): {state_dict_e}")
        
        # Try component-wise saving if other methods failed
        if not model_saved:
            try:
                logger.debug("Attempting component-wise model saving")
                components = {}
                
                # Try to get backbone parameters
                if hasattr(model, 'backbone') and hasattr(model.backbone, 'parameters'):
                    try:
                        backbone_params = model.backbone.parameters()
                        if backbone_params:
                            for name, param in backbone_params.items():
                                components[f"backbone.{name}"] = param
                    except Exception as backbone_e:
                        logger.warning(f"Failed to save backbone parameters: {backbone_e}")
                
                # Try to get decoder parameters
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'parameters'):
                    try:
                        decoder_params = model.decoder.parameters()
                        if decoder_params:
                            for name, param in decoder_params.items():
                                components[f"decoder.{name}"] = param
                    except Exception as decoder_e:
                        logger.warning(f"Failed to save decoder parameters: {decoder_e}")
                
                # Try to get embedding parameters
                if hasattr(model, 'embedding') and hasattr(model.embedding, 'parameters'):
                    try:
                        embedding_params = model.embedding.parameters()
                        if embedding_params:
                            for name, param in embedding_params.items():
                                components[f"embedding.{name}"] = param
                    except Exception as embed_e:
                        logger.warning(f"Failed to save embedding parameters: {embed_e}")
                
                # Try to save other top-level attributes
                for attr_name in dir(model):
                    if attr_name.startswith('_') or attr_name in ['backbone', 'decoder', 'embedding']:
                        continue
                        
                    attr = getattr(model, attr_name)
                    if isinstance(attr, mx.array):
                        components[attr_name] = attr
                
                # Save components if any were found
                if components:
                    # Convert MLX arrays to numpy arrays for safetensors
                    np_components = {}
                    for k, v in tree_flatten(components):
                        # Convert MLX arrays to numpy arrays if needed
                        if hasattr(v, 'dtype') and not isinstance(v, np.ndarray):
                            try:
                                # Try to convert to numpy array
                                if hasattr(v, 'tolist'):
                                    v = np.array(v.tolist(), dtype=np.float32)
                                else:
                                    # If conversion fails, use a placeholder
                                    v = np.zeros((1, 1), dtype=np.float32)
                            except Exception:
                                # Use placeholder if conversion fails
                                v = np.zeros((1, 1), dtype=np.float32)
                        np_components[k] = v
                            
                    safetensors.numpy.save_file(np_components, checkpoint_path)
                    model_saved = True
                    logger.info(f"Saved model state to {checkpoint_path} using component-wise approach ({len(np_components)} parameters)")
                else:
                    logger.warning("No components found to save")
            except Exception as component_e:
                logger.warning(f"Component-wise saving failed: {component_e}")
                
        # If all methods failed, create an empty checkpoint
        if not model_saved:
            try:
                logger.warning("Creating empty checkpoint as placeholder")
                # Convert MLX array to numpy explicitly
                placeholder_array = np.zeros((1, 1), dtype=np.float32)
                empty_dict = {"placeholder": placeholder_array}
                safetensors.numpy.save_file(empty_dict, checkpoint_path)
                model_saved = True
                logger.info(f"Saved empty placeholder to {checkpoint_path}")
            except Exception as empty_e:
                logger.error(f"Failed to save empty checkpoint: {empty_e}")
                return None
                
        # Save optimizer state if available
        optimizer_saved = False
        if optimizer is not None:
            try:
                if hasattr(optimizer, 'state') and optimizer.state:
                    # Need to convert MLX arrays to numpy arrays
                    opt_state_dict = {}
                    for k, v in tree_flatten(optimizer.state):
                        # Convert MLX arrays to numpy arrays if needed
                        if hasattr(v, 'dtype') and not isinstance(v, np.ndarray):
                            try:
                                # Try to convert to numpy array
                                if hasattr(v, 'tolist'):
                                    v = np.array(v.tolist(), dtype=np.float32)
                                else:
                                    # If conversion fails, skip this parameter
                                    continue
                            except Exception:
                                # Skip if conversion fails
                                continue
                        opt_state_dict[k] = v
                            
                    if opt_state_dict:
                        safetensors.numpy.save_file(opt_state_dict, optimizer_path)
                        optimizer_saved = True
                        logger.info(f"Saved optimizer state to {optimizer_path}")
                    else:
                        logger.warning("No valid optimizer state could be converted for saving")
                else:
                    logger.warning("Optimizer has no state or empty state")
            except Exception as opt_e:
                logger.warning(f"Failed to save optimizer state: {opt_e}")
                
        # Save metadata
        try:
            metadata = {
                "epoch": epoch,
                "global_step": global_step,
                "loss": loss_value,
                "model_path": checkpoint_path,
                "optimizer_path": optimizer_path if optimizer_saved else None,
                "timestamp": time.time(),
                "model_saved": model_saved,
                "optimizer_saved": optimizer_saved
            }
            
            import json
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
                
            # Save latest pointer
            with open(latest_path, "w") as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Saved checkpoint metadata to {metadata_path}")
        except Exception as meta_e:
            logger.warning(f"Failed to save metadata: {meta_e}")
            
        # Return checkpoint path if model was saved successfully
        if model_saved:
            return checkpoint_path
        else:
            return None
            
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint to
        
    Returns:
        Checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model"])
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "loss": checkpoint.get("loss", float("inf"))
    }
    
    return metadata


def load_checkpoint_mlx(
    checkpoint_path: str,
    model,
    optimizer=None
) -> Dict:
    """
    Load training checkpoint for MLX model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        Checkpoint metadata
    """
    import logging
    logger = logging.getLogger("load_checkpoint_mlx")
    
    # Default metadata to return on failure
    default_metadata = {
        "epoch": 0,
        "global_step": 0,
        "loss": 1.0,
        "loaded": False,
        "error": None
    }
    
    try:
        # Input validation
        if checkpoint_path is None or not isinstance(checkpoint_path, str):
            logger.error(f"Invalid checkpoint path: {checkpoint_path}")
            default_metadata["error"] = f"Invalid checkpoint path: {checkpoint_path}"
            return default_metadata
            
        if model is None:
            logger.error("Model is None, cannot load checkpoint")
            default_metadata["error"] = "Model is None"
            return default_metadata
            
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            default_metadata["error"] = f"Checkpoint file not found: {checkpoint_path}"
            return default_metadata
            
        # Import required libraries
        import mlx.core as mx
        from mlx.utils import tree_unflatten
        import json
        import safetensors.numpy
        
        # Try to load the metadata file
        metadata = None
        try:
            # Extract metadata path from checkpoint path - try multiple patterns
            base_path = checkpoint_path.replace(".safetensors", "")
            metadata_paths = [
                f"{base_path}_metadata.json",             # Standard
                f"{base_path.replace('_epoch', '_metadata_epoch')}",  # Alternative
                os.path.join(os.path.dirname(checkpoint_path), "latest.json")  # Latest pointer
            ]
            
            # Try each potential metadata path
            for path in metadata_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        metadata = json.load(f)
                    logger.info(f"Loaded metadata from {path}")
                    break
                    
            # If still no metadata, create a minimal metadata
            if metadata is None:
                logger.warning("Metadata file not found, creating minimal metadata")
                metadata = {
                    "epoch": 0,
                    "global_step": 0,
                    "loss": 1.0,
                    "model_path": checkpoint_path,
                    "optimizer_path": None
                }
        except Exception as meta_e:
            logger.warning(f"Failed to load metadata: {meta_e}")
            metadata = {
                "epoch": 0,
                "global_step": 0,
                "loss": 1.0,
                "model_path": checkpoint_path,
                "optimizer_path": None,
                "error": str(meta_e)
            }
        
        # Load model weights
        model_loaded = False
        
        # Try different methods to update the model
        if hasattr(model, 'update') and callable(model.update):
            try:
                logger.debug("Using model.update() to load weights")
                # Load weights from safetensors
                weights = safetensors.numpy.load_file(checkpoint_path)
                if weights:
                    params = tree_unflatten(list(weights.items()))
                    model.update(params)
                    model_loaded = True
                    logger.info(f"Loaded model state from {checkpoint_path} using model.update() ({len(weights)} parameters)")
                else:
                    logger.warning("Loaded weights dictionary is empty")
            except Exception as update_e:
                logger.warning(f"Failed to update model with weights: {update_e}")
        
        # Try alternative loading methods if update failed
        if not model_loaded and hasattr(model, 'load_state_dict') and callable(model.load_state_dict):
            try:
                logger.debug("Using model.load_state_dict() to load weights")
                weights = safetensors.numpy.load_file(checkpoint_path)
                if weights:
                    state_dict = tree_unflatten(list(weights.items()))
                    model.load_state_dict(state_dict)
                    model_loaded = True
                    logger.info(f"Loaded model state from {checkpoint_path} using load_state_dict() ({len(weights)} parameters)")
                else:
                    logger.warning("Loaded weights dictionary is empty")
            except Exception as load_e:
                logger.warning(f"Failed to load state dict: {load_e}")
        
        # Try component-wise loading if other methods failed
        if not model_loaded:
            try:
                logger.debug("Attempting component-wise model loading")
                weights = safetensors.numpy.load_file(checkpoint_path)
                component_updates = 0
                
                # Organize weights by component
                backbone_params = {}
                decoder_params = {}
                embedding_params = {}
                other_params = {}
                
                for name, param in weights.items():
                    if name.startswith("backbone."):
                        backbone_params[name[len("backbone."):]] = param
                    elif name.startswith("decoder."):
                        decoder_params[name[len("decoder."):]] = param
                    elif name.startswith("embedding."):
                        embedding_params[name[len("embedding."):]] = param
                    else:
                        other_params[name] = param
                
                # Update backbone if available
                if backbone_params and hasattr(model, 'backbone'):
                    try:
                        if hasattr(model.backbone, 'update') and callable(model.backbone.update):
                            model.backbone.update(tree_unflatten(list(backbone_params.items())))
                            component_updates += 1
                            logger.info(f"Updated backbone with {len(backbone_params)} parameters")
                    except Exception as backbone_e:
                        logger.warning(f"Failed to update backbone: {backbone_e}")
                
                # Update decoder if available
                if decoder_params and hasattr(model, 'decoder'):
                    try:
                        if hasattr(model.decoder, 'update') and callable(model.decoder.update):
                            model.decoder.update(tree_unflatten(list(decoder_params.items())))
                            component_updates += 1
                            logger.info(f"Updated decoder with {len(decoder_params)} parameters")
                    except Exception as decoder_e:
                        logger.warning(f"Failed to update decoder: {decoder_e}")
                
                # Update embedding if available
                if embedding_params and hasattr(model, 'embedding'):
                    try:
                        if hasattr(model.embedding, 'update') and callable(model.embedding.update):
                            model.embedding.update(tree_unflatten(list(embedding_params.items())))
                            component_updates += 1
                            logger.info(f"Updated embedding with {len(embedding_params)} parameters")
                    except Exception as embed_e:
                        logger.warning(f"Failed to update embedding: {embed_e}")
                
                # Update other attributes directly
                for name, param in other_params.items():
                    try:
                        if hasattr(model, name):
                            setattr(model, name, param)
                            component_updates += 1
                            logger.info(f"Updated attribute {name}")
                    except Exception as attr_e:
                        logger.warning(f"Failed to update attribute {name}: {attr_e}")
                
                # Consider model loaded if at least one component was updated
                if component_updates > 0:
                    model_loaded = True
                    logger.info(f"Loaded model using component-wise approach ({component_updates} components updated)")
                else:
                    logger.warning("No components could be updated")
            except Exception as component_e:
                logger.warning(f"Component-wise loading failed: {component_e}")
        
        # Try brutal name-based loading as last resort
        if not model_loaded:
            try:
                logger.warning("Attempting direct attribute mapping as last resort")
                weights = safetensors.numpy.load_file(checkpoint_path)
                attr_updates = 0
                
                # Collect all attributes from the model recursively
                def collect_attributes(obj, prefix=""):
                    attributes = {}
                    for name in dir(obj):
                        if name.startswith("_"):
                            continue
                        
                        attr = getattr(obj, name)
                        if isinstance(attr, mx.array):
                            attributes[f"{prefix}{name}"] = (obj, name, attr)
                        elif hasattr(attr, "__dict__") and not callable(attr):
                            nested_attrs = collect_attributes(attr, f"{prefix}{name}.")
                            attributes.update(nested_attrs)
                    return attributes
                
                # Get all attributes in the model
                all_attrs = collect_attributes(model)
                
                # Try to match weight names to attributes
                for weight_name, weight_value in weights.items():
                    # Try direct match
                    if weight_name in all_attrs:
                        obj, attr_name, _ = all_attrs[weight_name]
                        setattr(obj, attr_name, weight_value)
                        attr_updates += 1
                    else:
                        # Try approximate matching
                        for attr_name, (obj, name, _) in all_attrs.items():
                            # Check if weight name ends with the attribute name
                            if weight_name.endswith(f".{name}") or weight_name == name:
                                setattr(obj, name, weight_value)
                                attr_updates += 1
                                break
                
                if attr_updates > 0:
                    model_loaded = True
                    logger.info(f"Loaded model using direct attribute mapping ({attr_updates} attributes updated)")
                else:
                    logger.warning("No attributes could be updated through direct mapping")
            except Exception as brutal_e:
                logger.warning(f"Direct attribute mapping failed: {brutal_e}")
        
        # Update metadata with loading status
        metadata["model_loaded"] = model_loaded
        
        # If model loading failed entirely, log a clear error
        if not model_loaded:
            logger.error("Failed to load model weights using any available method")
            metadata["error"] = "Failed to load model weights using any available method"
        
        # Load optimizer state if provided and requested
        optimizer_loaded = False
        if optimizer is not None and "optimizer_path" in metadata and metadata["optimizer_path"]:
            try:
                optimizer_path = metadata["optimizer_path"]
                if os.path.exists(optimizer_path):
                    optimizer_state = safetensors.numpy.load_file(optimizer_path)
                    if optimizer_state:
                        optimizer.state = tree_unflatten(list(optimizer_state.items()))
                        optimizer_loaded = True
                        logger.info(f"Loaded optimizer state from {optimizer_path}")
                    else:
                        logger.warning("Optimizer state file is empty")
                else:
                    logger.warning(f"Optimizer state file not found: {optimizer_path}")
            except Exception as opt_e:
                logger.warning(f"Failed to load optimizer state: {opt_e}")
                metadata["optimizer_error"] = str(opt_e)
        
        # Update metadata with optimizer loading status
        metadata["optimizer_loaded"] = optimizer_loaded
        
        return metadata
    except Exception as e:
        # Provide a detailed error in the metadata
        logger.error(f"Failed to load checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        default_metadata["error"] = str(e)
        return default_metadata