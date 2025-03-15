"""Training utilities for CSM."""

import os
import logging
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
    
    batch_size, seq_len, _ = input_tokens.shape
    
    # Create positions
    positions = mx.arange(0, seq_len)
    positions = mx.expand_dims(positions, 0)
    positions = mx.repeat(positions, batch_size, 0)
    
    # Forward pass logic for MLX model
    # This would need to be implemented based on the MLX model architecture
    
    # Placeholder loss computation
    # Actual implementation would compute cross-entropy losses for semantic and acoustic tokens
    
    total_loss = mx.zeros(1)
    losses = {"semantic_loss": mx.zeros(1), "acoustic_loss": mx.zeros(1)}
    
    return total_loss, losses


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
    from mlx.utils import tree_flatten
    import safetensors.numpy
    
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{name}_epoch{epoch}_step{global_step}.safetensors")
    
    # Flatten model weights
    weights = model.parameters()
    flattened_weights = dict(tree_flatten(weights))
    
    # Save model weights
    safetensors.numpy.save_file(flattened_weights, checkpoint_path)
    
    # Save optimizer state separately
    optimizer_path = os.path.join(save_dir, f"{name}_optimizer_epoch{epoch}_step{global_step}.safetensors")
    flattened_opt_state = dict(tree_flatten(optimizer.state))
    safetensors.numpy.save_file(flattened_opt_state, optimizer_path)
    
    # Save metadata
    metadata_path = os.path.join(save_dir, f"{name}_metadata_epoch{epoch}_step{global_step}.json")
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": float(loss),
        "model_path": checkpoint_path,
        "optimizer_path": optimizer_path
    }
    
    with open(metadata_path, "w") as f:
        import json
        json.dump(metadata, f, indent=4)
    
    # Save latest checkpoint info for resuming
    latest_path = os.path.join(save_dir, f"{name}_latest.json")
    with open(latest_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    return checkpoint_path


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
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    import json
    import safetensors.numpy
    
    # Extract metadata path from checkpoint path
    base_path = checkpoint_path.replace(".safetensors", "")
    metadata_path = f"{base_path}_metadata.json"
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Load model weights
    weights = safetensors.numpy.load_file(checkpoint_path)
    params = tree_unflatten(list(weights.items()))
    model.update(params)
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_path" in metadata:
        optimizer_state = safetensors.numpy.load_file(metadata["optimizer_path"])
        optimizer.state = tree_unflatten(list(optimizer_state.items()))
    
    return metadata