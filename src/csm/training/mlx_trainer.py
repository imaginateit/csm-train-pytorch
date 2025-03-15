"""MLX trainer for CSM models."""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Callable
from tqdm import tqdm
import numpy as np
import logging

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .utils import (
    setup_logger,
    compute_loss_mlx,
    save_checkpoint_mlx,
    load_checkpoint_mlx
)


class CSMMLXTrainer:
    """MLX trainer for CSM models."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        log_file: Optional[str] = None,
        learning_rate: float = 1e-5,
        backbone_lr_multiplier: float = 0.1,
        decoder_lr_multiplier: float = 1.0,
        embedding_lr_multiplier: float = 0.5,
        semantic_weight: float = 100.0,
        acoustic_weight: float = 1.0,
        weight_decay: float = 0.01
    ):
        """
        Initialize the MLX trainer.
        
        Args:
            model_path: Path to model checkpoint (safetensors format)
            output_dir: Directory to save outputs
            log_file: Path to log file (optional)
            learning_rate: Base learning rate
            backbone_lr_multiplier: Multiplier for backbone learning rate
            decoder_lr_multiplier: Multiplier for decoder learning rate
            embedding_lr_multiplier: Multiplier for embedding learning rate
            semantic_weight: Weight for semantic token loss
            acoustic_weight: Weight for acoustic token loss
            weight_decay: Weight decay for optimizer
        """
        if not HAS_MLX:
            raise ImportError(
                "MLX is required for the MLX trainer. "
                "Install it with 'pip install mlx'."
            )
            
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger(
            "csm_mlx_trainer",
            log_file or str(self.output_dir / "training.log")
        )
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.backbone_lr_multiplier = backbone_lr_multiplier
        self.decoder_lr_multiplier = decoder_lr_multiplier
        self.embedding_lr_multiplier = embedding_lr_multiplier
        self.semantic_weight = semantic_weight
        self.acoustic_weight = acoustic_weight
        self.weight_decay = weight_decay
        
        # Load the model
        self.logger.info(f"Loading MLX model from {model_path}")
        self.model = None
        self.optimizer = None
        self._load_model()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
    
    def _load_model(self):
        """Load the CSM model in MLX format."""
        # This would be implemented based on the MLX version of the model
        # Placeholder implementation - actual code would need to:
        # 1. Initialize the MLX model architecture
        # 2. Load weights from safetensors file
        # 3. Convert PyTorch weights to MLX format if needed
        
        from csm.mlx.components.model_wrapper import MLXModelWrapper
        from csm.mlx.mlx_wrapper import PyTorchToMLXConverter
        
        # Create MLX model (placeholder)
        self.logger.info("Creating MLX model")
        # self.model = MLXModel()
        
        # Load weights
        self.logger.info(f"Loading weights from {self.model_path}")
        # If weights are in PyTorch format, convert to MLX
        if self.model_path.endswith(".pt"):
            # Conversion logic
            pass
        else:
            # Direct loading of MLX weights
            pass
    
    def prepare_optimizer(
        self,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        freeze_embeddings: bool = False
    ):
        """
        Prepare optimizer for the MLX model.
        
        Args:
            freeze_backbone: Whether to freeze backbone parameters
            freeze_decoder: Whether to freeze decoder parameters
            freeze_embeddings: Whether to freeze embedding parameters
        """
        # Get all trainable parameters
        # Note: MLX doesn't have requires_grad, so we need a different approach
        # This is a placeholder implementation
        
        # Create optimizer
        self.optimizer = optim.Adam(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Count parameters
        total_params = sum(np.prod(p.shape) for p in self.model.parameters().values())
        self.logger.info(f"Training with {total_params:,} parameters")
    
    def train_step(self, batch):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss value
        """
        # Define the loss function
        def loss_fn(model_params):
            # Set model parameters
            self.model.update(model_params)
            
            # Forward pass
            loss, _ = compute_loss_mlx(
                self.model,
                batch["input_tokens"],
                batch["input_masks"],
                batch["target_audio_tokens"],
                self.semantic_weight,
                self.acoustic_weight
            )
            return loss
        
        # Use MLX's value_and_grad to get loss and gradients
        loss, grads = nn.value_and_grad(loss_fn)(self.model.parameters())
        
        # Apply gradients
        self.optimizer.update(self.model, grads)
        
        # Explicitly evaluate to ensure values are computed
        # (MLX is lazy by default)
        mx.eval(self.model.parameters(), self.optimizer.state)
        
        return loss
    
    def train(
        self,
        train_dataset,
        val_dataset=None,
        batch_size: int = 2,
        epochs: int = 5,
        val_every: int = 100,
        save_every: int = 500,
        max_grad_norm: float = 1.0,
        resume_from: Optional[str] = None
    ):
        """
        Train the model using MLX.
        
        Args:
            train_dataset: Training dataset (MLX format)
            val_dataset: Validation dataset (MLX format)
            batch_size: Batch size
            epochs: Number of epochs to train
            val_every: Validate every N steps
            save_every: Save checkpoint every N steps
            max_grad_norm: Maximum gradient norm for clipping
            resume_from: Path to checkpoint to resume from (optional)
        """
        # Make sure optimizer is created
        if self.optimizer is None:
            self.prepare_optimizer()
        
        # Resume from checkpoint if requested
        if resume_from:
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            metadata = load_checkpoint_mlx(
                resume_from,
                self.model,
                self.optimizer
            )
            self.epoch = metadata["epoch"]
            self.global_step = metadata["global_step"]
            self.best_loss = metadata["loss"]
        
        # Training loop
        self.logger.info("Starting MLX training")
        
        for epoch in range(self.epoch, self.epoch + epochs):
            epoch_start = time.time()
            train_losses = []
            
            # Progress bar for batches
            n_batches = len(train_dataset) // batch_size
            pbar = tqdm(total=n_batches, desc=f"Epoch {epoch+1}/{self.epoch+epochs}")
            
            # Iterate through training batches
            for batch_idx in range(n_batches):
                # Get batch
                batch = train_dataset.get_batch(batch_idx, batch_size)
                
                # Perform training step
                loss = self.train_step(batch)
                
                # Log loss
                train_losses.append(float(loss))
                
                # Update global step and progress bar
                self.global_step += 1
                pbar.set_postfix({
                    "loss": float(loss),
                    "step": self.global_step
                })
                pbar.update(1)
                
                # Validate if needed
                if val_dataset and self.global_step % val_every == 0:
                    val_loss = self._validate(val_dataset, batch_size)
                    self.logger.info(
                        f"Epoch {epoch+1}, Step {self.global_step}, "
                        f"Val Loss: {val_loss:.6f}"
                    )
                    
                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        save_checkpoint_mlx(
                            self.model,
                            self.optimizer,
                            epoch + 1,
                            self.global_step,
                            val_loss,
                            str(self.output_dir),
                            "best"
                        )
                
                # Save checkpoint if needed
                if self.global_step % save_every == 0:
                    save_checkpoint_mlx(
                        self.model,
                        self.optimizer,
                        epoch + 1,
                        self.global_step,
                        float(loss),
                        str(self.output_dir)
                    )
            
            # End of epoch
            pbar.close()
            epoch_duration = time.time() - epoch_start
            
            # Compute average loss for epoch
            avg_loss = float(np.mean(train_losses))
            self.logger.info(
                f"Epoch {epoch+1} completed in {epoch_duration:.2f}s, "
                f"Avg Loss: {avg_loss:.6f}"
            )
            
            # Save epoch checkpoint
            save_checkpoint_mlx(
                self.model,
                self.optimizer,
                epoch + 1,
                self.global_step,
                avg_loss,
                str(self.output_dir),
                f"epoch_{epoch+1}"
            )
            
            # Update epoch counter
            self.epoch = epoch + 1
        
        # Save final model
        self.logger.info("Training completed")
        save_checkpoint_mlx(
            self.model,
            self.optimizer,
            self.epoch,
            self.global_step,
            avg_loss,
            str(self.output_dir),
            "final"
        )
        
        return self.best_loss
    
    def _validate(self, val_dataset, batch_size: int) -> float:
        """
        Validate the model.
        
        Args:
            val_dataset: Validation dataset
            batch_size: Batch size
            
        Returns:
            Validation loss
        """
        total_loss = 0.0
        num_batches = min(len(val_dataset) // batch_size, 10)  # Limit validation to 10 batches
        
        for batch_idx in range(num_batches):
            # Get batch
            batch = val_dataset.get_batch(batch_idx, batch_size)
            
            # Compute loss
            loss, _ = compute_loss_mlx(
                self.model,
                batch["input_tokens"],
                batch["input_masks"],
                batch["target_audio_tokens"],
                self.semantic_weight,
                self.acoustic_weight
            )
            
            # Ensure loss is evaluated
            loss_value = float(mx.eval(loss))
            total_loss += loss_value
        
        return total_loss / max(1, num_batches)
    
    def generate_sample(
        self,
        text: str,
        speaker_id: int = 0,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a sample from the current MLX model.
        
        Args:
            text: Text to generate
            speaker_id: Speaker ID to use
            output_path: Path to save output (optional)
            
        Returns:
            Path to generated audio file
        """
        # Not implemented yet
        self.logger.warning("MLX sample generation not implemented yet")
        
        if output_path is None:
            output_path = str(self.output_dir / f"sample_step_{self.global_step}.wav")
        
        # Placeholder for MLX generation
        return output_path