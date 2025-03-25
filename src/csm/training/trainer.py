"""Trainer implementations for CSM."""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

from csm.models.model import Model, ModelArgs
from csm.generator import Generator, load_csm_1b
from .utils import (
    setup_logger,
    compute_loss,
    save_checkpoint,
    load_checkpoint
)
from csm.data import CSMDataset, create_dataloader


class CSMTrainer:
    """PyTorch trainer for CSM models."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        device: str = "cuda",
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
        Initialize the trainer.
        
        Args:
            model_path: Path to model checkpoint
            output_dir: Directory to save outputs
            device: Device to use for training
            log_file: Path to log file (optional)
            learning_rate: Base learning rate
            backbone_lr_multiplier: Multiplier for backbone learning rate
            decoder_lr_multiplier: Multiplier for decoder learning rate
            embedding_lr_multiplier: Multiplier for embedding learning rate
            semantic_weight: Weight for semantic token loss
            acoustic_weight: Weight for acoustic token loss
            weight_decay: Weight decay for optimizer
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Set up logging
        self.logger = setup_logger(
            "csm_trainer",
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
        self.logger.info(f"Loading model from {model_path}")
        self.model = None
        self.optimizer = None
        self._load_model()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
    
    def _load_model(self):
        """Load the CSM model and prepare for training."""
        # Skip loading for empty path (used in tests)
        if not self.model_path:
            self.logger.warning("Empty model path provided. Model will need to be set manually.")
            return
            
        # Load model directly if standard format
        if self.model_path.endswith(".pt"):
            # Create model
            model_args = ModelArgs(
                backbone_flavor="llama-1B",
                decoder_flavor="llama-100M",
                text_vocab_size=128256,
                audio_vocab_size=2051,
                audio_num_codebooks=32,
            )
            self.model = Model(model_args).to(device=self.device)
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Initialize caches for training
            self.model.setup_caches(4)  # Assume batch size 4 for now
        else:
            # Try to load from generator
            generator = load_csm_1b(self.model_path, self.device)
            self.model = generator._model
            
            # Ensure caches are setup for training
            self.model.setup_caches(4)  # Assume batch size 4 for now
    
    def prepare_optimizer(
        self,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        freeze_embeddings: bool = False
    ):
        """
        Prepare optimizer with parameter-specific learning rates.
        
        Args:
            freeze_backbone: Whether to freeze backbone parameters
            freeze_decoder: Whether to freeze decoder parameters
            freeze_embeddings: Whether to freeze embedding parameters
        """
        # Group parameters by component
        backbone_params = []
        decoder_params = []
        embedding_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if freeze_backbone and "backbone" in name:
                param.requires_grad = False
            elif freeze_decoder and "decoder" in name:
                param.requires_grad = False
            elif freeze_embeddings and "embeddings" in name:
                param.requires_grad = False
            
            if param.requires_grad:
                if "backbone" in name:
                    backbone_params.append(param)
                elif "decoder" in name:
                    decoder_params.append(param)
                elif "embeddings" in name:
                    embedding_params.append(param)
                else:
                    other_params.append(param)
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Training with {total_params:,} trainable parameters")
        
        # Create parameter groups with different learning rates
        param_groups = [
            {"params": backbone_params, "lr": self.learning_rate * self.backbone_lr_multiplier},
            {"params": decoder_params, "lr": self.learning_rate * self.decoder_lr_multiplier},
            {"params": embedding_params, "lr": self.learning_rate * self.embedding_lr_multiplier},
            {"params": other_params, "lr": self.learning_rate}
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
    
    def train(
        self,
        train_dataset: CSMDataset,
        val_dataset: Optional[CSMDataset] = None,
        batch_size: int = 2,
        accumulation_steps: int = 4,
        epochs: int = 5,
        val_every: int = 100,
        save_every: int = 500,
        max_grad_norm: float = 1.0,
        resume_from: Optional[str] = None
    ):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            batch_size: Batch size
            accumulation_steps: Number of batches to accumulate before updating weights
            epochs: Number of epochs to train
            val_every: Validate every N steps
            save_every: Save checkpoint every N steps
            max_grad_norm: Maximum gradient norm for clipping
            resume_from: Path to checkpoint to resume from (optional)
        """
        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        if val_dataset:
            val_loader = create_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
        
        # Make sure optimizer is created
        if self.optimizer is None:
            self.prepare_optimizer()
        
        # Resume from checkpoint if requested
        if resume_from:
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            metadata = load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.device
            )
            self.epoch = metadata["epoch"]
            self.global_step = metadata["global_step"]
            self.best_loss = metadata["loss"]
        
        # Training loop
        self.logger.info("Starting training")
        self.model.train()
        
        for epoch in range(self.epoch, self.epoch + epochs):
            epoch_start = time.time()
            train_losses = []
            
            # Progress bar for batches
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{self.epoch+epochs}")
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_tokens = batch["input_tokens"].to(self.device)
                input_masks = batch["input_masks"].to(self.device)
                target_audio_tokens = batch["target_audio_tokens"].to(self.device)
                
                # Forward pass and loss computation
                loss, loss_details = compute_loss(
                    self.model,
                    input_tokens,
                    input_masks,
                    target_audio_tokens,
                    self.semantic_weight,
                    self.acoustic_weight
                )
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                # Log loss
                train_losses.append(loss.item() * accumulation_steps)
                
                # Update weights if enough batches accumulated
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )
                    
                    # Update weights
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update global step and progress bar
                    self.global_step += 1
                    pbar.set_postfix({
                        "loss": loss.item() * accumulation_steps,
                        "step": self.global_step
                    })
                    pbar.update(1)
                    
                    # Validate if needed
                    if val_dataset and self.global_step % val_every == 0:
                        val_loss = self._validate(val_loader)
                        self.logger.info(
                            f"Epoch {epoch+1}, Step {self.global_step}, "
                            f"Val Loss: {val_loss:.6f}"
                        )
                        
                        # Save best model
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            save_checkpoint(
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
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            epoch + 1,
                            self.global_step,
                            np.mean(train_losses[-accumulation_steps:]),
                            str(self.output_dir)
                        )
            
            # End of epoch
            pbar.close()
            epoch_duration = time.time() - epoch_start
            
            # Compute average loss for epoch
            avg_loss = np.mean(train_losses)
            self.logger.info(
                f"Epoch {epoch+1} completed in {epoch_duration:.2f}s, "
                f"Avg Loss: {avg_loss:.6f}"
            )
            
            # Save epoch checkpoint
            save_checkpoint(
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
        save_checkpoint(
            self.model,
            self.optimizer,
            self.epoch,
            self.global_step,
            avg_loss,
            str(self.output_dir),
            "final"
        )
        
        return self.best_loss
    
    def _validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_tokens = batch["input_tokens"].to(self.device)
                input_masks = batch["input_masks"].to(self.device)
                target_audio_tokens = batch["target_audio_tokens"].to(self.device)
                
                # Forward pass and loss computation
                loss, _ = compute_loss(
                    self.model,
                    input_tokens,
                    input_masks,
                    target_audio_tokens,
                    self.semantic_weight,
                    self.acoustic_weight
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / max(1, num_batches)
    
    def generate_sample(
        self,
        text: str,
        speaker_id: int = 0,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a sample from the current model.
        
        Args:
            text: Text to generate
            speaker_id: Speaker ID to use
            output_path: Path to save output (optional)
            
        Returns:
            Path to generated audio file
        """
        import torchaudio
        
        if output_path is None:
            output_path = str(self.output_dir / f"sample_step_{self.global_step}.wav")
        
        # Create generator for the model
        generator = Generator(self.model)
        
        # Generate audio
        self.model.eval()
        with torch.no_grad():
            audio = generator.generate(
                text=text,
                speaker=speaker_id,
                context=[]
            )
        self.model.train()
        
        # Save audio
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator._audio_tokenizer.sample_rate)
        
        return output_path
