"""
Multi-speaker LoRA fine-tuning implementation for CSM models.

This module extends the LoRA trainer with support for fine-tuning
multiple speakers in a single training run, sharing some layers
while having speaker-specific LoRA adapters for others.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import tempfile

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .lora_trainer import CSMLoRATrainer
from .utils import setup_logger, compute_loss_mlx, save_checkpoint_mlx, load_checkpoint_mlx


class MultiSpeakerLoRATrainer:
    """
    Trainer for fine-tuning multiple speakers with LoRA in a single training run.
    
    This trainer manages multiple LoRA adapters that share some layers while having
    speaker-specific layers for others, enabling efficient multi-speaker fine-tuning.
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        speaker_ids: List[int],
        log_file: Optional[str] = None,
        learning_rate: float = 1e-4,
        semantic_weight: float = 100.0,
        acoustic_weight: float = 1.0,
        weight_decay: float = 0.01,
        # LoRA specific parameters
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        # Sharing configuration
        share_backbone: bool = True,
        share_decoder: bool = False,
        # Layer targeting
        target_modules: Optional[List[str]] = None,
        target_backbone_layers: Optional[List[int]] = None,
        target_decoder_layers: Optional[List[int]] = None,
        # Other parameters
        lora_use_bias: bool = False
    ):
        """
        Initialize the multi-speaker LoRA trainer.
        
        Args:
            model_path: Path to model checkpoint (safetensors format)
            output_dir: Directory to save outputs
            speaker_ids: List of speaker IDs to fine-tune
            log_file: Path to log file (optional)
            learning_rate: Base learning rate
            semantic_weight: Weight for semantic token loss
            acoustic_weight: Weight for acoustic token loss
            weight_decay: Weight decay for optimizer
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability for LoRA layers
            share_backbone: Whether to share backbone LoRA weights across speakers
            share_decoder: Whether to share decoder LoRA weights across speakers
            target_modules: List of module types to apply LoRA to (default: ["q_proj", "v_proj"])
            target_backbone_layers: List of backbone layers to apply LoRA to (default: all)
            target_decoder_layers: List of decoder layers to apply LoRA to (default: all)
            lora_use_bias: Whether to use bias in LoRA layers
        """
        if not HAS_MLX:
            raise ImportError(
                "MLX is required for the multi-speaker LoRA trainer. "
                "Install it with 'pip install mlx'."
            )
        
        # Basic parameters
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.speaker_ids = speaker_ids
        
        # Set up logging
        self.logger = setup_logger(
            "multi_speaker_lora_trainer",
            log_file or str(self.output_dir / "multi_speaker_training.log")
        )
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.semantic_weight = semantic_weight
        self.acoustic_weight = acoustic_weight
        self.weight_decay = weight_decay
        
        # LoRA hyperparameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.target_backbone_layers = target_backbone_layers
        self.target_decoder_layers = target_decoder_layers
        self.lora_use_bias = lora_use_bias
        
        # Sharing configuration
        self.share_backbone = share_backbone
        self.share_decoder = share_decoder
        
        # Log configuration
        self.logger.info(f"Initializing multi-speaker LoRA trainer for {len(speaker_ids)} speakers")
        self.logger.info(f"Speaker IDs: {speaker_ids}")
        self.logger.info(f"Sharing backbone: {share_backbone}")
        self.logger.info(f"Sharing decoder: {share_decoder}")
        self.logger.info(f"LoRA parameters: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        self.logger.info(f"Target modules: {self.target_modules}")
        
        # Create speaker-specific trainers
        self.trainers = {}
        self.initialize_trainers()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
    
    def initialize_trainers(self):
        """Initialize individual trainers for each speaker."""
        self.logger.info(f"Creating individual trainers for {len(self.speaker_ids)} speakers")
        
        # Create a temporary directory for shared model
        shared_tmp_dir = tempfile.mkdtemp(prefix="csm_shared_")
        shared_model_path = os.path.join(shared_tmp_dir, "shared_model.safetensors")
        
        # If sharing any components, create a shared trainer first
        shared_trainer = None
        if self.share_backbone or self.share_decoder:
            # Create shared trainer with specified components
            self.logger.info("Creating shared trainer for common components")
            
            # Determine which layers to target in shared trainer
            shared_target_backbone_layers = self.target_backbone_layers if self.share_backbone else []
            shared_target_decoder_layers = self.target_decoder_layers if self.share_decoder else []
            
            # Create a temporary directory for speaker models
            os.makedirs(os.path.join(self.output_dir, "shared"), exist_ok=True)
            
            # Create shared trainer
            shared_trainer = CSMLoRATrainer(
                model_path=self.model_path,
                output_dir=os.path.join(self.output_dir, "shared"),
                learning_rate=self.learning_rate,
                semantic_weight=self.semantic_weight,
                acoustic_weight=self.acoustic_weight,
                weight_decay=self.weight_decay,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                target_layers=shared_target_backbone_layers if self.share_backbone else None,
                lora_use_bias=self.lora_use_bias
            )
            
            # Save shared model to use as a base for speaker-specific models
            shared_trainer.save_model(shared_model_path, save_mode="lora")
            self.logger.info(f"Saved shared model to {shared_model_path}")
        
        # Create speaker-specific trainers
        for speaker_id in self.speaker_ids:
            # Create a directory for this speaker
            speaker_dir = os.path.join(self.output_dir, f"speaker_{speaker_id}")
            os.makedirs(speaker_dir, exist_ok=True)
            
            # Determine which components this speaker should have
            speaker_target_backbone_layers = [] if self.share_backbone else self.target_backbone_layers
            speaker_target_decoder_layers = [] if self.share_decoder else self.target_decoder_layers
            
            # Start with either the shared model or the original model
            base_model_path = shared_model_path if shared_trainer is not None else self.model_path
            
            # Create trainer for this speaker
            self.logger.info(f"Creating trainer for speaker {speaker_id}")
            trainer = CSMLoRATrainer(
                model_path=base_model_path,
                output_dir=speaker_dir,
                learning_rate=self.learning_rate,
                semantic_weight=self.semantic_weight,
                acoustic_weight=self.acoustic_weight,
                weight_decay=self.weight_decay,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                target_layers=speaker_target_backbone_layers,
                lora_use_bias=self.lora_use_bias
            )
            
            # Store trainer in dictionary
            self.trainers[speaker_id] = trainer
        
        # Store shared trainer for later use
        self.shared_trainer = shared_trainer
        self.logger.info(f"Created {len(self.trainers)} speaker-specific trainers")
    
    def prepare_optimizers(self):
        """Prepare optimizers for all trainers."""
        self.logger.info("Preparing optimizers for all trainers")
        if self.shared_trainer is not None:
            self.shared_trainer.prepare_optimizer()
        
        for speaker_id, trainer in self.trainers.items():
            self.logger.info(f"Preparing optimizer for speaker {speaker_id}")
            trainer.prepare_optimizer()
    
    def train(
        self,
        speaker_datasets: Dict[int, Tuple],
        batch_size: int = 2,
        epochs: int = 5,
        val_every: int = 100,
        save_every: int = 500,
        max_grad_norm: float = 1.0,
        resume_from: Optional[Dict[int, str]] = None
    ):
        """
        Train all speakers' LoRA adapters.
        
        Args:
            speaker_datasets: Dictionary mapping speaker IDs to (train_dataset, val_dataset) tuples
            batch_size: Batch size
            epochs: Number of epochs
            val_every: Validate every N steps
            save_every: Save checkpoint every N steps
            max_grad_norm: Maximum gradient norm for clipping
            resume_from: Dictionary mapping speaker IDs to checkpoint paths to resume from
            
        Returns:
            Dictionary mapping speaker IDs to best validation losses
        """
        self.logger.info(f"Starting multi-speaker training for {len(self.speaker_ids)} speakers")
        
        # Validate speaker datasets
        for speaker_id in self.speaker_ids:
            if speaker_id not in speaker_datasets:
                self.logger.warning(f"No dataset provided for speaker {speaker_id}")
        
        # Prepare all optimizers
        self.prepare_optimizers()
        
        # Resume from checkpoints if provided
        if resume_from:
            for speaker_id, checkpoint_path in resume_from.items():
                if speaker_id in self.trainers:
                    self.logger.info(f"Resuming speaker {speaker_id} from {checkpoint_path}")
                    metadata = load_checkpoint_mlx(
                        checkpoint_path,
                        self.trainers[speaker_id].model,
                        self.trainers[speaker_id].optimizer
                    )
                    self.trainers[speaker_id].epoch = metadata["epoch"]
                    self.trainers[speaker_id].global_step = metadata["global_step"]
                    self.trainers[speaker_id].best_loss = metadata["loss"]
        
        # Start training all speakers
        best_losses = {}
        for epoch in range(epochs):
            epoch_start = time.time()
            self.logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            # Train each speaker
            for speaker_id, (train_dataset, val_dataset) in speaker_datasets.items():
                if speaker_id not in self.trainers:
                    continue
                
                trainer = self.trainers[speaker_id]
                self.logger.info(f"Training speaker {speaker_id} for epoch {epoch+1}")
                
                # Train for one epoch
                best_loss = trainer.train(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    batch_size=batch_size,
                    epochs=1,  # Just one epoch at a time
                    val_every=val_every,
                    save_every=save_every,
                    max_grad_norm=max_grad_norm
                )
                
                # Update best loss
                best_losses[speaker_id] = best_loss
            
            # Log epoch results
            epoch_duration = time.time() - epoch_start
            self.logger.info(f"Completed epoch {epoch+1} in {epoch_duration:.2f}s")
            for speaker_id, best_loss in best_losses.items():
                self.logger.info(f"Speaker {speaker_id} best loss: {best_loss:.6f}")
            
            # Update global epoch counter
            self.epoch = epoch + 1
        
        # Save final models
        self.save_all_models()
        
        return best_losses
    
    def save_all_models(self):
        """Save all speaker models."""
        self.logger.info("Saving all speaker models")
        
        # Save shared model if it exists
        if self.shared_trainer is not None:
            shared_path = os.path.join(self.output_dir, "shared", "shared_lora.safetensors")
            self.logger.info(f"Saving shared model to {shared_path}")
            self.shared_trainer.save_model(shared_path, save_mode="lora")
        
        # Save each speaker model
        for speaker_id, trainer in self.trainers.items():
            save_path = os.path.join(self.output_dir, f"speaker_{speaker_id}", f"speaker_{speaker_id}_lora.safetensors")
            self.logger.info(f"Saving model for speaker {speaker_id} to {save_path}")
            trainer.save_model(save_path, save_mode="lora")
    
    def load_speaker_model(self, speaker_id: int, checkpoint_path: str):
        """
        Load LoRA weights for a specific speaker.
        
        Args:
            speaker_id: Speaker ID
            checkpoint_path: Path to checkpoint to load
        """
        if speaker_id not in self.trainers:
            self.logger.error(f"No trainer found for speaker {speaker_id}")
            return
        
        self.logger.info(f"Loading model for speaker {speaker_id} from {checkpoint_path}")
        self.trainers[speaker_id].load_lora_weights(checkpoint_path)
    
    def generate_sample(
        self,
        text: str,
        speaker_id: int,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a sample using a specific speaker's model.
        
        Args:
            text: Text to generate
            speaker_id: Speaker ID to use
            output_path: Path to save output (optional)
            
        Returns:
            Path to generated audio file
        """
        if speaker_id not in self.trainers:
            self.logger.error(f"No trainer found for speaker {speaker_id}")
            return ""
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"speaker_{speaker_id}", f"sample_{int(time.time())}.wav")
        
        self.logger.info(f"Generating sample for speaker {speaker_id}: '{text}'")
        return self.trainers[speaker_id].generate_sample(
            text=text,
            speaker_id=speaker_id,
            output_path=output_path
        )
    
    def merge_speaker_models(self, shared_weight: float = 0.5):
        """
        Merge speaker models to create an interpolated model.
        
        Args:
            shared_weight: Weight of shared components (0.0-1.0)
            
        Returns:
            Dictionary mapping speaker IDs to merged model paths
        """
        merged_models = {}
        
        # Only merge if we have shared components
        if self.shared_trainer is None:
            self.logger.warning("No shared components to merge")
            return merged_models
        
        self.logger.info(f"Merging models with shared_weight={shared_weight}")
        
        # Get shared model parameters
        if hasattr(self.shared_trainer.model, 'get_lora_params'):
            shared_params = self.shared_trainer.model.get_lora_params()
        else:
            shared_params = {}
        
        # Merge each speaker model
        for speaker_id, trainer in self.trainers.items():
            # Get speaker-specific parameters
            if hasattr(trainer.model, 'get_lora_params'):
                speaker_params = trainer.model.get_lora_params()
            else:
                speaker_params = {}
            
            # Create merged parameters
            merged_params = {}
            
            # Add all parameters from shared model with weight
            for param_name, param_value in shared_params.items():
                merged_params[param_name] = param_value * shared_weight
            
            # Add all parameters from speaker model with weight
            for param_name, param_value in speaker_params.items():
                if param_name in merged_params:
                    # Weighted average for shared parameters
                    merged_params[param_name] += param_value * (1.0 - shared_weight)
                else:
                    # Speaker-specific parameters get full weight
                    merged_params[param_name] = param_value
            
            # Update the model with merged parameters
            if hasattr(trainer.model, 'update'):
                trainer.model.update(merged_params)
            
            # Save the merged model
            merged_path = os.path.join(self.output_dir, f"speaker_{speaker_id}", f"speaker_{speaker_id}_merged.safetensors")
            trainer.save_model(merged_path, save_mode="lora")
            
            merged_models[speaker_id] = merged_path
            self.logger.info(f"Saved merged model for speaker {speaker_id} to {merged_path}")
        
        return merged_models