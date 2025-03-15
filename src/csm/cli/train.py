"""Command-line tool for training CSM with PyTorch."""

import os
import sys
import argparse
import torch
import torchaudio
from pathlib import Path
from typing import List, Optional

from csm.generator import Segment
from csm.training.data import (
    CSMDataset,
    CSMDataProcessor,
    ContextualExampleGenerator,
    TrainingExample
)
from csm.training.trainer import CSMTrainer
from csm.generator import load_csm_1b
from csm.training.utils import setup_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train CSM model")
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the CSM model checkpoint"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="csm_trained",
        help="Directory to save outputs"
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing audio files (.wav)"
    )
    
    data_group.add_argument(
        "--transcript-dir",
        type=str,
        required=True,
        help="Directory containing transcript files (.txt)"
    )
    
    data_group.add_argument(
        "--alignment-dir",
        type=str,
        help="Directory containing word-level alignments (.json)"
    )
    
    data_group.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Speaker ID to assign to data (0-9)"
    )
    
    data_group.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Portion of data to use for validation (0.0-1.0)"
    )
    
    # Training arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Base learning rate"
    )
    
    training_group.add_argument(
        "--backbone-lr-multiplier",
        type=float,
        default=0.1,
        help="Multiplier for backbone learning rate"
    )
    
    training_group.add_argument(
        "--decoder-lr-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for decoder learning rate"
    )
    
    training_group.add_argument(
        "--embedding-lr-multiplier",
        type=float,
        default=0.5,
        help="Multiplier for embedding learning rate"
    )
    
    training_group.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training"
    )
    
    training_group.add_argument(
        "--accumulation-steps",
        type=int,
        default=4,
        help="Number of batches to accumulate before updating weights"
    )
    
    training_group.add_argument(
        "--semantic-weight",
        type=float,
        default=100.0,
        help="Weight for semantic token loss"
    )
    
    training_group.add_argument(
        "--acoustic-weight",
        type=float,
        default=1.0,
        help="Weight for acoustic token loss"
    )
    
    training_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer"
    )
    
    training_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    
    training_group.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone parameters during training"
    )
    
    training_group.add_argument(
        "--freeze-decoder",
        action="store_true",
        help="Freeze decoder parameters during training"
    )
    
    training_group.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze embedding parameters during training"
    )
    
    training_group.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    # Hardware arguments
    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    
    hw_group.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for data loading"
    )
    
    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    
    log_group.add_argument(
        "--val-every",
        type=int,
        default=100,
        help="Validate every N steps"
    )
    
    log_group.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )
    
    log_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def load_data(args) -> List[TrainingExample]:
    """
    Load data from the specified directories.
    
    Args:
        args: Command-line arguments
        
    Returns:
        List of TrainingExample objects
    """
    logger = setup_logger("csm_data_loader")
    logger.info("Loading data...")
    
    data_processor = CSMDataProcessor()
    examples = []
    
    # Get all audio files
    audio_dir = Path(args.audio_dir)
    audio_files = sorted(list(audio_dir.glob("*.wav")))
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Process each file
    for audio_file in audio_files:
        transcript_file = Path(args.transcript_dir) / f"{audio_file.stem}.txt"
        
        if not transcript_file.exists():
            logger.warning(f"No transcript found for {audio_file}, skipping")
            continue
        
        # Check for alignment file
        alignment_file = None
        if args.alignment_dir:
            alignment_path = Path(args.alignment_dir) / f"{audio_file.stem}.json"
            if alignment_path.exists():
                alignment_file = alignment_path
        
        logger.info(f"Processing {audio_file.name}")
        file_examples = data_processor.prepare_from_audio_file(
            audio_file,
            transcript_file,
            args.speaker_id,
            alignment_file
        )
        
        examples.extend(file_examples)
        logger.info(f"Generated {len(file_examples)} examples from {audio_file.name}")
    
    logger.info(f"Total examples: {len(examples)}")
    return examples


def prepare_dataset(examples: List[TrainingExample], args) -> tuple:
    """
    Prepare datasets for training.
    
    Args:
        examples: List of examples
        args: Command-line arguments
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger = setup_logger("csm_dataset_prep")
    
    # Split into training and validation sets
    val_size = int(len(examples) * args.val_split)
    train_size = len(examples) - val_size
    
    logger.info(f"Splitting data: {train_size} training, {val_size} validation examples")
    
    train_examples = examples[:train_size]
    val_examples = examples[train_size:]
    
    # Load a model to get tokenizers
    logger.info("Loading model to get tokenizers...")
    generator = load_csm_1b(args.model_path, device="cpu")
    text_tokenizer = generator._text_tokenizer
    audio_tokenizer = generator._audio_tokenizer
    
    # Create contextual examples for training
    logger.info("Creating contextual examples...")
    context_generator = ContextualExampleGenerator(max_context_turns=3)
    
    # For training, we assume examples are independent (no context)
    train_contextual = [{"context": [], "target": ex} for ex in train_examples]
    val_contextual = [{"context": [], "target": ex} for ex in val_examples]
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = CSMDataset(
        train_contextual,
        text_tokenizer,
        audio_tokenizer
    )
    
    val_dataset = CSMDataset(
        val_contextual,
        text_tokenizer,
        audio_tokenizer
    )
    
    return train_dataset, val_dataset


def main():
    """Run the training process."""
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(
        "csm_train", 
        args.log_file,
        level=log_level
    )
    
    logger.info("Starting CSM training")
    logger.info(f"Device: {args.device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    examples = load_data(args)
    
    if not examples:
        logger.error("No training examples found. Exiting.")
        sys.exit(1)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_dataset(examples, args)
    
    # Create trainer
    trainer = CSMTrainer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        log_file=args.log_file,
        learning_rate=args.learning_rate,
        backbone_lr_multiplier=args.backbone_lr_multiplier,
        decoder_lr_multiplier=args.decoder_lr_multiplier,
        embedding_lr_multiplier=args.embedding_lr_multiplier,
        semantic_weight=args.semantic_weight,
        acoustic_weight=args.acoustic_weight,
        weight_decay=args.weight_decay
    )
    
    # Prepare optimizer
    trainer.prepare_optimizer(
        freeze_backbone=args.freeze_backbone,
        freeze_decoder=args.freeze_decoder,
        freeze_embeddings=args.freeze_embeddings
    )
    
    # Train model
    best_loss = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        val_every=args.val_every,
        save_every=args.save_every,
        max_grad_norm=args.max_grad_norm,
        resume_from=args.resume_from
    )
    
    # Generate a sample
    logger.info("Generating a sample from the trained model...")
    sample_path = trainer.generate_sample(
        text="This is a sample from the fine-tuned model.",
        speaker_id=args.speaker_id
    )
    
    logger.info(f"Sample saved to {sample_path}")
    logger.info(f"Training completed with best validation loss: {best_loss:.6f}")
    logger.info(f"Model checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()