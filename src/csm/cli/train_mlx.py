"""Command-line tool for training CSM with MLX on Apple Silicon."""

import os
import sys
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from csm.training.mlx_trainer import CSMMLXTrainer
from csm.data import (
    CSMDataProcessor,
    ContextualExampleGenerator
)
from csm.data.training_data import TrainingExample
from csm.training.utils import setup_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train CSM model with MLX")
    
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
        default="csm_mlx_trained",
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
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length for training"
    )
    
    training_group.add_argument(
        "--conversational",
        action="store_true",
        help="Enable conversational context in training"
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
    
    # MLX-specific arguments
    mlx_group = parser.add_argument_group("MLX")
    mlx_group.add_argument(
        "--autotune",
        action="store_true",
        help="Enable MLX kernel autotuning"
    )
    
    mlx_group.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for MLX operations"
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


class MLXDataset:
    """Dataset for MLX training with efficient conversion to MLX arrays."""
    
    def __init__(self, examples, text_tokenizer=None, audio_tokenizer=None, max_seq_len=2048):
        """
        Initialize MLX dataset.
        
        Args:
            examples: List of training examples (in context-target format)
            text_tokenizer: Tokenizer for text
            audio_tokenizer: Tokenizer for audio
            max_seq_len: Maximum sequence length
        """
        self.examples = examples
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.max_seq_len = max_seq_len
        
        # Check if we need to create custom tokenizers
        if self.text_tokenizer is None or self.audio_tokenizer is None:
            logger = logging.getLogger("MLXDataset")
            logger.warning("Tokenizers not provided, using fallback tokenizers")
            
            # Create fallback tokenizers if needed
            if self.text_tokenizer is None:
                from csm.models.model import get_tokenizer
                try:
                    self.text_tokenizer = get_tokenizer()
                except Exception:
                    # Mock tokenizer
                    class MockTokenizer:
                        def encode(self, text):
                            # Convert text to random tokens
                            return [1] * (len(text) // 3)
                    self.text_tokenizer = MockTokenizer()
            
            if self.audio_tokenizer is None:
                # Mock audio tokenizer
                class MockAudioTokenizer:
                    def encode(self, audio):
                        # Create random audio tokens
                        if isinstance(audio, torch.Tensor):
                            length = audio.size(-1) // 1000
                            return [torch.ones((32, length))]
                        else:
                            # Completely mock
                            return [torch.ones((32, 20))]
                self.audio_tokenizer = MockAudioTokenizer()
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def get_batch(self, batch_idx, batch_size):
        """
        Get a batch of data for MLX training.
        
        Args:
            batch_idx: Batch index
            batch_size: Batch size
            
        Returns:
            Dictionary with input_tokens, input_masks, target_audio_tokens as MLX arrays
        """
        logger = logging.getLogger("MLXDataset")
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(self.examples))
        
        batch_examples = self.examples[start_idx:end_idx]
        
        # Process each example - similar to CSMDataset.__getitem__
        input_tokens_list = []
        input_masks_list = []
        target_audio_tokens_list = []
        
        for example in batch_examples:
            try:
                # Process context segments
                tokens = []
                masks = []
                
                # Process context segments first if any
                for ctx in example.get("context", []):
                    try:
                        # Process context segments into tokens and masks
                        text_tokens, text_masks = self._tokenize_text_segment(ctx.text, ctx.speaker_id)
                        tokens.append(text_tokens)
                        masks.append(text_masks)
                        
                        # If available and needed, also process audio context
                        audio_tokens, audio_masks = self._tokenize_audio(ctx.audio)
                        tokens.append(audio_tokens)
                        masks.append(audio_masks)
                    except Exception as ctx_e:
                        logger.warning(f"Error processing context: {ctx_e}")
                        # Continue with next context item
                        continue
                
                # Process target text only (for input)
                target = example["target"]
                target_text_tokens, target_text_masks = self._tokenize_text_segment(
                    target.text, target.speaker_id
                )
                tokens.append(target_text_tokens)
                masks.append(target_text_masks)
                
                # Concatenate all tokens and masks
                input_tokens = torch.cat(tokens, dim=0) if tokens else torch.zeros((0, 33), dtype=torch.int64)
                input_masks = torch.cat(masks, dim=0) if masks else torch.zeros((0, 33), dtype=torch.bool)
                
                # Process target audio (for output)
                target_audio_tokens = self._tokenize_audio_for_target(target.audio)
                
                # Ensure we're within max sequence length
                if input_tokens.shape[0] > self.max_seq_len:
                    # Truncate from the beginning, but always keep the target text
                    keep_len = min(self.max_seq_len, target_text_tokens.shape[0])
                    start_idx = input_tokens.shape[0] - keep_len
                    input_tokens = input_tokens[start_idx:]
                    input_masks = input_masks[start_idx:]
                
                # Convert to numpy arrays for MLX compatibility
                input_tokens_list.append(input_tokens.numpy())
                input_masks_list.append(input_masks.numpy())
                target_audio_tokens_list.append(target_audio_tokens.numpy())
                
            except Exception as e:
                logger.error(f"Error processing example {batch_idx}: {e}")
                # Add dummy data to maintain batch size
                input_tokens_list.append(np.zeros((10, 33), dtype=np.int32))
                input_masks_list.append(np.zeros((10, 33), dtype=np.bool_))
                target_audio_tokens_list.append(np.zeros((10, 32), dtype=np.int32))
        
        # Ensure we have at least one valid example
        if not input_tokens_list or not input_masks_list or not target_audio_tokens_list:
            logger.error("No valid examples in batch")
            # Return empty batch with expected shapes
            return {
                "input_tokens": mx.array(np.zeros((0, 1, 33), dtype=np.int32)),
                "input_masks": mx.array(np.zeros((0, 1, 33), dtype=np.bool_)),
                "target_audio_tokens": mx.array(np.zeros((0, 1, 32), dtype=np.int32))
            }
        
        try:
            # Pad to the same lengths within the batch
            max_input_len = max(tokens.shape[0] for tokens in input_tokens_list if tokens.shape[0] > 0)
            max_target_len = max(tokens.shape[0] for tokens in target_audio_tokens_list if tokens.shape[0] > 0)
            
            # Create padded batch arrays
            batch_input_tokens = np.zeros((end_idx - start_idx, max_input_len, 33), dtype=np.int32)
            batch_input_masks = np.zeros((end_idx - start_idx, max_input_len, 33), dtype=np.bool_)
            batch_target_audio_tokens = np.zeros((end_idx - start_idx, max_target_len, 32), dtype=np.int32)
            
            # Fill with actual data
            for i, (input_tokens, input_masks, target_audio_tokens) in enumerate(zip(
                input_tokens_list, input_masks_list, target_audio_tokens_list
            )):
                # Handle input tokens and masks (if valid shapes)
                if input_tokens.shape[0] > 0 and input_tokens.shape[1] == 33:
                    input_length = min(input_tokens.shape[0], max_input_len)
                    batch_input_tokens[i, :input_length, :] = input_tokens[:input_length]
                    batch_input_masks[i, :input_length, :] = input_masks[:input_length]
                
                # Handle target audio tokens (if valid shapes)
                if target_audio_tokens.shape[0] > 0 and target_audio_tokens.shape[1] <= 32:
                    target_length = min(target_audio_tokens.shape[0], max_target_len)
                    target_width = target_audio_tokens.shape[1]
                    batch_target_audio_tokens[i, :target_length, :target_width] = target_audio_tokens[:target_length]
            
            # Convert to MLX arrays
            return {
                "input_tokens": mx.array(batch_input_tokens),
                "input_masks": mx.array(batch_input_masks),
                "target_audio_tokens": mx.array(batch_target_audio_tokens)
            }
        
        except Exception as batch_e:
            logger.error(f"Error creating batch: {batch_e}")
            # Return a minimal valid batch as fallback
            return {
                "input_tokens": mx.array(np.zeros((1, 10, 33), dtype=np.int32)),
                "input_masks": mx.array(np.zeros((1, 10, 33), dtype=np.bool_)),
                "target_audio_tokens": mx.array(np.zeros((1, 10, 32), dtype=np.int32))
            }
    
    def _tokenize_text_segment(self, text, speaker):
        """
        Tokenize a text segment.
        
        Args:
            text: Text to tokenize
            speaker: Speaker ID
            
        Returns:
            Tuple of (tokens, masks) as PyTorch tensors
        """
        try:
            # Use tokenizer if available
            if hasattr(self.text_tokenizer, 'encode'):
                text_tokens = self.text_tokenizer.encode(f"[{speaker}]{text}")
                if not isinstance(text_tokens, list) and hasattr(text_tokens, 'tolist'):
                    text_tokens = text_tokens.tolist()
            else:
                # Fallback
                text_tokens = [1] * (len(text) // 3)
            
            # Create frame and mask
            text_frame = torch.zeros((len(text_tokens), 33), dtype=torch.int64)
            text_frame_mask = torch.zeros((len(text_tokens), 33), dtype=torch.bool)
            text_frame[:, -1] = torch.tensor(text_tokens)
            text_frame_mask[:, -1] = True
            
            return text_frame, text_frame_mask
        except Exception as e:
            logger = logging.getLogger("MLXDataset")
            logger.warning(f"Error tokenizing text: {e}")
            # Return minimal valid output
            return torch.zeros((1, 33), dtype=torch.int64), torch.zeros((1, 33), dtype=torch.bool)
    
    def _tokenize_audio(self, audio):
        """
        Tokenize an audio segment.
        
        Args:
            audio: Audio tensor
            
        Returns:
            Tuple of (tokens, masks) as PyTorch tensors
        """
        try:
            # Use tokenizer if available
            if hasattr(self.audio_tokenizer, 'encode') and hasattr(audio, 'unsqueeze'):
                audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))
                if isinstance(audio_tokens, list) and len(audio_tokens) > 0:
                    audio_tokens = audio_tokens[0]
            else:
                # Fallback - create mock tokens
                audio_tokens = torch.ones((32, 20))
            
            # Add EOS frame if needed
            if audio_tokens.dim() == 2:
                eos_frame = torch.zeros((audio_tokens.size(0), 1))
                audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
            
            # Create frame and mask
            audio_frame = torch.zeros((audio_tokens.size(1), 33), dtype=torch.int64)
            audio_frame_mask = torch.zeros((audio_tokens.size(1), 33), dtype=torch.bool)
            audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
            audio_frame_mask[:, :-1] = True
            
            return audio_frame, audio_frame_mask
        except Exception as e:
            logger = logging.getLogger("MLXDataset")
            logger.warning(f"Error tokenizing audio: {e}")
            # Return minimal valid output
            return torch.zeros((1, 33), dtype=torch.int64), torch.zeros((1, 33), dtype=torch.bool)
    
    def _tokenize_audio_for_target(self, audio):
        """
        Process audio specifically for target representation.
        
        Args:
            audio: Audio tensor
            
        Returns:
            Target audio tokens as PyTorch tensor
        """
        try:
            if hasattr(self.audio_tokenizer, 'encode'):
                if hasattr(audio, 'unsqueeze'):
                    try:
                        audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))
                        if isinstance(audio_tokens, list) and len(audio_tokens) > 0:
                            audio_tokens = audio_tokens[0]
                        if audio_tokens.dim() > 1:
                            audio_tokens = audio_tokens.transpose(0, 1)
                        return audio_tokens
                    except Exception:
                        pass
            
            # Fallback for any failure
            return torch.ones((5, 32), dtype=torch.int64)
        except Exception as e:
            logger = logging.getLogger("MLXDataset")
            logger.warning(f"Error creating target audio tokens: {e}")
            # Return minimal valid output
            return torch.ones((5, 32), dtype=torch.int64)


def load_data(args) -> List[TrainingExample]:
    """
    Load data from the specified directories.
    
    Args:
        args: Command-line arguments
        
    Returns:
        List of TrainingExample objects
    """
    logger = setup_logger("csm_mlx_data_loader")
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


def prepare_mlx_dataset(examples: List[TrainingExample], args) -> tuple:
    """
    Prepare datasets for MLX training.
    
    Args:
        examples: List of examples
        args: Command-line arguments
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger = setup_logger("csm_mlx_dataset_prep")
    
    # Split into training and validation sets
    val_size = int(len(examples) * args.val_split)
    train_size = len(examples) - val_size
    
    logger.info(f"Splitting data: {train_size} training, {val_size} validation examples")
    
    train_examples = examples[:train_size]
    val_examples = examples[train_size:]
    
    # Load a model to get tokenizers
    logger.info("Loading model to get tokenizers...")
    text_tokenizer = None
    audio_tokenizer = None
    
    try:
        # Try to load tokenizers from model
        from csm.models.model import get_tokenizer
        text_tokenizer = get_tokenizer()
        logger.info("Successfully loaded text tokenizer")
    except Exception as e:
        logger.warning(f"Failed to load text tokenizer: {e}")
    
    try:
        # Try to load audio tokenizer from model
        from csm.generator import load_csm_1b
        generator = load_csm_1b(args.model_path, device="cpu")
        audio_tokenizer = generator._audio_tokenizer
        logger.info("Successfully loaded audio tokenizer")
    except Exception as e:
        logger.warning(f"Failed to load audio tokenizer: {e}")
    
    # Create contextual examples for training
    logger.info("Creating contextual examples...")
    context_generator = ContextualExampleGenerator(max_context_turns=3)
    
    # Check if we should create conversational examples with context
    if hasattr(args, 'conversational') and args.conversational:
        logger.info("Creating contextual conversation examples...")
        
        # Group examples by metadata or filename to form conversations
        conversations = {}
        for ex in examples:
            # Use metadata to group if available
            file_id = None
            if ex.metadata and 'file_id' in ex.metadata:
                file_id = ex.metadata['file_id']
            elif ex.metadata and 'id' in ex.metadata:
                file_id = ex.metadata['id']
                
            # If no metadata, try to use other fields for grouping
            if not file_id and hasattr(ex, 'filename'):
                file_id = ex.filename
                
            # If still no ID, use speaker as fallback
            if not file_id:
                file_id = f"speaker_{ex.speaker_id}"
                
            if file_id not in conversations:
                conversations[file_id] = []
            conversations[file_id].append(ex)
        
        # Form contextual examples from conversations
        train_contextual = []
        val_contextual = []
        
        # Process training examples
        for conv_id, conv_examples in conversations.items():
            # Only include in training if first example is in training set
            if conv_examples[0] in train_examples:
                # Create conversational examples with context
                conv_contextual = context_generator.create_contextual_examples(conv_examples)
                train_contextual.extend(conv_contextual)
        
        # Process validation examples (no context for simplicity)
        val_contextual = [{"context": [], "target": ex} for ex in val_examples]
        
        logger.info(f"Created {len(train_contextual)} contextual training examples")
        logger.info(f"Created {len(val_contextual)} validation examples")
    else:
        # For non-conversational training, examples are independent (no context)
        logger.info("Creating independent examples (no conversation context)...")
        train_contextual = [{"context": [], "target": ex} for ex in train_examples]
        val_contextual = [{"context": [], "target": ex} for ex in val_examples]
    
    # Create datasets
    logger.info("Creating MLX datasets...")
    train_dataset = MLXDataset(
        train_contextual,
        text_tokenizer,
        audio_tokenizer,
        max_seq_len=args.max_seq_len if hasattr(args, 'max_seq_len') else 2048
    )
    
    val_dataset = MLXDataset(
        val_contextual,
        text_tokenizer,
        audio_tokenizer,
        max_seq_len=args.max_seq_len if hasattr(args, 'max_seq_len') else 2048
    )
    
    return train_dataset, val_dataset


def main():
    """Run the MLX training process."""
    if not HAS_MLX:
        print("Error: MLX is required for this tool.")
        print("Install it with: pip install mlx")
        sys.exit(1)
    
    args = parse_args()
    
    # Set MLX environment variables if specified
    if args.autotune:
        os.environ["MLX_AUTOTUNE"] = "1"
    
    if args.num_threads:
        os.environ["MLX_NUM_THREADS"] = str(args.num_threads)
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(
        "csm_mlx_train", 
        args.log_file,
        level=log_level
    )
    
    logger.info("Starting CSM MLX training")
    logger.info(f"MLX configuration: autotune={args.autotune}, num_threads={args.num_threads}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    examples = load_data(args)
    
    if not examples:
        logger.error("No training examples found. Exiting.")
        sys.exit(1)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_mlx_dataset(examples, args)
    
    # Create trainer
    trainer = CSMMLXTrainer(
        model_path=args.model_path,
        output_dir=args.output_dir,
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
        epochs=args.epochs,
        val_every=args.val_every,
        save_every=args.save_every,
        max_grad_norm=args.max_grad_norm,
        resume_from=args.resume_from
    )
    
    # Generate a sample
    logger.info("Generating a sample from the trained model...")
    sample_path = trainer.generate_sample(
        text="This is a sample from the MLX fine-tuned model.",
        speaker_id=args.speaker_id
    )
    
    logger.info(f"Sample saved to {sample_path}")
    logger.info(f"Training completed with best validation loss: {best_loss:.6f}")
    logger.info(f"Model checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()