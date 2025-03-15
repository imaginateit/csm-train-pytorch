#!/usr/bin/env python
"""
Basic LoRA fine-tuning example for CSM models.

This script demonstrates how to fine-tune a CSM model using LoRA with MLX.
It shows the simplest way to perform parameter-efficient fine-tuning with
default settings for quick adaptation.
"""

import os
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Basic LoRA fine-tuning for CSM models"
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained CSM model (safetensors or PyTorch format)"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing audio files for fine-tuning"
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        required=True,
        help="Directory containing transcript files for fine-tuning"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lora_finetuned",
        help="Directory to save fine-tuned model and logs (default: ./lora_finetuned)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Speaker ID to use for training (default: 0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training (default: 2)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate sample audio after training"
    )
    parser.add_argument(
        "--sample-prompt",
        type=str,
        default="This is a test of my new fine-tuned voice model.",
        help="Prompt for sample generation"
    )
    
    return parser.parse_args()

def main():
    """Main function for basic LoRA fine-tuning."""
    # Parse arguments
    args = parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return 1
    
    if not os.path.exists(args.audio_dir):
        logger.error(f"Audio directory does not exist: {args.audio_dir}")
        return 1
    
    if not os.path.exists(args.transcript_dir):
        logger.error(f"Transcript directory does not exist: {args.transcript_dir}")
        return 1
    
    # Check for MLX
    try:
        import mlx.core as mx
    except ImportError:
        logger.error("MLX is required for LoRA fine-tuning but not installed.")
        logger.error("Install it with: pip install mlx")
        return 1
    
    # Check for CSM
    try:
        from csm.training.lora_trainer import CSMLoRATrainer
    except ImportError:
        logger.error("CSM package not found. Please install it with: pip install -e .")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Log fine-tuning parameters
    logger.info("Fine-tuning parameters:")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Audio directory: {args.audio_dir}")
    logger.info(f"  Transcript directory: {args.transcript_dir}")
    logger.info(f"  Speaker ID: {args.speaker_id}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    
    # Initialize trainer with default LoRA settings for simplicity
    logger.info("Initializing LoRA trainer with default settings (r=8, alpha=16)")
    trainer = CSMLoRATrainer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        # Default LoRA settings for simplicity:
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        # Target attention modules only for efficiency:
        target_modules=["q_proj", "v_proj"],
        # Use all layers
        target_layers=None,
        # No bias for simplicity
        lora_use_bias=False
    )
    
    # Prepare data
    logger.info("Preparing datasets")
    from csm.data import CSMDataProcessor, ContextualExampleGenerator
    from csm.training.data import MLXDataset
    
    # Process audio files
    processor = CSMDataProcessor()
    all_examples = []
    
    # Get list of audio files
    import glob
    audio_files = glob.glob(os.path.join(args.audio_dir, "**/*.wav"), recursive=True)
    logger.info(f"Found {len(audio_files)} audio files")
    
    for audio_file in audio_files:
        # Get matching transcript file
        base_name = os.path.basename(audio_file).replace(".wav", "")
        transcript_file = os.path.join(args.transcript_dir, f"{base_name}.txt")
        
        # Skip if transcript doesn't exist
        if not os.path.exists(transcript_file):
            logger.warning(f"No transcript found for {audio_file}, skipping")
            continue
        
        # Process file
        try:
            examples = processor.prepare_from_audio_file(
                audio_file,
                transcript_file,
                speaker_id=args.speaker_id
            )
            all_examples.extend(examples)
            logger.info(f"Processed {audio_file}")
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
    
    logger.info(f"Processed {len(all_examples)} examples")
    
    # Add context (optional, but helps with coherence)
    generator = ContextualExampleGenerator(max_context_turns=2)
    contextual_examples = generator.create_contextual_examples(all_examples)
    
    # Split into train and validation sets
    import random
    random.shuffle(contextual_examples)
    val_split = 0.1
    val_size = int(len(contextual_examples) * val_split)
    train_examples = contextual_examples[val_size:]
    val_examples = contextual_examples[:val_size]
    
    logger.info(f"Split into {len(train_examples)} train and {len(val_examples)} validation examples")
    
    # Create datasets
    train_dataset = MLXDataset(train_examples, max_seq_len=2048)
    val_dataset = MLXDataset(val_examples, max_seq_len=2048)
    
    # Prepare optimizer
    logger.info("Preparing optimizer")
    trainer.prepare_optimizer()
    
    # Train model
    logger.info("Starting fine-tuning")
    best_loss = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_every=100,
        save_every=500,
        max_grad_norm=1.0
    )
    
    logger.info(f"Fine-tuning completed with best validation loss: {best_loss:.6f}")
    
    # Save the fine-tuned model
    save_path = os.path.join(args.output_dir, "fine_tuned_model.safetensors")
    logger.info(f"Saving fine-tuned LoRA model to {save_path}")
    trainer.save_model(save_path, save_mode="lora")
    
    # Generate sample if requested
    if args.generate_sample:
        logger.info("Generating sample audio")
        sample_path = os.path.join(args.output_dir, "sample.wav")
        try:
            trainer.generate_sample(
                text=args.sample_prompt,
                speaker_id=args.speaker_id,
                output_path=sample_path
            )
            logger.info(f"Sample audio saved to {sample_path}")
        except Exception as e:
            logger.error(f"Error generating sample: {e}")
    
    logger.info("Basic LoRA fine-tuning completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())