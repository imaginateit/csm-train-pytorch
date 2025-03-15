#!/usr/bin/env python
"""
Low-resource LoRA fine-tuning example for CSM models.

This script demonstrates how to fine-tune a CSM model using LoRA with
minimal data (as few as 5-10 minutes of audio). It uses several techniques
to maximize the effectiveness of limited data.
"""

import os
import argparse
import logging
import random
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Low-resource LoRA fine-tuning for CSM models"
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
        help="Directory containing limited audio files"
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        required=True,
        help="Directory containing transcript files for audio"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lora_low_resource",
        help="Directory to save fine-tuned model and logs (default: ./lora_low_resource)"
    )
    
    # Data augmentation parameters
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Enable data augmentation to increase effective dataset size"
    )
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=3,
        help="Factor by which to augment the data (default: 3)"
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Speaker ID to use for training (default: 0)"
    )
    
    # LoRA parameters for low-resource settings
    parser.add_argument(
        "--lora-r",
        type=int,
        default=4,
        help="LoRA rank - lower for less data to prevent overfitting (default: 4)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=8.0,
        help="LoRA alpha scaling factor (default: 8.0)"
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Which modules to apply LoRA to (default: q_proj v_proj)"
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to target - can limit to fewer layers for less data (default: all)"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs - more for less data (default: 15)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training - smaller for less data (default: 1)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay - higher for less data to prevent overfitting (default: 0.05)"
    )
    
    # Generation options
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate sample audio after training"
    )
    parser.add_argument(
        "--sample-prompt",
        type=str,
        default="This is a test of voice adaptation with minimal training data.",
        help="Prompt for sample generation"
    )
    
    return parser.parse_args()

def augment_audio_data(examples, factor=3):
    """
    Augment audio data by creating variations.
    
    This is a placeholder for real audio augmentation that would:
    1. Add slight speed/pitch variations
    2. Add mild background noise
    3. Apply subtle EQ changes
    
    For this example, we just duplicate with small modifications.
    """
    augmented_examples = []
    for example in examples:
        # Add the original example
        augmented_examples.append(example)
        
        # Create augmented versions
        for i in range(factor - 1):
            # Create a shallow copy of the example
            aug_example = dict(example)
            
            # Note: In a real implementation, this would modify the audio tensor
            # with speed changes, pitch shifts, etc.
            # Here we're just demonstrating the concept
            
            augmented_examples.append(aug_example)
    
    return augmented_examples

def main():
    """Main function for low-resource LoRA fine-tuning."""
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
    log_path = os.path.join(args.output_dir, "low_resource_training.log")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_path}")
    
    # Add file handler to logger
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log parameters
    logger.info("Low-resource fine-tuning parameters:")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Audio directory: {args.audio_dir}")
    logger.info(f"  Transcript directory: {args.transcript_dir}")
    logger.info(f"  Speaker ID: {args.speaker_id}")
    logger.info(f"  Augmentation: {args.augmentation}")
    if args.augmentation:
        logger.info(f"  Augmentation factor: {args.augmentation_factor}")
    logger.info(f"  LoRA rank: {args.lora_r}")
    logger.info(f"  LoRA alpha: {args.lora_alpha}")
    logger.info(f"  Target modules: {args.target_modules}")
    logger.info(f"  Target layers: {args.target_layers}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    
    # Initialize trainer with low-resource settings
    logger.info(f"Initializing LoRA trainer for low-resource adaptation")
    trainer = CSMLoRATrainer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        log_file=log_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        # Low-resource LoRA settings (lower rank, proportional alpha):
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,  # Higher dropout to prevent overfitting on small data
        # Target fewer modules for efficiency:
        target_modules=args.target_modules,
        # Optionally target fewer layers
        target_layers=args.target_layers,
        # Add bias for more expressivity with small data
        lora_use_bias=True
    )
    
    # Prepare data - low resource scenario
    logger.info("Preparing low-resource datasets")
    from csm.data import CSMDataProcessor, ContextualExampleGenerator
    from csm.training.data import MLXDataset
    
    # Process limited audio files
    processor = CSMDataProcessor()
    all_examples = []
    
    # Get list of audio files
    import glob
    audio_files = glob.glob(os.path.join(args.audio_dir, "**/*.wav"), recursive=True)
    
    # Report on data size
    import subprocess
    total_duration = 0
    for audio_file in audio_files:
        try:
            # Try to get audio duration using soxi
            result = subprocess.run(
                ["soxi", "-D", audio_file],
                capture_output=True,
                text=True,
                check=True
            )
            duration = float(result.stdout.strip())
            total_duration += duration
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            # If soxi fails, just ignore
            pass
    
    if total_duration > 0:
        logger.info(f"Found {len(audio_files)} audio files with total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    else:
        logger.info(f"Found {len(audio_files)} audio files (duration unknown)")
    
    # Process each file
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
            logger.info(f"Processed {audio_file} into {len(examples)} examples")
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
    
    logger.info(f"Processed {len(all_examples)} examples from limited data")
    
    # Data augmentation for low-resource scenario
    if args.augmentation:
        logger.info(f"Applying data augmentation with factor {args.augmentation_factor}")
        augmented_examples = augment_audio_data(all_examples, factor=args.augmentation_factor)
        logger.info(f"Augmented dataset contains {len(augmented_examples)} examples")
        all_examples = augmented_examples
    
    # For low-resource, we maximize context to help the model
    generator = ContextualExampleGenerator(max_context_turns=3)
    contextual_examples = generator.create_contextual_examples(all_examples)
    
    # For very small datasets, use a larger validation percentage
    # to ensure we have enough validation data
    val_split = min(0.2, 5 / len(contextual_examples))
    
    # Ensure at least one example in validation
    val_size = max(1, int(len(contextual_examples) * val_split))
    
    # Split into train and validation sets
    random.shuffle(contextual_examples)
    train_examples = contextual_examples[val_size:]
    val_examples = contextual_examples[:val_size]
    
    logger.info(f"Split into {len(train_examples)} train and {len(val_examples)} validation examples")
    
    # For low-resource, we might want to use shorter sequences
    # to increase the number of training examples
    max_seq_len = min(1024, args.batch_size * 512)
    
    # Create datasets
    train_dataset = MLXDataset(train_examples, max_seq_len=max_seq_len)
    val_dataset = MLXDataset(val_examples, max_seq_len=max_seq_len)
    
    # Prepare optimizer
    logger.info("Preparing optimizer")
    trainer.prepare_optimizer()
    
    # Train model with frequent validation and early stopping
    logger.info("Starting low-resource LoRA fine-tuning")
    best_loss = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_every=max(1, len(train_dataset) // 10),  # Validate frequently
        save_every=max(1, len(train_dataset) // 5),  # Save frequently
        max_grad_norm=0.5  # Tighter gradient clipping for small datasets
    )
    
    logger.info(f"Low-resource fine-tuning completed with best validation loss: {best_loss:.6f}")
    
    # Save the fine-tuned model
    lora_save_path = os.path.join(args.output_dir, "low_resource_lora.safetensors")
    logger.info(f"Saving LoRA model to {lora_save_path}")
    trainer.save_model(lora_save_path, save_mode="lora")
    
    # Generate sample if requested
    if args.generate_samples:
        logger.info(f"Generating sample with prompt: '{args.sample_prompt}'")
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
    
    logger.info("Low-resource LoRA fine-tuning completed successfully")
    
    # Summary of tips for low-resource fine-tuning
    logger.info("\nSummary of low-resource fine-tuning tips:")
    logger.info("1. Use a lower LoRA rank (r=2-8) to prevent overfitting")
    logger.info("2. Apply data augmentation to artificially increase dataset size")
    logger.info("3. Use more training epochs (10-20) but with early stopping")
    logger.info("4. Target fewer modules (q_proj, v_proj are most effective)")
    logger.info("5. Apply higher weight decay (0.05-0.1) to prevent overfitting")
    logger.info("6. Use smaller batch size (1-2) for better per-example learning")
    logger.info("7. Select high-quality training examples with clear articulation")
    logger.info("8. Ensure transcripts are accurate and well-formatted")
    
    return 0

if __name__ == "__main__":
    exit(main())