#!/usr/bin/env python
"""
Style transfer LoRA fine-tuning example for CSM models.

This script demonstrates how to use LoRA for style transfer with CSM models.
It uses more aggressive LoRA settings with higher rank and alpha values to
better capture style characteristics.
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
        description="Style transfer LoRA fine-tuning for CSM models"
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
        help="Directory containing audio files in target style"
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
        default="./style_finetuned",
        help="Directory to save fine-tuned model and logs (default: ./style_finetuned)"
    )
    
    # Style-specific arguments
    parser.add_argument(
        "--style-name",
        type=str,
        default="custom_style",
        help="Name of the style being trained (for model naming)"
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Speaker ID to use for training (default: 0)"
    )
    
    # LoRA parameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank - higher values capture more style details (default: 32)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=64.0,
        help="LoRA alpha scaling factor (default: 64.0)"
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj", "o_proj"],
        help="Which modules to apply LoRA to (default: q_proj v_proj o_proj)"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
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
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=120.0,
        help="Weight for semantic token loss (default: 120.0)"
    )
    
    # Generation options
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate sample audio after training"
    )
    parser.add_argument(
        "--sample-prompts",
        type=str,
        nargs="+",
        default=["This is a test of the style-transferred voice model."],
        help="Prompts for sample generation"
    )
    
    return parser.parse_args()

def main():
    """Main function for style transfer LoRA fine-tuning."""
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
    logger.info("Style transfer fine-tuning parameters:")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Audio directory: {args.audio_dir}")
    logger.info(f"  Transcript directory: {args.transcript_dir}")
    logger.info(f"  Style name: {args.style_name}")
    logger.info(f"  Speaker ID: {args.speaker_id}")
    logger.info(f"  LoRA rank: {args.lora_r}")
    logger.info(f"  LoRA alpha: {args.lora_alpha}")
    logger.info(f"  Target modules: {args.target_modules}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Semantic weight: {args.semantic_weight}")
    
    # Initialize trainer with style-specific settings
    logger.info(f"Initializing LoRA trainer for style transfer (r={args.lora_r}, alpha={args.lora_alpha})")
    trainer = CSMLoRATrainer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        semantic_weight=args.semantic_weight,
        acoustic_weight=1.0,
        # Style-specific LoRA settings (higher rank, higher alpha):
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,  # Add some dropout for better generalization
        # Target more modules for comprehensive style capture:
        target_modules=args.target_modules,
        # Use all layers
        target_layers=None,
        # Use bias for more expressivity
        lora_use_bias=True
    )
    
    # Prepare data
    logger.info("Preparing style datasets")
    from csm.data import CSMDataProcessor, ContextualExampleGenerator
    from csm.training.data import MLXDataset
    
    # Process style audio files
    processor = CSMDataProcessor()
    all_examples = []
    
    # Get list of style audio files
    import glob
    audio_files = glob.glob(os.path.join(args.audio_dir, "**/*.wav"), recursive=True)
    logger.info(f"Found {len(audio_files)} style audio files")
    
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
            logger.info(f"Processed style audio: {audio_file}")
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
    
    logger.info(f"Processed {len(all_examples)} style examples")
    
    # For style, we use less context to focus on the style itself
    generator = ContextualExampleGenerator(max_context_turns=1)
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
    logger.info("Starting style transfer fine-tuning")
    best_loss = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_every=50,  # Validate more frequently
        save_every=200,
        max_grad_norm=1.0
    )
    
    logger.info(f"Style transfer fine-tuning completed with best validation loss: {best_loss:.6f}")
    
    # Save the fine-tuned model
    lora_save_path = os.path.join(args.output_dir, f"{args.style_name}_lora.safetensors")
    full_save_path = os.path.join(args.output_dir, f"{args.style_name}_full.safetensors")
    
    # Save both LoRA and full model for flexibility
    logger.info(f"Saving style LoRA model to {lora_save_path}")
    trainer.save_model(lora_save_path, save_mode="lora")
    
    logger.info(f"Saving full merged model to {full_save_path}")
    trainer.save_model(full_save_path, save_mode="full")
    
    # Generate samples if requested
    if args.generate_samples:
        for i, prompt in enumerate(args.sample_prompts):
            logger.info(f"Generating style sample {i+1} with prompt: '{prompt}'")
            sample_path = os.path.join(args.output_dir, f"sample_{i+1}.wav")
            try:
                trainer.generate_sample(
                    text=prompt,
                    speaker_id=args.speaker_id,
                    output_path=sample_path
                )
                logger.info(f"Style sample audio saved to {sample_path}")
            except Exception as e:
                logger.error(f"Error generating style sample: {e}")
    
    logger.info("Style transfer LoRA fine-tuning completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())