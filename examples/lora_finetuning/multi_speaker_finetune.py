#!/usr/bin/env python
"""
Multi-speaker LoRA fine-tuning example for CSM models.

This script demonstrates how to fine-tune a CSM model for multiple
speakers in a single training run, sharing common components while
maintaining speaker-specific adaptation layers.
"""

import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-speaker LoRA fine-tuning for CSM models"
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained CSM model (safetensors or PyTorch format)"
    )
    parser.add_argument(
        "--speakers-config",
        type=str,
        required=True,
        help="Path to speakers configuration JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./multi_speaker_finetuned",
        help="Directory to save fine-tuned models and logs"
    )
    
    # Component sharing configuration
    parser.add_argument(
        "--share-backbone",
        action="store_true",
        default=True,
        help="Share backbone LoRA weights across speakers (default: True)"
    )
    parser.add_argument(
        "--share-decoder",
        action="store_true",
        default=False,
        help="Share decoder LoRA weights across speakers (default: False)"
    )
    
    # LoRA parameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha scaling factor (default: 16.0)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability (default: 0.05)"
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Which modules to apply LoRA to (default: q_proj v_proj)"
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
        default=100.0,
        help="Weight for semantic token loss (default: 100.0)"
    )
    parser.add_argument(
        "--acoustic-weight",
        type=float,
        default=1.0,
        help="Weight for acoustic token loss (default: 1.0)"
    )
    
    # Generation options
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate sample audio for each speaker after training"
    )
    parser.add_argument(
        "--sample-text",
        type=str,
        default="This is a test of multi-speaker voice adaptation using LoRA fine-tuning.",
        help="Text to use for sample generation"
    )
    
    # Model merging options
    parser.add_argument(
        "--merge-models",
        action="store_true",
        help="Merge shared and speaker-specific models after training"
    )
    parser.add_argument(
        "--shared-weight",
        type=float,
        default=0.5,
        help="Weight to give shared parameters in merged models (0.0-1.0)"
    )
    
    return parser.parse_args()

def load_speakers_config(config_path):
    """
    Load speaker configuration from JSON file.
    
    Expected format:
    {
        "speakers": [
            {
                "id": 0,
                "name": "Speaker1",
                "audio_dir": "/path/to/speaker1/audio",
                "transcript_dir": "/path/to/speaker1/transcripts",
                "alignment_dir": "/path/to/speaker1/alignments" (optional)
            },
            ...
        ]
    }
    """
    logger.info(f"Loading speakers configuration from {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"Speakers configuration file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate configuration
        if 'speakers' not in config or not isinstance(config['speakers'], list):
            logger.error("Invalid speakers configuration: missing or invalid 'speakers' list")
            return None
            
        for speaker in config['speakers']:
            # Check required fields
            for field in ['id', 'audio_dir', 'transcript_dir']:
                if field not in speaker:
                    logger.error(f"Invalid speaker configuration: missing required field '{field}'")
                    return None
                    
            # Check that directories exist
            for dir_field in ['audio_dir', 'transcript_dir']:
                if not os.path.exists(speaker[dir_field]):
                    logger.error(f"Directory not found for speaker {speaker['id']}: {speaker[dir_field]}")
                    return None
                    
            # Check optional alignment dir
            if 'alignment_dir' in speaker and not os.path.exists(speaker['alignment_dir']):
                logger.error(f"Alignment directory not found for speaker {speaker['id']}: {speaker['alignment_dir']}")
                return None
                
        logger.info(f"Loaded configuration for {len(config['speakers'])} speakers")
        return config
        
    except Exception as e:
        logger.error(f"Error loading speakers configuration: {e}")
        return None

def prepare_speaker_data(speaker_config, max_seq_len=2048):
    """
    Prepare training and validation data for a speaker.
    
    Args:
        speaker_config: Configuration for the speaker
        max_seq_len: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from csm.data import CSMDataProcessor, ContextualExampleGenerator
    from csm.training.data import MLXDataset
    
    logger.info(f"Preparing data for speaker {speaker_config['id']}")
    
    speaker_id = speaker_config['id']
    audio_dir = speaker_config['audio_dir']
    transcript_dir = speaker_config['transcript_dir']
    alignment_dir = speaker_config.get('alignment_dir')
    
    # Process audio files
    processor = CSMDataProcessor()
    all_examples = []
    
    # Get list of audio files
    import glob
    audio_files = glob.glob(os.path.join(audio_dir, "**/*.wav"), recursive=True)
    logger.info(f"Found {len(audio_files)} audio files for speaker {speaker_id}")
    
    # Process each file
    for audio_file in audio_files:
        # Get matching transcript file
        base_name = os.path.basename(audio_file).replace(".wav", "")
        transcript_file = os.path.join(transcript_dir, f"{base_name}.txt")
        
        # Skip if transcript doesn't exist
        if not os.path.exists(transcript_file):
            logger.warning(f"No transcript found for {audio_file}, skipping")
            continue
        
        # Get alignment file if available
        alignment_file = None
        if alignment_dir:
            alignment_path = os.path.join(alignment_dir, f"{base_name}.json")
            if os.path.exists(alignment_path):
                alignment_file = alignment_path
        
        # Process file
        try:
            # Call with or without alignment depending on availability
            if alignment_file:
                examples = processor.prepare_from_audio_file(
                    audio_file,
                    transcript_file,
                    alignment_file=alignment_file,
                    speaker_id=speaker_id
                )
            else:
                examples = processor.prepare_from_audio_file(
                    audio_file,
                    transcript_file,
                    speaker_id=speaker_id
                )
                
            all_examples.extend(examples)
            logger.info(f"Processed {audio_file} for speaker {speaker_id}")
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
    
    logger.info(f"Processed {len(all_examples)} examples for speaker {speaker_id}")
    
    # Add context
    generator = ContextualExampleGenerator(max_context_turns=2)
    contextual_examples = generator.create_contextual_examples(all_examples)
    
    # Split into train and validation sets
    import random
    random.shuffle(contextual_examples)
    val_split = 0.1
    val_size = int(len(contextual_examples) * val_split)
    train_examples = contextual_examples[val_size:]
    val_examples = contextual_examples[:val_size]
    
    logger.info(f"Split into {len(train_examples)} train and {len(val_examples)} validation examples for speaker {speaker_id}")
    
    # Create datasets
    train_dataset = MLXDataset(train_examples, max_seq_len=max_seq_len)
    val_dataset = MLXDataset(val_examples, max_seq_len=max_seq_len)
    
    return train_dataset, val_dataset

def main():
    """Main function for multi-speaker LoRA fine-tuning."""
    # Parse arguments
    args = parse_args()
    
    # Load speakers configuration
    config = load_speakers_config(args.speakers_config)
    if config is None:
        return 1
    
    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
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
        from csm.training.multi_speaker_lora import MultiSpeakerLoRATrainer
    except ImportError:
        logger.error("CSM package not found. Please install it with: pip install -e .")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "multi_speaker_training.log")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_path}")
    
    # Add file handler to logger
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Extract speaker IDs
    speaker_ids = [speaker['id'] for speaker in config['speakers']]
    
    # Initialize multi-speaker trainer
    logger.info("Initializing multi-speaker LoRA trainer")
    trainer = MultiSpeakerLoRATrainer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        speaker_ids=speaker_ids,
        log_file=log_path,
        learning_rate=args.learning_rate,
        semantic_weight=args.semantic_weight,
        acoustic_weight=args.acoustic_weight,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        share_backbone=args.share_backbone,
        share_decoder=args.share_decoder,
        target_modules=args.target_modules
    )
    
    # Prepare datasets for each speaker
    speaker_datasets = {}
    for speaker in config['speakers']:
        speaker_id = speaker['id']
        try:
            train_dataset, val_dataset = prepare_speaker_data(speaker)
            speaker_datasets[speaker_id] = (train_dataset, val_dataset)
        except Exception as e:
            logger.error(f"Error preparing data for speaker {speaker_id}: {e}")
            logger.error("Skipping this speaker")
    
    if not speaker_datasets:
        logger.error("No valid speaker datasets prepared. Aborting.")
        return 1
    
    # Train all speakers
    logger.info(f"Starting multi-speaker training for {len(speaker_datasets)} speakers")
    best_losses = trainer.train(
        speaker_datasets=speaker_datasets,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_every=100,
        save_every=500
    )
    
    # Report training results
    logger.info("Multi-speaker training completed")
    for speaker_id, loss in best_losses.items():
        logger.info(f"Speaker {speaker_id} best validation loss: {loss:.6f}")
    
    # Generate samples if requested
    if args.generate_samples:
        logger.info("Generating samples for each speaker")
        for speaker_id in speaker_datasets.keys():
            sample_path = os.path.join(args.output_dir, f"speaker_{speaker_id}", f"sample.wav")
            try:
                trainer.generate_sample(
                    text=args.sample_text,
                    speaker_id=speaker_id,
                    output_path=sample_path
                )
                logger.info(f"Generated sample for speaker {speaker_id} at {sample_path}")
            except Exception as e:
                logger.error(f"Error generating sample for speaker {speaker_id}: {e}")
    
    # Merge models if requested
    if args.merge_models:
        logger.info(f"Merging models with shared weight {args.shared_weight}")
        merged_models = trainer.merge_speaker_models(shared_weight=args.shared_weight)
        
        for speaker_id, model_path in merged_models.items():
            logger.info(f"Merged model for speaker {speaker_id} saved to {model_path}")
            
            # Generate sample with merged model
            if args.generate_samples:
                # Load merged model
                trainer.load_speaker_model(speaker_id, model_path)
                
                # Generate sample
                merged_sample_path = os.path.join(args.output_dir, f"speaker_{speaker_id}", f"merged_sample.wav")
                try:
                    trainer.generate_sample(
                        text=args.sample_text,
                        speaker_id=speaker_id,
                        output_path=merged_sample_path
                    )
                    logger.info(f"Generated sample with merged model for speaker {speaker_id} at {merged_sample_path}")
                except Exception as e:
                    logger.error(f"Error generating sample with merged model for speaker {speaker_id}: {e}")
    
    logger.info("Multi-speaker LoRA fine-tuning completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())