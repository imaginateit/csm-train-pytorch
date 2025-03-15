#!/usr/bin/env python
"""
CLI for multi-speaker fine-tuning of CSM models with LoRA using MLX.

This script provides a command-line interface for fine-tuning CSM models
for multiple speakers in a single training run using LoRA (Low-Rank Adaptation)
with MLX on Apple Silicon.
"""

import os
import sys
import time
import argparse
import logging
import json
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

class SmartFormatter(argparse.HelpFormatter):
    """Help formatter that respects newlines in help text."""
    
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        return argparse.HelpFormatter._split_lines(self, text, width)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-speaker fine-tuning for CSM models with LoRA using MLX",
        formatter_class=SmartFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained CSM model (safetensors or PyTorch format)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save fine-tuned models and logs"
    )
    parser.add_argument(
        "--speakers-config",
        type=str,
        required=True,
        help="Path to JSON file with speaker configurations"
    )
    
    # LoRA configuration
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
        default=0.0,
        help="LoRA dropout probability (default: 0.0)"
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=None,
        help=("R|LoRA target modules to adapt (default: ['q_proj', 'v_proj'])\n"
              "Options: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj")
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to apply LoRA to (default: all layers)"
    )
    parser.add_argument(
        "--lora-bias",
        action="store_true",
        help="Use bias in LoRA layers"
    )
    
    # Training parameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
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
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per speaker (default: 2)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs (default: 5)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=100,
        help="Validate every N steps (default: 100)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (default: 1.0)"
    )
    
    # Data processing
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--context-turns",
        type=int,
        default=2,
        help="Number of context turns to include (default: 2)"
    )
    
    # Output options
    parser.add_argument(
        "--save-mode",
        type=str,
        choices=["lora", "full", "both"],
        default="lora",
        help=("R|How to save the fine-tuned model (default: lora)\n"
              "  lora: Save only LoRA parameters\n"
              "  full: Save the full model with merged weights\n"
              "  both: Save both LoRA parameters and merged model")
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level (default: info)"
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate audio samples after training"
    )
    parser.add_argument(
        "--sample-prompt",
        type=str,
        default="This is a test of the fine-tuned voice model.",
        help="Prompt for sample generation"
    )
    
    # Additional options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--sample-speakers",
        type=int,
        default=None,
        help="Train on a sample of N speakers (for testing)"
    )
    
    args = parser.parse_args()
    return args


def setup_logger(name, log_file=None, level_name="info"):
    """Set up logger with specified level."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    level = level_map.get(level_name.lower(), logging.INFO)
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_speaker_configs(config_path: str, sample_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load speaker configurations from JSON file.
    
    Expected format:
    [
        {
            "name": "speaker1",
            "speaker_id": 1,
            "audio_dir": "/path/to/speaker1/audio",
            "transcript_dir": "/path/to/speaker1/transcripts",
            "alignment_dir": "/path/to/speaker1/alignments",  # optional
            "lora_r": 8,  # optional, override global setting
            "epochs": 5,  # optional, override global setting
            "learning_rate": 1e-4  # optional, override global setting
        },
        ...
    ]
    
    Args:
        config_path: Path to JSON configuration file
        sample_n: Optional number of speakers to sample
        
    Returns:
        List of speaker configurations
    """
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    # Validate configs
    for i, config in enumerate(configs):
        # Check required fields
        required_fields = ["name", "speaker_id", "audio_dir", "transcript_dir"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Speaker config {i} missing required field: {field}")
        
        # Check that directories exist
        for dir_field in ["audio_dir", "transcript_dir"]:
            if not os.path.exists(config[dir_field]):
                raise ValueError(f"Directory does not exist: {config[dir_field]}")
        
        if "alignment_dir" in config and config["alignment_dir"] and not os.path.exists(config["alignment_dir"]):
            raise ValueError(f"Alignment directory does not exist: {config['alignment_dir']}")
    
    # Sample if requested
    if sample_n is not None and sample_n < len(configs):
        import random
        configs = random.sample(configs, sample_n)
    
    return configs


def prepare_speaker_data(
    speaker_config: Dict[str, Any],
    global_args: argparse.Namespace,
    logger: logging.Logger
) -> Tuple[Any, Any]:
    """
    Prepare training and validation data for a single speaker.
    
    Args:
        speaker_config: Speaker configuration
        global_args: Global arguments
        logger: Logger instance
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from csm.cli.finetune_lora import prepare_data
    
    # Override global settings with speaker-specific settings if present
    val_split = speaker_config.get("val_split", global_args.val_split)
    max_seq_len = speaker_config.get("max_seq_len", global_args.max_seq_len)
    context_turns = speaker_config.get("context_turns", global_args.context_turns)
    batch_size = speaker_config.get("batch_size", global_args.batch_size)
    
    # Get paths
    audio_dir = speaker_config["audio_dir"]
    transcript_dir = speaker_config["transcript_dir"]
    alignment_dir = speaker_config.get("alignment_dir")
    speaker_id = speaker_config["speaker_id"]
    
    logger.info(f"Preparing data for speaker {speaker_config['name']} (ID: {speaker_id})")
    
    # Prepare data
    train_dataset, val_dataset = prepare_data(
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        alignment_dir=alignment_dir,
        speaker_id=speaker_id,
        val_split=val_split,
        max_seq_len=max_seq_len,
        context_turns=context_turns,
        batch_size=batch_size
    )
    
    logger.info(f"Speaker {speaker_config['name']}: {len(train_dataset)} training batches, "
                f"{len(val_dataset)} validation batches")
    
    return train_dataset, val_dataset


def finetune_speaker(
    speaker_config: Dict[str, Any],
    global_args: argparse.Namespace,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Fine-tune a model for a single speaker.
    
    Args:
        speaker_config: Speaker configuration
        global_args: Global arguments
        logger: Logger instance
        
    Returns:
        Dictionary with training results
    """
    # Create speaker-specific output directory
    speaker_name = speaker_config["name"]
    speaker_id = speaker_config["speaker_id"]
    speaker_output_dir = os.path.join(global_args.output_dir, f"speaker_{speaker_name}")
    os.makedirs(speaker_output_dir, exist_ok=True)
    
    # Set up speaker-specific log file
    log_file = os.path.join(speaker_output_dir, "training.log")
    speaker_logger = setup_logger(
        f"speaker_{speaker_name}",
        log_file=log_file,
        level_name=global_args.log_level
    )
    
    # Log speaker config
    speaker_logger.info(f"Starting fine-tuning for speaker: {speaker_name} (ID: {speaker_id})")
    speaker_logger.info("Speaker configuration:")
    for key, value in speaker_config.items():
        speaker_logger.info(f"  {key}: {value}")
    
    try:
        # Prepare data
        speaker_logger.info("Preparing datasets")
        train_dataset, val_dataset = prepare_speaker_data(
            speaker_config=speaker_config,
            global_args=global_args,
            logger=speaker_logger
        )
        
        # Get speaker-specific overrides or use global values
        lora_r = speaker_config.get("lora_r", global_args.lora_r)
        lora_alpha = speaker_config.get("lora_alpha", global_args.lora_alpha)
        lora_dropout = speaker_config.get("lora_dropout", global_args.lora_dropout)
        learning_rate = speaker_config.get("learning_rate", global_args.learning_rate)
        epochs = speaker_config.get("epochs", global_args.epochs)
        
        # Override target modules if specified
        target_modules = global_args.target_modules
        if "target_modules" in speaker_config:
            target_modules = speaker_config["target_modules"]
        
        # Initialize LoRA trainer
        speaker_logger.info("Initializing LoRA trainer")
        from csm.training.lora_trainer import CSMLoRATrainer
        
        trainer = CSMLoRATrainer(
            model_path=global_args.model_path,
            output_dir=speaker_output_dir,
            log_file=log_file,
            learning_rate=learning_rate,
            semantic_weight=global_args.semantic_weight,
            acoustic_weight=global_args.acoustic_weight,
            weight_decay=global_args.weight_decay,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            target_layers=global_args.target_layers,
            lora_use_bias=global_args.lora_bias
        )
        
        # Prepare optimizer
        speaker_logger.info("Preparing optimizer")
        trainer.prepare_optimizer()
        
        # Fine-tune the model
        speaker_logger.info("Starting fine-tuning")
        start_time = time.time()
        best_loss = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=speaker_config.get("batch_size", global_args.batch_size),
            epochs=epochs,
            val_every=global_args.val_every,
            save_every=global_args.save_every,
            max_grad_norm=global_args.max_grad_norm
        )
        training_time = time.time() - start_time
        
        speaker_logger.info(f"Fine-tuning completed in {training_time:.2f} seconds with "
                           f"best validation loss: {best_loss:.6f}")
        
        # Save the fine-tuned model
        save_mode = speaker_config.get("save_mode", global_args.save_mode)
        speaker_logger.info(f"Saving fine-tuned model in {save_mode} mode")
        save_path = os.path.join(speaker_output_dir, f"{speaker_name}_model.safetensors")
        trainer.save_model(save_path, save_mode=save_mode)
        
        # Generate sample if requested
        if global_args.generate_samples:
            speaker_logger.info("Generating sample audio")
            sample_path = os.path.join(speaker_output_dir, f"{speaker_name}_sample.wav")
            
            # Get custom prompt if specified
            sample_prompt = speaker_config.get("sample_prompt", global_args.sample_prompt)
            
            try:
                trainer.generate_sample(
                    text=sample_prompt,
                    speaker_id=speaker_id,
                    output_path=sample_path
                )
                speaker_logger.info(f"Sample audio saved to {sample_path}")
            except Exception as e:
                speaker_logger.error(f"Error generating sample: {e}")
        
        # Return results
        results = {
            "speaker_name": speaker_name,
            "speaker_id": speaker_id,
            "best_loss": float(best_loss),
            "training_time": training_time,
            "epochs": epochs,
            "lora_r": lora_r,
            "model_path": save_path,
            "success": True
        }
        
        speaker_logger.info("Fine-tuning completed successfully")
        return results
    
    except Exception as e:
        speaker_logger.error(f"Error during fine-tuning: {e}")
        import traceback
        speaker_logger.error(traceback.format_exc())
        
        # Return error results
        return {
            "speaker_name": speaker_name,
            "speaker_id": speaker_id,
            "error": str(e),
            "success": False
        }


def main():
    """Main function for the multi-speaker LoRA fine-tuning CLI."""
    # Check for MLX
    if not HAS_MLX:
        print("Error: MLX is required but not installed.")
        print("Install it with: pip install mlx")
        return 1
    
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logger("multi_speaker_finetuning", level_name=args.log_level)
    logger.info(f"Starting CSM multi-speaker LoRA fine-tuning with MLX")
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up file logger
    log_file = os.path.join(args.output_dir, "multi_speaker_training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log all arguments
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check if running on Apple Silicon
    try:
        import platform
        is_apple_silicon = platform.processor() == 'arm'
        logger.info(f"Running on Apple Silicon: {is_apple_silicon}")
    except:
        logger.warning("Could not determine if running on Apple Silicon")
    
    try:
        # Load speaker configurations
        logger.info(f"Loading speaker configurations from {args.speakers_config}")
        speaker_configs = load_speaker_configs(args.speakers_config, args.sample_speakers)
        logger.info(f"Loaded {len(speaker_configs)} speaker configurations")
        
        # Process each speaker
        results = []
        for i, speaker_config in enumerate(speaker_configs):
            logger.info(f"Processing speaker {i+1}/{len(speaker_configs)}: {speaker_config['name']}")
            speaker_result = finetune_speaker(
                speaker_config=speaker_config,
                global_args=args,
                logger=logger
            )
            results.append(speaker_result)
            
            # Log progress
            successes = sum(1 for r in results if r.get("success", False))
            logger.info(f"Completed {i+1}/{len(speaker_configs)} speakers ({successes} successful)")
        
        # Save overall results
        overall_results = {
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_speakers": len(speaker_configs),
            "successful_speakers": sum(1 for r in results if r.get("success", False)),
            "failed_speakers": sum(1 for r in results if not r.get("success", False)),
            "speaker_results": results,
            "args": vars(args)
        }
        
        results_path = os.path.join(args.output_dir, "multi_speaker_results.json")
        with open(results_path, "w") as f:
            json.dump(overall_results, f, indent=2)
        
        logger.info(f"Multi-speaker fine-tuning completed: {overall_results['successful_speakers']} successes, "
                   f"{overall_results['failed_speakers']} failures")
        logger.info(f"Results saved to {results_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during multi-speaker fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())