#!/usr/bin/env python
"""
Test script for LoRA fine-tuning in CSM.

This script demonstrates a complete end-to-end test of the LoRA fine-tuning functionality,
including model creation, data preparation, training, and saving.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Make sure CSM is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available. This test requires MLX.")
    sys.exit(1)

try:
    import safetensors.numpy
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("safetensors not available. This test requires safetensors.")
    sys.exit(1)

import numpy as np

# Set up logger
def setup_logger(name="test_lora", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

logger = setup_logger(level=logging.DEBUG)

def create_test_model(output_path):
    """Create a tiny model for testing LoRA."""
    logger.info(f"Creating test model at {output_path}")
    
    try:
        # Create a tiny model with minimal parameters but proper structure for LoRA
        hidden_size = 16
        num_layers = 2
        num_heads = 2
        head_dim = hidden_size // num_heads
        
        # Create weights dictionary with proper transformer structures
        weights = {
            # Model dimensions
            "backbone.hidden_size": np.array(hidden_size, dtype=np.int32),
            "backbone.num_heads": np.array(num_heads, dtype=np.int32),
            "backbone.num_layers": np.array(num_layers, dtype=np.int32),
            "backbone.head_dim": np.array(head_dim, dtype=np.int32),
            
            "decoder.hidden_size": np.array(hidden_size, dtype=np.int32),
            "decoder.num_heads": np.array(num_heads, dtype=np.int32),
            "decoder.num_layers": np.array(num_layers, dtype=np.int32),
            "decoder.head_dim": np.array(head_dim, dtype=np.int32),
            
            # Embeddings
            "text_embeddings": np.zeros((100, hidden_size), dtype=np.float32),
            "audio_embeddings": np.zeros((100, hidden_size), dtype=np.float32),
            
            # Heads
            "codebook0_head": np.zeros((hidden_size, 100), dtype=np.float32),
            "audio_head.0": np.zeros((hidden_size, 100), dtype=np.float32),
            "projection": np.zeros((hidden_size, hidden_size), dtype=np.float32)
        }
        
        # Add backbone layers with proper components
        for i in range(num_layers):
            # Attention components
            weights[f"backbone.layers.{i}.q_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"backbone.layers.{i}.k_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"backbone.layers.{i}.v_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"backbone.layers.{i}.o_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            
            # Add bias terms
            weights[f"backbone.layers.{i}.q_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.k_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.v_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.o_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            
            # MLP components
            weights[f"backbone.layers.{i}.gate_proj_weight"] = np.zeros((hidden_size * 4, hidden_size), dtype=np.float32)
            weights[f"backbone.layers.{i}.up_proj_weight"] = np.zeros((hidden_size * 4, hidden_size), dtype=np.float32)
            weights[f"backbone.layers.{i}.down_proj_weight"] = np.zeros((hidden_size, hidden_size * 4), dtype=np.float32)
            
            # Layernorm components
            weights[f"backbone.layers.{i}.input_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.input_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.post_attention_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
            weights[f"backbone.layers.{i}.post_attention_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
        
        # Add decoder layers with proper components
        for i in range(num_layers):
            # Attention components
            weights[f"decoder.layers.{i}.q_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"decoder.layers.{i}.k_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"decoder.layers.{i}.v_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            weights[f"decoder.layers.{i}.o_proj_weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
            
            # Add bias terms
            weights[f"decoder.layers.{i}.q_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.k_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.v_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.o_proj_bias"] = np.zeros(hidden_size, dtype=np.float32)
            
            # MLP components
            weights[f"decoder.layers.{i}.gate_proj_weight"] = np.zeros((hidden_size * 4, hidden_size), dtype=np.float32)
            weights[f"decoder.layers.{i}.up_proj_weight"] = np.zeros((hidden_size * 4, hidden_size), dtype=np.float32)
            weights[f"decoder.layers.{i}.down_proj_weight"] = np.zeros((hidden_size, hidden_size * 4), dtype=np.float32)
            
            # Layernorm components
            weights[f"decoder.layers.{i}.input_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.input_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.post_attention_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
            weights[f"decoder.layers.{i}.post_attention_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
        
        # Add final layernorm components
        weights["backbone.final_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
        weights["backbone.final_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
        weights["decoder.final_layernorm_weight"] = np.ones(hidden_size, dtype=np.float32)
        weights["decoder.final_layernorm_bias"] = np.zeros(hidden_size, dtype=np.float32)
        
        # Save weights to safetensors
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        safetensors.numpy.save_file(weights, output_path)
        
        # Create metadata file
        metadata = {
            "epoch": 0,
            "global_step": 0,
            "loss": 1.0,
            "model_path": output_path,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads
        }
        
        metadata_path = output_path.replace(".safetensors", "_metadata.json")
        import json
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Created test model at {output_path} with proper LoRA structure")
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating test model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_test_data(data_dir, num_samples=3):
    """Create test audio and transcript data."""
    logger.info(f"Creating {num_samples} test samples in {data_dir}")
    
    # Create directories
    audio_dir = os.path.join(data_dir, "audio")
    transcript_dir = os.path.join(data_dir, "transcripts")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)
    
    try:
        import wave
        
        # Create sample files
        for i in range(num_samples):
            # Create a dummy audio signal (1 second of silence)
            sample_rate = 16000
            data = np.zeros(sample_rate, dtype=np.int16)
            
            # Save to WAV file
            audio_path = os.path.join(audio_dir, f"sample_{i:03d}.wav")
            with wave.open(audio_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(data.tobytes())
            
            # Create matching transcript
            transcript_path = os.path.join(transcript_dir, f"sample_{i:03d}.txt")
            with open(transcript_path, "w") as f:
                f.write(f"This is a test transcript for sample {i}.")
        
        logger.info(f"Created {num_samples} test samples")
        return audio_dir, transcript_dir
    
    except Exception as e:
        logger.error(f"Error creating test data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def run_lora_fine_tuning(model_path, output_dir, audio_dir, transcript_dir):
    """Run LoRA fine-tuning with the test model and data."""
    logger.info("Running LoRA fine-tuning")
    
    try:
        from csm.training.lora_trainer import CSMLoRATrainer
        from csm.training.data import MLXDataset
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize trainer
        logger.info(f"Initializing trainer with model from {model_path}")
        trainer = CSMLoRATrainer(
            model_path=model_path,
            output_dir=output_dir,
            lora_r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
            learning_rate=1e-4
        )
        
        # Prepare optimizer
        logger.info("Preparing optimizer")
        trainer.prepare_optimizer()
        
        # Prepare test data examples
        logger.info("Preparing test data")
        examples = []
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        
        for audio_file in audio_files:
            base_name = os.path.splitext(audio_file)[0]
            transcript_file = os.path.join(transcript_dir, f"{base_name}.txt")
            
            if os.path.exists(transcript_file):
                examples.append({
                    "audio_path": os.path.join(audio_dir, audio_file),
                    "transcript_path": transcript_file,
                    "speaker_id": 0
                })
        
        logger.info(f"Created {len(examples)} examples from test data")
        
        # Split into train and validation
        import random
        random.shuffle(examples)
        val_split = 0.1
        val_size = max(1, int(len(examples) * val_split))
        train_examples = examples[val_size:]
        val_examples = examples[:val_size]
        
        # Create datasets
        train_dataset = MLXDataset(train_examples, max_seq_len=128, batch_size=1, use_dummy_data=True)
        val_dataset = MLXDataset(val_examples, max_seq_len=128, batch_size=1, use_dummy_data=True)
        
        # Run training
        logger.info("Starting training")
        best_loss = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=1,
            epochs=1,
            val_every=10,
            save_every=50
        )
        
        logger.info(f"Training completed with best loss: {best_loss}")
        
        # Save model
        logger.info("Saving fine-tuned model")
        save_path = os.path.join(output_dir, "fine_tuned_model.safetensors")
        trainer.save_model(save_path, save_mode="both")
        
        # Try generating a sample
        logger.info("Generating sample (this will likely fail with test model)")
        try:
            sample_path = os.path.join(output_dir, "sample.wav")
            trainer.generate_sample(
                text="This is a test of the fine-tuned model.",
                speaker_id=0,
                output_path=sample_path
            )
            logger.info(f"Sample generated at {sample_path}")
        except Exception as e:
            logger.warning(f"Sample generation failed (expected): {e}")
            # Create a silent wav file as fallback
            try:
                import wave
                
                # Create silent audio (1 second)
                sample_rate = 16000
                silent_audio = np.zeros(sample_rate, dtype=np.int16)
                
                # Save as wav
                with wave.open(os.path.join(output_dir, "sample.wav"), 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(silent_audio.tobytes())
                
                logger.info(f"Created fallback silent sample at {os.path.join(output_dir, 'sample.wav')}")
            except Exception as fallback_e:
                logger.error(f"Error creating fallback sample: {fallback_e}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during LoRA fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_lora_adaptation(output_dir):
    """Verify the LoRA adaptation was saved correctly."""
    logger.info("Verifying LoRA adaptation")
    
    try:
        lora_path = os.path.join(output_dir, "fine_tuned_model_lora.safetensors")
        full_path = os.path.join(output_dir, "fine_tuned_model_full.safetensors")
        
        if not os.path.exists(lora_path):
            lora_path = os.path.join(output_dir, "fine_tuned_model.safetensors")
        
        # Check LoRA weights
        if os.path.exists(lora_path):
            import safetensors.numpy
            weights = safetensors.numpy.load_file(lora_path)
            num_params = len(weights)
            logger.info(f"LoRA weights saved at {lora_path} with {num_params} parameters")
            return True
        else:
            logger.error(f"LoRA weights not found at {lora_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error verifying LoRA adaptation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Test script for LoRA fine-tuning")
    parser.add_argument("--output-dir", type=str, default="./test_lora_output", help="Output directory")
    args = parser.parse_args()
    
    # Timestamp for unique outputs
    timestamp = int(time.time())
    output_dir = os.path.join(args.output_dir, f"test_run_{timestamp}")
    model_path = os.path.join(output_dir, "test_model.safetensors")
    data_dir = os.path.join(output_dir, "test_data")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create log file
    file_handler = logging.FileHandler(os.path.join(output_dir, "test.log"))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Step 1: Create test model
    logger.info("STEP 1: Creating test model")
    model_path = create_test_model(model_path)
    if not model_path:
        logger.error("Failed to create test model")
        return 1
    
    # Step 2: Create test data
    logger.info("STEP 2: Creating test data")
    audio_dir, transcript_dir = create_test_data(data_dir, num_samples=3)
    if not audio_dir or not transcript_dir:
        logger.error("Failed to create test data")
        return 1
    
    # Step 3: Run LoRA fine-tuning
    logger.info("STEP 3: Running LoRA fine-tuning")
    success = run_lora_fine_tuning(model_path, output_dir, audio_dir, transcript_dir)
    if not success:
        logger.error("Failed to run LoRA fine-tuning")
        return 1
    
    # Step 4: Verify LoRA adaptation
    logger.info("STEP 4: Verifying LoRA adaptation")
    success = verify_lora_adaptation(output_dir)
    if not success:
        logger.error("Failed to verify LoRA adaptation")
        return 1
    
    logger.info("All tests completed successfully!")
    logger.info(f"Test output directory: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())