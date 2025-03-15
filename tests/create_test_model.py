#!/usr/bin/env python
"""
Create a test model for LoRA fine-tuning testing.

This script creates a small test model in safetensors format
that can be used to test the LoRA fine-tuning process end-to-end.
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available. This script requires MLX.")
    import sys
    sys.exit(1)

try:
    import safetensors.numpy
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("safetensors not available. Install with: pip install safetensors")
    import sys
    sys.exit(1)

def setup_logger():
    """Set up logger for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_tiny_model(
    output_path: str,
    hidden_size: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    text_vocab_size: int = 1000,
    audio_vocab_size: int = 200,
    audio_num_codebooks: int = 4,
    include_kv_cache: bool = False
):
    """
    Create a tiny test model in safetensors format.
    
    Args:
        output_path: Path to save the model
        hidden_size: Hidden size for the model
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        text_vocab_size: Size of text vocabulary
        audio_vocab_size: Size of audio vocabulary per codebook
        audio_num_codebooks: Number of audio codebooks
        include_kv_cache: Whether to include KV cache tensors
    """
    logger = setup_logger()
    logger.info(f"Creating tiny test model with hidden_size={hidden_size}, "
                f"num_layers={num_layers}, num_heads={num_heads}")
    
    # Calculate dimensions
    head_dim = hidden_size // num_heads
    
    # Create model parameters dictionary
    model_params = {}
    
    # Add model configuration
    model_params["config.model_type"] = np.array([1], dtype=np.int32)  # CSM model
    model_params["config.hidden_size"] = np.array([hidden_size], dtype=np.int32)
    model_params["config.num_layers"] = np.array([num_layers], dtype=np.int32)
    model_params["config.num_heads"] = np.array([num_heads], dtype=np.int32)
    model_params["config.head_dim"] = np.array([head_dim], dtype=np.int32)
    model_params["config.num_kv_heads"] = np.array([num_heads], dtype=np.int32)
    
    # Add text embedding
    model_params["backbone.text_embedding.weight"] = np.random.normal(
        0, 0.02, (text_vocab_size, hidden_size)
    ).astype(np.float32)
    
    # Add backbone parameters
    for layer_idx in range(num_layers):
        layer_prefix = f"backbone.layers.{layer_idx}"
        
        # Layer norm weights
        model_params[f"{layer_prefix}.input_layernorm_weight"] = np.ones(hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.input_layernorm_bias"] = np.zeros(hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.post_attention_layernorm_weight"] = np.ones(hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.post_attention_layernorm_bias"] = np.zeros(hidden_size).astype(np.float32)
        
        # Q/K/V projections - create all naming variants for compatibility
        # Primary names that match the model structure
        model_params[f"{layer_prefix}.attn.q_proj.weight"] = np.random.normal(
            0, 0.02, (hidden_size, hidden_size)
        ).astype(np.float32)
        model_params[f"{layer_prefix}.attn.k_proj.weight"] = np.random.normal(
            0, 0.02, (hidden_size, hidden_size)
        ).astype(np.float32)
        model_params[f"{layer_prefix}.attn.v_proj.weight"] = np.random.normal(
            0, 0.02, (hidden_size, hidden_size)
        ).astype(np.float32)
        
        # Alternative names for different model implementations
        model_params[f"{layer_prefix}.attn.q_proj_weight"] = model_params[f"{layer_prefix}.attn.q_proj.weight"].copy()
        model_params[f"{layer_prefix}.attn.k_proj_weight"] = model_params[f"{layer_prefix}.attn.k_proj.weight"].copy()
        model_params[f"{layer_prefix}.attn.v_proj_weight"] = model_params[f"{layer_prefix}.attn.v_proj.weight"].copy()
        
        # Output projection - both naming conventions
        model_params[f"{layer_prefix}.attn.o_proj.weight"] = np.random.normal(
            0, 0.02, (hidden_size, hidden_size)
        ).astype(np.float32)
        model_params[f"{layer_prefix}.attn.o_proj_weight"] = model_params[f"{layer_prefix}.attn.o_proj.weight"].copy()
        
        # MLP weights with both naming patterns
        # Primary names with dotted notation
        model_params[f"{layer_prefix}.mlp.gate_proj.weight"] = np.random.normal(
            0, 0.02, (hidden_size * 4, hidden_size)
        ).astype(np.float32)
        model_params[f"{layer_prefix}.mlp.up_proj.weight"] = np.random.normal(
            0, 0.02, (hidden_size * 4, hidden_size)
        ).astype(np.float32)
        model_params[f"{layer_prefix}.mlp.down_proj.weight"] = np.random.normal(
            0, 0.02, (hidden_size, hidden_size * 4)
        ).astype(np.float32)
        
        # Underscore notation for compatibility
        model_params[f"{layer_prefix}.mlp.gate_proj_weight"] = model_params[f"{layer_prefix}.mlp.gate_proj.weight"].copy()
        model_params[f"{layer_prefix}.mlp.up_proj_weight"] = model_params[f"{layer_prefix}.mlp.up_proj.weight"].copy()
        model_params[f"{layer_prefix}.mlp.down_proj_weight"] = model_params[f"{layer_prefix}.mlp.down_proj.weight"].copy()
        
        # Add bias terms with both naming conventions
        # QKV bias terms - dotted notation
        model_params[f"{layer_prefix}.attn.q_proj.bias"] = np.zeros(hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.attn.k_proj.bias"] = np.zeros(hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.attn.v_proj.bias"] = np.zeros(hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.attn.o_proj.bias"] = np.zeros(hidden_size).astype(np.float32)
        
        # QKV bias terms - underscore notation (for compatibility)
        model_params[f"{layer_prefix}.attn.q_proj_bias"] = model_params[f"{layer_prefix}.attn.q_proj.bias"].copy()
        model_params[f"{layer_prefix}.attn.k_proj_bias"] = model_params[f"{layer_prefix}.attn.k_proj.bias"].copy()
        model_params[f"{layer_prefix}.attn.v_proj_bias"] = model_params[f"{layer_prefix}.attn.v_proj.bias"].copy()
        model_params[f"{layer_prefix}.attn.o_proj_bias"] = model_params[f"{layer_prefix}.attn.o_proj.bias"].copy()
        
        # MLP bias terms - both notations
        model_params[f"{layer_prefix}.mlp.gate_proj.bias"] = np.zeros(hidden_size * 4).astype(np.float32)
        model_params[f"{layer_prefix}.mlp.up_proj.bias"] = np.zeros(hidden_size * 4).astype(np.float32)
        model_params[f"{layer_prefix}.mlp.down_proj.bias"] = np.zeros(hidden_size).astype(np.float32)
        
        # MLP bias terms - underscore notation (for compatibility)
        model_params[f"{layer_prefix}.mlp.gate_proj_bias"] = model_params[f"{layer_prefix}.mlp.gate_proj.bias"].copy()
        model_params[f"{layer_prefix}.mlp.up_proj_bias"] = model_params[f"{layer_prefix}.mlp.up_proj.bias"].copy()
        model_params[f"{layer_prefix}.mlp.down_proj_bias"] = model_params[f"{layer_prefix}.mlp.down_proj.bias"].copy()
        
        # Rotary embeddings
        if include_kv_cache:
            model_params[f"{layer_prefix}.attn.rotary_emb.cos_cached"] = np.cos(
                np.arange(0, 2048).reshape(-1, 1) * np.exp(
                    -np.arange(0, head_dim, 2) * (np.log(10000.0) / head_dim)
                )
            ).astype(np.float32)
            model_params[f"{layer_prefix}.attn.rotary_emb.sin_cached"] = np.sin(
                np.arange(0, 2048).reshape(-1, 1) * np.exp(
                    -np.arange(0, head_dim, 2) * (np.log(10000.0) / head_dim)
                )
            ).astype(np.float32)
    
    # Add backbone final layer norm
    model_params["backbone.final_layernorm_weight"] = np.ones(hidden_size).astype(np.float32)
    model_params["backbone.final_layernorm_bias"] = np.zeros(hidden_size).astype(np.float32)
    
    # Add decoder parameters - similar to backbone but with smaller size
    decoder_hidden_size = hidden_size // 2
    decoder_num_heads = num_heads // 2
    decoder_head_dim = decoder_hidden_size // decoder_num_heads
    
    # Audio embeddings
    model_params["decoder.audio_embedding.weight"] = np.random.normal(
        0, 0.02, (audio_vocab_size, decoder_hidden_size)
    ).astype(np.float32)
    
    for layer_idx in range(max(1, num_layers // 2)):
        layer_prefix = f"decoder.layers.{layer_idx}"
        
        # Layer norm weights
        model_params[f"{layer_prefix}.input_layernorm_weight"] = np.ones(decoder_hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.input_layernorm_bias"] = np.zeros(decoder_hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.post_attention_layernorm_weight"] = np.ones(decoder_hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.post_attention_layernorm_bias"] = np.zeros(decoder_hidden_size).astype(np.float32)
        
        # Q/K/V projections with both naming patterns
        # Primary names that match the model structure
        model_params[f"{layer_prefix}.attn.q_proj.weight"] = np.random.normal(
            0, 0.02, (decoder_hidden_size, decoder_hidden_size)
        ).astype(np.float32)
        model_params[f"{layer_prefix}.attn.k_proj.weight"] = np.random.normal(
            0, 0.02, (decoder_hidden_size, decoder_hidden_size)
        ).astype(np.float32)
        model_params[f"{layer_prefix}.attn.v_proj.weight"] = np.random.normal(
            0, 0.02, (decoder_hidden_size, decoder_hidden_size)
        ).astype(np.float32)
        
        # Alternative names for different model implementations
        model_params[f"{layer_prefix}.attn.q_proj_weight"] = model_params[f"{layer_prefix}.attn.q_proj.weight"].copy()
        model_params[f"{layer_prefix}.attn.k_proj_weight"] = model_params[f"{layer_prefix}.attn.k_proj.weight"].copy()
        model_params[f"{layer_prefix}.attn.v_proj_weight"] = model_params[f"{layer_prefix}.attn.v_proj.weight"].copy()
        
        # Output projection with both naming patterns
        model_params[f"{layer_prefix}.attn.o_proj.weight"] = np.random.normal(
            0, 0.02, (decoder_hidden_size, decoder_hidden_size)
        ).astype(np.float32)
        model_params[f"{layer_prefix}.attn.o_proj_weight"] = model_params[f"{layer_prefix}.attn.o_proj.weight"].copy()
        
        # MLP weights with both naming patterns
        # Primary names with dotted notation
        model_params[f"{layer_prefix}.mlp.gate_proj.weight"] = np.random.normal(
            0, 0.02, (decoder_hidden_size * 4, decoder_hidden_size)
        ).astype(np.float32)
        model_params[f"{layer_prefix}.mlp.up_proj.weight"] = np.random.normal(
            0, 0.02, (decoder_hidden_size * 4, decoder_hidden_size)
        ).astype(np.float32)
        model_params[f"{layer_prefix}.mlp.down_proj.weight"] = np.random.normal(
            0, 0.02, (decoder_hidden_size, decoder_hidden_size * 4)
        ).astype(np.float32)
        
        # Underscore notation for compatibility
        model_params[f"{layer_prefix}.mlp.gate_proj_weight"] = model_params[f"{layer_prefix}.mlp.gate_proj.weight"].copy()
        model_params[f"{layer_prefix}.mlp.up_proj_weight"] = model_params[f"{layer_prefix}.mlp.up_proj.weight"].copy()
        model_params[f"{layer_prefix}.mlp.down_proj_weight"] = model_params[f"{layer_prefix}.mlp.down_proj.weight"].copy()
        
        # Add bias terms with both naming conventions
        # QKV bias terms - dotted notation
        model_params[f"{layer_prefix}.attn.q_proj.bias"] = np.zeros(decoder_hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.attn.k_proj.bias"] = np.zeros(decoder_hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.attn.v_proj.bias"] = np.zeros(decoder_hidden_size).astype(np.float32)
        model_params[f"{layer_prefix}.attn.o_proj.bias"] = np.zeros(decoder_hidden_size).astype(np.float32)
        
        # QKV bias terms - underscore notation (for compatibility)
        model_params[f"{layer_prefix}.attn.q_proj_bias"] = model_params[f"{layer_prefix}.attn.q_proj.bias"].copy()
        model_params[f"{layer_prefix}.attn.k_proj_bias"] = model_params[f"{layer_prefix}.attn.k_proj.bias"].copy()
        model_params[f"{layer_prefix}.attn.v_proj_bias"] = model_params[f"{layer_prefix}.attn.v_proj.bias"].copy()
        model_params[f"{layer_prefix}.attn.o_proj_bias"] = model_params[f"{layer_prefix}.attn.o_proj.bias"].copy()
        
        # MLP bias terms - both notations
        model_params[f"{layer_prefix}.mlp.gate_proj.bias"] = np.zeros(decoder_hidden_size * 4).astype(np.float32)
        model_params[f"{layer_prefix}.mlp.up_proj.bias"] = np.zeros(decoder_hidden_size * 4).astype(np.float32)
        model_params[f"{layer_prefix}.mlp.down_proj.bias"] = np.zeros(decoder_hidden_size).astype(np.float32)
        
        # MLP bias terms - underscore notation (for compatibility)
        model_params[f"{layer_prefix}.mlp.gate_proj_bias"] = model_params[f"{layer_prefix}.mlp.gate_proj.bias"].copy()
        model_params[f"{layer_prefix}.mlp.up_proj_bias"] = model_params[f"{layer_prefix}.mlp.up_proj.bias"].copy()
        model_params[f"{layer_prefix}.mlp.down_proj_bias"] = model_params[f"{layer_prefix}.mlp.down_proj.bias"].copy()
        
        # Rotary embeddings
        if include_kv_cache:
            model_params[f"{layer_prefix}.attn.rotary_emb.cos_cached"] = np.cos(
                np.arange(0, 2048).reshape(-1, 1) * np.exp(
                    -np.arange(0, decoder_head_dim, 2) * (np.log(10000.0) / decoder_head_dim)
                )
            ).astype(np.float32)
            model_params[f"{layer_prefix}.attn.rotary_emb.sin_cached"] = np.sin(
                np.arange(0, 2048).reshape(-1, 1) * np.exp(
                    -np.arange(0, decoder_head_dim, 2) * (np.log(10000.0) / decoder_head_dim)
                )
            ).astype(np.float32)
    
    # Add decoder final layer norm
    model_params["decoder.final_layernorm_weight"] = np.ones(decoder_hidden_size).astype(np.float32)
    model_params["decoder.final_layernorm_bias"] = np.zeros(decoder_hidden_size).astype(np.float32)
    
    # Add output projections for each codebook
    for i in range(audio_num_codebooks):
        model_params[f"decoder.output_projection.{i}.weight"] = np.random.normal(
            0, 0.02, (audio_vocab_size, decoder_hidden_size)
        ).astype(np.float32)
        model_params[f"decoder.output_projection.{i}.bias"] = np.zeros(audio_vocab_size).astype(np.float32)
        
    # Add codebook heads for semantic loss
    for i in range(audio_num_codebooks):
        model_params[f"codebook{i}_head.weight"] = np.random.normal(
            0, 0.02, (audio_vocab_size, hidden_size)
        ).astype(np.float32)
        model_params[f"codebook{i}_head.bias"] = np.zeros(audio_vocab_size).astype(np.float32)
    
    # Add metadata
    model_params["config"] = np.array([
        hidden_size, num_layers, num_heads, text_vocab_size, 
        audio_vocab_size, audio_num_codebooks
    ]).astype(np.int32)
    
    # Save the model to safetensors format
    logger.info(f"Saving model with {len(model_params)} parameters to {output_path}")
    safetensors.numpy.save_file(model_params, output_path)
    logger.info(f"Model saved successfully")
    return output_path

def create_test_data(
    output_dir: str,
    num_samples: int = 5,
    sample_duration_sec: float = 2.0,
    sample_rate: int = 16000
):
    """
    Create test audio and transcript data.
    
    Args:
        output_dir: Directory to save test data
        num_samples: Number of test samples to create
        sample_duration_sec: Duration of each audio sample in seconds
        sample_rate: Sample rate for audio files
    """
    logger = setup_logger()
    logger.info(f"Creating {num_samples} test audio and transcript files in {output_dir}")
    
    # Create output directories
    audio_dir = os.path.join(output_dir, "audio")
    transcript_dir = os.path.join(output_dir, "transcripts")
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)
    
    # Create test data
    try:
        import torch
        import torchaudio
        HAS_TORCH_AUDIO = True
    except ImportError:
        logger.warning("torchaudio not available. Using placeholder audio.")
        HAS_TORCH_AUDIO = False
    
    test_texts = [
        "This is a test sentence for fine-tuning.",
        "The quick brown fox jumps over the lazy dog.",
        "Speech synthesis is the artificial production of human speech.",
        "LoRA fine-tuning enables efficient model adaptation.",
        "Machine learning models can be adapted to new domains.",
        "Neural networks learn to map inputs to outputs.",
        "Audio processing requires specialized techniques.",
        "The rain in Spain falls mainly on the plain.",
        "To be or not to be, that is the question.",
        "Four score and seven years ago our fathers brought forth."
    ]
    
    for i in range(num_samples):
        # Create audio file
        audio_path = os.path.join(audio_dir, f"sample_{i:03d}.wav")
        
        if HAS_TORCH_AUDIO:
            # Generate a sine wave with varying frequency
            sample_length = int(sample_duration_sec * sample_rate)
            t = torch.linspace(0, sample_duration_sec, sample_length)
            freq = 440 * (1 + 0.1 * (i % 5))  # Vary frequency slightly
            audio = 0.5 * torch.sin(2 * torch.pi * freq * t)
            
            # Add some noise
            noise = torch.randn_like(audio) * 0.01
            audio = audio + noise
            
            # Normalize
            audio = audio / torch.max(torch.abs(audio))
            
            # Save as WAV file
            torchaudio.save(
                audio_path,
                audio.unsqueeze(0),
                sample_rate,
                encoding="PCM_S",
                bits_per_sample=16
            )
        else:
            # Create empty file as placeholder
            with open(audio_path, "wb") as f:
                f.write(b"PLACEHOLDER AUDIO")
        
        # Create transcript file
        transcript_path = os.path.join(transcript_dir, f"sample_{i:03d}.txt")
        
        # Select text or generate random text
        if i < len(test_texts):
            text = test_texts[i]
        else:
            # Generate some random text
            words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
                     "speech", "audio", "model", "neural", "network", "fine-tuning", 
                     "adaptation", "transformer"]
            text = " ".join(words[j % len(words)] for j in range(10 + i % 5))
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text)
    
    logger.info(f"Created {num_samples} test samples in {output_dir}")
    return audio_dir, transcript_dir

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create test model and data for LoRA fine-tuning testing")
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tests/data",
        help="Directory to save test model and data"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Hidden size for test model"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of test audio samples to create"
    )
    parser.add_argument(
        "--include-kv-cache",
        action="store_true",
        help="Include KV cache tensors in model"
    )
    
    return parser.parse_args()

def main():
    """Main function to create test model and data."""
    args = parse_args()
    
    # Create test model
    model_path = os.path.join(args.output_dir, "test_model.safetensors")
    create_tiny_model(
        output_path=model_path,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        include_kv_cache=args.include_kv_cache
    )
    
    # Create test data
    audio_dir, transcript_dir = create_test_data(
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    print(f"Test model created at: {model_path}")
    print(f"Test audio files created at: {audio_dir}")
    print(f"Test transcript files created at: {transcript_dir}")
    print("\nTo test LoRA fine-tuning, run:")
    print(f"python -m csm.cli.finetune_lora --model-path {model_path} --output-dir ./lora_output \\\n"
          f"  --audio-dir {audio_dir} --transcript-dir {transcript_dir} \\\n"
          f"  --batch-size 2 --epochs 2 --lora-r 4")
    
    print("\nOr to test the HuggingFace example:")
    print(f"mkdir -p ./test_hf_data")
    print(f"cp -r {audio_dir}/* ./test_hf_data/")
    print(f"cp -r {transcript_dir}/* ./test_hf_data/")
    print(f"python examples/huggingface_lora_finetune.py --model-path {model_path} \\\n"
          f"  --dataset local --audio-dir {audio_dir} --transcript-dir {transcript_dir} \\\n"
          f"  --batch-size 2 --epochs 2 --lora-r 4 --output-dir ./lora_hf_output")

if __name__ == "__main__":
    main()