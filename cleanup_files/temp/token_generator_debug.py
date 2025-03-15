#!/usr/bin/env python
"""
Debug script to compare token generation between PyTorch and MLX implementations.
"""

import os
import sys
import torch
import numpy as np
import torchaudio
from collections import Counter
import matplotlib.pyplot as plt

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import CSM modules
from csm.generator import Generator
from csm.cli.mlx_components.generator import MLXGenerator
from csm.cli.mlx_components.config import MLXConfig

def compare_token_histograms(pt_tokens, mlx_tokens, save_path="token_histograms.png"):
    """Compare and visualize token distributions between PyTorch and MLX."""
    # Flatten tokens
    pt_flat = pt_tokens.flatten().tolist()
    mlx_flat = mlx_tokens.flatten().tolist()
    
    # Count token occurrences
    pt_counter = Counter(pt_flat)
    mlx_counter = Counter(mlx_flat)
    
    # Get top tokens
    pt_most_common = pt_counter.most_common(20)
    mlx_most_common = mlx_counter.most_common(20)
    
    # Prepare plot data
    pt_tokens_vals = [t[0] for t in pt_most_common]
    pt_tokens_counts = [t[1] for t in pt_most_common]
    mlx_tokens_vals = [t[0] for t in mlx_most_common]
    mlx_tokens_counts = [t[1] for t in mlx_most_common]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PyTorch plot
    ax1.bar(range(len(pt_tokens_vals)), pt_tokens_counts, tick_label=pt_tokens_vals)
    ax1.set_title("PyTorch Token Distribution")
    ax1.set_xlabel("Token Value")
    ax1.set_ylabel("Count")
    
    # MLX plot
    ax2.bar(range(len(mlx_tokens_vals)), mlx_tokens_counts, tick_label=mlx_tokens_vals)
    ax2.set_title("MLX Token Distribution")
    ax2.set_xlabel("Token Value")
    ax2.set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved token histograms to {save_path}")
    
    # Print statistics
    print("\nPyTorch Token Statistics:")
    print(f"Total tokens: {len(pt_flat)}")
    print(f"Unique tokens: {len(pt_counter)}")
    print(f"Most common tokens: {pt_most_common[:10]}")
    print(f"Token range: {min(pt_flat)} to {max(pt_flat)}")
    
    print("\nMLX Token Statistics:")
    print(f"Total tokens: {len(mlx_flat)}")
    print(f"Unique tokens: {len(mlx_counter)}")
    print(f"Most common tokens: {mlx_most_common[:10]}")
    print(f"Token range: {min(mlx_flat)} to {max(mlx_flat)}")
    
    # Check for problematic tokens
    pt_problematic = sum(1 for t in pt_flat if 1 <= t <= 31)
    mlx_problematic = sum(1 for t in mlx_flat if 1 <= t <= 31)
    
    print(f"\nPyTorch problematic tokens (1-31): {pt_problematic} ({pt_problematic/len(pt_flat)*100:.2f}%)")
    print(f"MLX problematic tokens (1-31): {mlx_problematic} ({mlx_problematic/len(mlx_flat)*100:.2f}%)")
    
    # Token overlap analysis
    pt_set = set(pt_flat)
    mlx_set = set(mlx_flat)
    common_tokens = pt_set.intersection(mlx_set)
    
    print(f"\nToken overlap: {len(common_tokens)} tokens")
    print(f"Percentage of PyTorch tokens in MLX: {len(common_tokens)/len(pt_set)*100:.2f}%")
    print(f"Percentage of MLX tokens in PyTorch: {len(common_tokens)/len(mlx_set)*100:.2f}%")

def generate_and_compare(text="Hello, this is a test of the CSM speech model.", voice="warm"):
    """Generate and compare token sequences."""
    # Initialize models
    print("Loading PyTorch model...")
    from csm.generator import load_csm_1b
    pt_model = load_csm_1b(device="cpu")  # Use CPU since we're debugging
    print("Model loaded!")
    
    print("Loading MLX model...")
    mlx_config = MLXConfig()
    mlx_config.debug = True
    mlx_model = MLXGenerator(model=pt_model._model, tokenizer=pt_model._tokenizer, debug=True)
    print("MLX model loaded!")
    
    # Set speaker
    speaker = pt_model.get_speaker_id(voice)
    print(f"Using voice '{voice}' (speaker ID: {speaker})")
    
    # Generate tokens with PyTorch
    print("\nGenerating with PyTorch...")
    pt_audio = pt_model.generate(
        text=text,
        speaker=speaker,
        context=[],
        max_audio_length_ms=5000,  # 5 seconds
        temperature=1.0,
    )
    print(f"PyTorch audio shape: {pt_audio.shape}")
    
    # Save the PyTorch audio
    torchaudio.save("debug_pytorch.wav", pt_audio.unsqueeze(0), 24000)
    print("Saved PyTorch audio to debug_pytorch.wav")
    
    # Generate with MLX
    print("\nGenerating with MLX...")
    mlx_audio = mlx_model.generate_speech(
        text=text,
        speaker=speaker,
        context=[],
        max_audio_length_ms=5000,  # 5 seconds
        temperature=1.0,
    )
    print(f"MLX audio shape: {mlx_audio.shape}")
    
    # Save the MLX audio
    torchaudio.save("debug_mlx.wav", mlx_audio.unsqueeze(0), 24000)
    print("Saved MLX audio to debug_mlx.wav")
    
    # Retrieve tokens for analysis
    # For PyTorch this is already stored in the model
    if hasattr(pt_model, '_last_tokens') and pt_model._last_tokens is not None:
        pt_tokens = pt_model._last_tokens
    else:
        print("Warning: Could not retrieve PyTorch tokens")
        pt_tokens = torch.zeros((1, 32, 100))
    
    # For MLX, check the model attribute
    if hasattr(mlx_model, '_last_tokens') and mlx_model._last_tokens is not None:
        mlx_tokens = mlx_model._last_tokens
    else:
        print("Warning: Could not retrieve MLX tokens")
        mlx_tokens = torch.zeros((1, 32, 100))
    
    # Save tokens for analysis
    torch.save(pt_tokens, "debug_pt_tokens.pt")
    torch.save(mlx_tokens, "debug_mlx_tokens.pt")
    print("Saved token tensors to debug_pt_tokens.pt and debug_mlx_tokens.pt")
    
    # Compare the token distributions
    compare_token_histograms(pt_tokens, mlx_tokens)
    
    return pt_tokens, mlx_tokens, pt_audio, mlx_audio

def swap_decode_test(pt_model, mlx_model, pt_tokens, mlx_tokens):
    """Cross-decode tokens - decode PT tokens with MLX and vice versa."""
    print("\n=== Cross-Decoding Experiment ===")
    
    # Decode MLX tokens with PyTorch decoder
    print("Decoding MLX tokens with PyTorch decoder...")
    try:
        pt_decoder = pt_model._audio_tokenizer
        mlx_with_pt_decoder = pt_decoder.decode(mlx_tokens)
        torchaudio.save("mlx_tokens_pt_decoder.wav", mlx_with_pt_decoder.unsqueeze(0), 24000)
        print("Saved cross-decoded audio to mlx_tokens_pt_decoder.wav")
    except Exception as e:
        print(f"Error decoding MLX tokens with PyTorch decoder: {e}")
    
    # Decode PyTorch tokens with MLX decoder
    print("Decoding PyTorch tokens with MLX decoder...")
    try:
        mlx_decoder = mlx_model._audio_tokenizer
        pt_with_mlx_decoder = mlx_decoder.decode(pt_tokens)
        torchaudio.save("pt_tokens_mlx_decoder.wav", pt_with_mlx_decoder.unsqueeze(0), 24000)
        print("Saved cross-decoded audio to pt_tokens_mlx_decoder.wav")
    except Exception as e:
        print(f"Error decoding PyTorch tokens with MLX decoder: {e}")

if __name__ == "__main__":
    # Run the main comparison
    pt_tokens, mlx_tokens, pt_audio, mlx_audio = generate_and_compare()
    
    # Load PyTorch model for cross-decoding test
    from csm.generator import load_csm_1b
    pt_model = load_csm_1b(device="cpu")
    mlx_config = MLXConfig()
    mlx_model = MLXGenerator(model=pt_model._model, tokenizer=pt_model._tokenizer, debug=True)
    
    # Run cross-decoding test
    swap_decode_test(pt_model, mlx_model, pt_tokens, mlx_tokens)