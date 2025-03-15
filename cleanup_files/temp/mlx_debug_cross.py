#!/usr/bin/env python
"""
Cross-testing tool that swaps token generation and decoding between PyTorch and MLX.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path to import csm modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from csm.generator import Generator
from csm.cli.mlx_components.config import MLXConfig
from csm.cli.mlx_wrapper import MLXWrapper

# Check if MLX is available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

def cross_test_tokens_and_decode(text: str = "Hello, this is a test of the speech model.", voice: str = "warm"):
    """
    Generate tokens with PyTorch and decode with MLX, then vice versa.
    This isolates whether the issue is in token generation or audio decoding.
    """
    # Load models
    print("Loading PyTorch model...")
    from csm.generator import load_csm_1b
    pt_model = load_csm_1b(device="cpu")
    
    if not MLX_AVAILABLE:
        print("MLX is not available. Cannot perform cross-testing.")
        return
    
    # Load MLX model
    print("Loading MLX model...")
    mlx_config = MLXConfig()
    mlx_config.debug = True
    
    # Create MLX wrapper for the PyTorch model
    mlx_wrapper = MLXWrapper(pt_model, mlx_config)
    
    # Get speaker ID
    speaker_id = pt_model.get_speaker_id(voice)
    print(f"Using voice: {voice} (speaker ID: {speaker_id})")
    
    # Generate tokens with PyTorch
    print("\n=== GENERATING TOKENS WITH PYTORCH ===")
    tokens_pt = generate_pytorch_tokens(pt_model, text, speaker_id)
    
    # Generate tokens with MLX
    print("\n=== GENERATING TOKENS WITH MLX ===")
    tokens_mlx = generate_mlx_tokens(mlx_wrapper, text, speaker_id)
    
    # Create dummy tokens with known safe values
    print("\n=== CREATING DUMMY TOKENS WITH KNOWN SAFE VALUES ===")
    dummy_tokens = create_dummy_tokens()
    
    # Cross-decode tokens with PyTorch
    print("\n=== DECODING WITH PYTORCH ===")
    audio_pt_from_pt = decode_with_pytorch(pt_model, tokens_pt, "audio_pt_from_pt.wav")
    audio_pt_from_mlx = decode_with_pytorch(pt_model, tokens_mlx, "audio_pt_from_mlx.wav")
    audio_pt_from_dummy = decode_with_pytorch(pt_model, dummy_tokens, "audio_pt_from_dummy.wav")
    
    # Cross-decode tokens with MLX
    print("\n=== DECODING WITH MLX ===")
    audio_mlx_from_pt = decode_with_mlx(mlx_wrapper, tokens_pt, "audio_mlx_from_pt.wav")
    audio_mlx_from_mlx = decode_with_mlx(mlx_wrapper, tokens_mlx, "audio_mlx_from_mlx.wav")
    audio_mlx_from_dummy = decode_with_mlx(mlx_wrapper, dummy_tokens, "audio_mlx_from_dummy.wav")
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    print("* audio_pt_from_pt.wav: PyTorch tokens → PyTorch decoding")
    print("* audio_pt_from_mlx.wav: MLX tokens → PyTorch decoding")
    print("* audio_pt_from_dummy.wav: Dummy tokens → PyTorch decoding")
    print("* audio_mlx_from_pt.wav: PyTorch tokens → MLX decoding")
    print("* audio_mlx_from_mlx.wav: MLX tokens → MLX decoding")
    print("* audio_mlx_from_dummy.wav: Dummy tokens → MLX decoding")

def generate_pytorch_tokens(model: Generator, text: str, speaker_id: int) -> torch.Tensor:
    """Generate audio tokens using PyTorch model."""
    
    # Tokenize text
    text_tokens = model.tokenize(text)
    
    # Access internal model
    internal_model = model._model
    
    # Generate token sequence
    samples = []
    with torch.no_grad():
        max_tokens = 32 * 10  # 10 frames
        for i in range(max_tokens // 32 + 1):
            # Create context with speaker ID and previous tokens
            if i == 0:
                context = torch.cat([
                    text_tokens,
                    torch.tensor([[65632 + speaker_id]])
                ], dim=1)
            else:
                context = torch.cat([
                    text_tokens,
                    torch.tensor([[65632 + speaker_id]]),
                    *[torch.tensor([s]) for s in samples]
                ], dim=1)
            
            # Get model output for this position
            logits = internal_model(context)
            audio_head_logits = internal_model.audio_head(logits[:, -1:])
            
            # Generate tokens for each codebook
            for j in range(min(32, audio_head_logits.shape[1])):
                codebook_logits = audio_head_logits[:, j, :]
                token = internal_model.sample_topk(codebook_logits, 50, 1.2)
                samples.append(token)
                
                # Stop if we've generated enough tokens
                if len(samples) >= max_tokens:
                    break
                    
            if len(samples) >= max_tokens:
                break
    
    # Format tokens as expected by the decoder: [batch, codebooks, sequence]
    num_codebooks = 32
    sequence_length = len(samples) // num_codebooks
    tokens = torch.stack(samples[:sequence_length * num_codebooks])
    tokens = tokens.reshape(num_codebooks, sequence_length).unsqueeze(0)
    
    print(f"Generated PyTorch tokens with shape: {tokens.shape}")
    print(f"Token range: {tokens.min().item()} to {tokens.max().item()}")
    
    # Check for problematic tokens
    problematic = ((tokens > 0) & (tokens < 32)).sum().item()
    print(f"Problematic tokens (1-31): {problematic}")
    
    # Save tokens for reference
    torch.save(tokens, "tokens_pytorch.pt")
    print("Saved tokens to tokens_pytorch.pt")
    
    return tokens

def generate_mlx_tokens(mlx_wrapper, text: str, speaker_id: int) -> torch.Tensor:
    """Generate audio tokens using MLX model."""
    
    # Generate tokens using MLX
    mlx_tokens = mlx_wrapper.generate_tokens(
        text=text,
        speaker=speaker_id,
        max_audio_length_ms=800,  # ~10 frames
        temperature=1.2,
        topk=50
    )
    
    print(f"Generated MLX tokens with shape: {mlx_tokens.shape}")
    print(f"Token range: {mlx_tokens.min().item()} to {mlx_tokens.max().item()}")
    
    # Check for problematic tokens
    problematic = ((mlx_tokens > 0) & (mlx_tokens < 32)).sum().item()
    print(f"Problematic tokens (1-31): {problematic}")
    
    # Save tokens for reference
    torch.save(mlx_tokens, "tokens_mlx.pt")
    print("Saved tokens to tokens_mlx.pt")
    
    return mlx_tokens

def create_dummy_tokens() -> torch.Tensor:
    """Create dummy tokens with known safe values."""
    # Create a set of known good tokens
    safe_tokens = [
        0,    # Silence token
        42,   # Safe token known to work
        100,  # Safe token known to work
        200,  # Safe token known to work
        300,  # Safe token known to work
        400,  # Safe token known to work
        500,  # Safe token known to work
        600,  # Safe token known to work
        700,  # Safe token known to work
        800,  # Safe token known to work
        900,  # Safe token known to work
        1000, # Safe token known to work
        1100, # Safe token known to work
        1200, # Safe token known to work
        1300, # Safe token known to work
        1400, # Safe token known to work
        1500, # Safe token known to work
        1600, # Safe token known to work
        1700, # Safe token known to work
        1800, # Safe token known to work
        1900, # Safe token known to work
        2000  # Safe token known to work
    ]
    
    # Create a pattern that should produce valid speech
    num_codebooks = 32
    sequence_length = 10
    
    dummy_tokens = torch.zeros((1, num_codebooks, sequence_length), dtype=torch.long)
    
    # Fill with a pattern of safe tokens
    for cb in range(num_codebooks):
        token_idx = cb % len(safe_tokens)
        token_value = safe_tokens[token_idx]
        dummy_tokens[0, cb, :] = token_value
    
    print(f"Created dummy tokens with shape: {dummy_tokens.shape}")
    print(f"Token range: {dummy_tokens.min().item()} to {dummy_tokens.max().item()}")
    
    # Save tokens for reference
    torch.save(dummy_tokens, "tokens_dummy.pt")
    print("Saved tokens to tokens_dummy.pt")
    
    return dummy_tokens

def decode_with_pytorch(model: Generator, tokens: torch.Tensor, output_file: str) -> torch.Tensor:
    """Decode tokens to audio using PyTorch."""
    print(f"\nDecoding with PyTorch to {output_file}...")
    
    # Get audio tokenizer
    tokenizer = model._audio_tokenizer
    
    # Decode tokens
    start_time = time.time()
    audio = tokenizer.decode(tokens)
    decode_time = time.time() - start_time
    
    # Save audio
    try:
        import torchaudio
        torchaudio.save(output_file, audio.unsqueeze(0), 24000)
        print(f"Audio saved to {output_file}")
    except Exception as e:
        print(f"Error saving audio: {e}")
    
    print(f"PyTorch decoding time: {decode_time:.2f}s")
    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: {audio.min().item():.2f} to {audio.max().item():.2f}")
    
    # Check for NaN values
    nan_count = torch.isnan(audio).sum().item()
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} NaN values in audio")
    
    return audio

def decode_with_mlx(mlx_wrapper, tokens: torch.Tensor, output_file: str) -> torch.Tensor:
    """Decode tokens to audio using MLX."""
    print(f"\nDecoding with MLX to {output_file}...")
    
    # Decode tokens
    start_time = time.time()
    audio = mlx_wrapper.decode_audio_tokens(tokens)
    decode_time = time.time() - start_time
    
    # Save audio
    try:
        import torchaudio
        torchaudio.save(output_file, audio.unsqueeze(0), 24000)
        print(f"Audio saved to {output_file}")
    except Exception as e:
        print(f"Error saving audio: {e}")
    
    print(f"MLX decoding time: {decode_time:.2f}s")
    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: {audio.min().item():.2f} to {audio.max().item():.2f}")
    
    # Check for NaN values
    nan_count = torch.isnan(audio).sum().item()
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} NaN values in audio")
    
    return audio

def main():
    """Main function for cross-testing tokens and decoding."""
    cross_test_tokens_and_decode()

if __name__ == "__main__":
    main()