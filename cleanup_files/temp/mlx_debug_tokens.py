#!/usr/bin/env python
"""
Token generation debugging tool that compares PyTorch and MLX token outputs.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path to import csm modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from csm.generator import Generator
from csm.cli.mlx_components.config import MLXConfig
from csm.cli.mlx_wrapper import load_mlx_model

# Check if MLX is available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

def save_tokens_to_json(pt_tokens: torch.Tensor, mlx_tokens: torch.Tensor, filename: str = "token_comparison.json"):
    """Save token comparison to a JSON file for detailed analysis."""
    data = {
        "pytorch": {
            "shape": list(pt_tokens.shape),
            "min": float(pt_tokens.min()),
            "max": float(pt_tokens.max()),
            "tokens": pt_tokens.cpu().numpy().tolist() if isinstance(pt_tokens, torch.Tensor) else pt_tokens.tolist(),
            "unique_count": len(torch.unique(pt_tokens)),
            "histogram": get_token_histogram(pt_tokens)
        },
        "mlx": {
            "shape": list(mlx_tokens.shape),
            "min": float(mlx_tokens.min()),
            "max": float(mlx_tokens.max()),
            "tokens": mlx_tokens.cpu().numpy().tolist() if isinstance(mlx_tokens, torch.Tensor) else mlx_tokens.tolist(),
            "unique_count": len(torch.unique(mlx_tokens)),
            "histogram": get_token_histogram(mlx_tokens)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Token comparison saved to {filename}")

def get_token_histogram(tokens: torch.Tensor) -> Dict[int, int]:
    """Create a histogram of token values."""
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)
    
    # Flatten tensor
    flat_tokens = tokens.flatten()
    
    # Count occurrences
    unique, counts = torch.unique(flat_tokens, return_counts=True)
    
    # Convert to dictionary
    histogram = {int(val.item()): int(count.item()) for val, count in zip(unique, counts)}
    
    return histogram

def generate_and_compare_tokens(text: str = "Hello, this is a test", voice: str = "warm", num_tokens: int = 100):
    """Generate audio tokens with both PyTorch and MLX and compare them."""
    print(f"Generating tokens for text: '{text}'")
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    pt_model = Generator()
    
    if not MLX_AVAILABLE:
        print("MLX is not available. Cannot perform comparison.")
        return
    
    # Load MLX model
    print("Loading MLX model...")
    mlx_config = MLXConfig()
    mlx_config.debug = True
    mlx_wrapper = load_mlx_model(None, mlx_config)
    
    # Set speaker ID
    speaker_id = pt_model.get_speaker_id(voice)
    print(f"Using voice: {voice} (speaker ID: {speaker_id})")
    
    # Generate tokens with PyTorch
    print("\nGenerating tokens with PyTorch...")
    pt_start = time.time()
    pt_tokens = generate_tokens_pytorch(pt_model, text, speaker_id, num_tokens)
    pt_time = time.time() - pt_start
    
    # Generate tokens with MLX
    print("\nGenerating tokens with MLX...")
    mlx_start = time.time()
    mlx_tokens = generate_tokens_mlx(mlx_wrapper, text, speaker_id, num_tokens)
    mlx_time = time.time() - mlx_start
    
    # Compare results
    print(f"\nPyTorch tokens shape: {pt_tokens.shape}, generation time: {pt_time:.2f}s")
    print(f"MLX tokens shape: {mlx_tokens.shape}, generation time: {mlx_time:.2f}s")
    
    # Basic token statistics
    print(f"\nPyTorch token range: {pt_tokens.min().item()} to {pt_tokens.max().item()}")
    print(f"MLX token range: {mlx_tokens.min().item()} to {mlx_tokens.max().item()}")
    
    # Count unique tokens
    pt_unique = torch.unique(pt_tokens)
    mlx_unique = torch.unique(mlx_tokens)
    
    print(f"PyTorch unique token count: {len(pt_unique)}")
    print(f"MLX unique token count: {len(mlx_unique)}")
    
    # Check for problematic tokens (1-31 range)
    pt_problematic = ((pt_tokens > 0) & (pt_tokens < 32)).sum().item()
    mlx_problematic = ((mlx_tokens > 0) & (mlx_tokens < 32)).sum().item()
    
    print(f"PyTorch problematic tokens (1-31 range): {pt_problematic}")
    print(f"MLX problematic tokens (1-31 range): {mlx_problematic}")
    
    # Check common patterns
    print("\nMost common PyTorch tokens:")
    pt_values, pt_counts = torch.unique(pt_tokens, return_counts=True)
    pt_sorted_indices = torch.argsort(pt_counts, descending=True)
    for i in range(min(10, len(pt_sorted_indices))):
        idx = pt_sorted_indices[i]
        print(f"  Token {pt_values[idx].item()}: {pt_counts[idx].item()} occurrences")
    
    print("\nMost common MLX tokens:")
    mlx_values, mlx_counts = torch.unique(mlx_tokens, return_counts=True)
    mlx_sorted_indices = torch.argsort(mlx_counts, descending=True)
    for i in range(min(10, len(mlx_sorted_indices))):
        idx = mlx_sorted_indices[i]
        print(f"  Token {mlx_values[idx].item()}: {mlx_counts[idx].item()} occurrences")
    
    # Calculate token overlap
    pt_set = set(pt_unique.tolist())
    mlx_set = set(mlx_unique.tolist())
    common_tokens = pt_set.intersection(mlx_set)
    
    print(f"\nToken set overlap: {len(common_tokens)} tokens ({len(common_tokens)/len(pt_set)*100:.1f}% of PyTorch tokens)")
    
    # Save detailed comparison
    save_tokens_to_json(pt_tokens, mlx_tokens)
    
    # Save tokens for audio generation test
    torch.save(pt_tokens, "pt_tokens_debug.pt")
    torch.save(mlx_tokens, "mlx_tokens_debug.pt")
    print("\nSaved token tensors to pt_tokens_debug.pt and mlx_tokens_debug.pt")
    
    return pt_tokens, mlx_tokens

def generate_tokens_pytorch(model: Generator, text: str, speaker_id: int, num_tokens: int) -> torch.Tensor:
    """Generate audio tokens using PyTorch model."""
    
    # Tokenize text
    text_tokens = model.tokenize(text)
    
    # Access internal model
    internal_model = model._model
    
    # Generate token sequence
    samples = []
    with torch.no_grad():
        for i in range(num_tokens // 32 + 1):
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
                if len(samples) >= num_tokens:
                    break
                    
            if len(samples) >= num_tokens:
                break
    
    # Format tokens as expected by the decoder: [batch, codebooks, sequence]
    num_codebooks = 32
    sequence_length = len(samples) // num_codebooks
    tokens = torch.stack(samples[:sequence_length * num_codebooks])
    tokens = tokens.reshape(num_codebooks, sequence_length).unsqueeze(0)
    
    return tokens

def generate_tokens_mlx(mlx_wrapper, text: str, speaker_id: int, num_tokens: int) -> torch.Tensor:
    """Generate audio tokens using MLX model."""
    
    # Use the MLX generate method
    max_audio_length_ms = (num_tokens // 32) * 80  # 80ms per frame, 32 codebooks per frame
    
    # Generate tokens using MLX
    mlx_tokens = mlx_wrapper.generate_tokens(
        text=text,
        speaker=speaker_id,
        max_audio_length_ms=max_audio_length_ms,
        temperature=1.2,
        topk=50
    )
    
    # Ensure we get exactly the tokens we need
    mlx_tokens = mlx_tokens[:, :, :num_tokens//32]
    
    return mlx_tokens

def test_decode_audio(model: Generator, pt_tokens: torch.Tensor, mlx_tokens: torch.Tensor):
    """Decode both token sets to audio and save to files."""
    print("\n=== Testing Audio Decoding ===")
    
    # Decode PyTorch tokens
    print("\nDecoding PyTorch tokens...")
    pt_start = time.time()
    pt_audio = model._audio_tokenizer.decode(pt_tokens)
    pt_time = time.time() - pt_start
    
    # Decode MLX tokens
    print("\nDecoding MLX tokens...")
    mlx_start = time.time()
    mlx_audio = model._audio_tokenizer.decode(mlx_tokens)
    mlx_time = time.time() - mlx_start
    
    # Save audio files
    import torchaudio
    torchaudio.save("pt_audio_debug.wav", pt_audio.unsqueeze(0), 24000)
    torchaudio.save("mlx_audio_debug.wav", mlx_audio.unsqueeze(0), 24000)
    
    print(f"Saved audio files to pt_audio_debug.wav and mlx_audio_debug.wav")
    print(f"PyTorch decoding time: {pt_time:.2f}s, MLX decoding time: {mlx_time:.2f}s")
    
    return pt_audio, mlx_audio

def main():
    """Main function for token generation debugging."""
    # Generate and compare tokens
    pt_tokens, mlx_tokens = generate_and_compare_tokens(
        text="Hello, this is a test of the speech model.",
        voice="warm",
        num_tokens=256  # 8 frames worth of tokens
    )
    
    # Load PyTorch model for audio decoding
    pt_model = Generator()
    
    # Test decoding to audio
    test_decode_audio(pt_model, pt_tokens, mlx_tokens)

if __name__ == "__main__":
    main()