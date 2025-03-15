#!/usr/bin/env python
"""
Debug utility to compare PyTorch and MLX implementations of the CSM model.

This script systematically compares the outputs of the PyTorch and MLX 
implementations at each stage of inference to identify discrepancies.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Add parent directory to path to import csm modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from csm.generator import Generator
from csm.cli.mlx_components.generator import MLXGenerator
from csm.cli.mlx_components.config import MLXConfig
from csm.cli.mlx_embedding import mlx_sample_topk
from csm.cli.mlx_wrapper import load_mlx_model


def load_models(ckpt_path: str = None) -> Tuple[Generator, MLXGenerator]:
    """Load both PyTorch and MLX models for comparison."""
    print("Loading PyTorch model...")
    pt_model = Generator()
    
    print("Loading MLX model...")
    mlx_config = MLXConfig()
    mlx_config.debug = True
    mlx_model = MLXGenerator(model=None, config=mlx_config)
    mlx_model.load_model(ckpt_path)
    
    return pt_model, mlx_model


def compare_tokenization(pt_model: Generator, mlx_model: MLXGenerator, text: str):
    """Compare text tokenization between PyTorch and MLX models."""
    print("\n=== COMPARING TEXT TOKENIZATION ===")
    
    # PyTorch tokenization
    pt_start = time.time()
    pt_tokens = pt_model.tokenize(text)
    pt_time = time.time() - pt_start
    
    # MLX tokenization
    mlx_start = time.time()
    mlx_tokens = mlx_model.tokenize(text)
    mlx_time = time.time() - mlx_start
    
    # Compare results
    print(f"PyTorch tokens shape: {pt_tokens.shape}, MLX tokens shape: {mlx_tokens.shape}")
    if torch.is_tensor(pt_tokens) and torch.is_tensor(mlx_tokens):
        tokens_equal = torch.equal(pt_tokens, mlx_tokens)
        print(f"Tokens equal: {tokens_equal}")
    else:
        print(f"Token types differ: PT: {type(pt_tokens)}, MLX: {type(mlx_tokens)}")
    
    print(f"PyTorch time: {pt_time:.4f}s, MLX time: {mlx_time:.4f}s")
    
    return pt_tokens, mlx_tokens


def compare_generation(pt_model: Generator, mlx_model: MLXGenerator, 
                     text: str, voice: str = "warm", max_length_ms: int = 5000):
    """Compare full text-to-audio generation process."""
    print("\n=== COMPARING FULL GENERATION PIPELINE ===")
    
    # Get speaker ID for the voice
    speaker = pt_model.get_speaker_id(voice)
    print(f"Using voice: {voice} (speaker ID: {speaker})")
    
    # PyTorch generation
    print("\nRunning PyTorch generation...")
    pt_start = time.time()
    pt_audio = pt_model.generate_speech(
        text=text,
        speaker=speaker,
        max_audio_length_ms=max_length_ms
    )
    pt_time = time.time() - pt_start
    
    # MLX generation
    print("\nRunning MLX generation...")
    mlx_start = time.time()
    mlx_audio = mlx_model.generate_speech(
        text=text,
        speaker=speaker,
        max_audio_length_ms=max_length_ms
    )
    mlx_time = time.time() - mlx_start
    
    # Compare results
    print(f"\nPyTorch audio shape: {pt_audio.shape}, MLX audio shape: {mlx_audio.shape}")
    
    if torch.is_tensor(pt_audio) and torch.is_tensor(mlx_audio):
        # Calculate basic statistics
        pt_min, pt_max = pt_audio.min().item(), pt_audio.max().item()
        mlx_min, mlx_max = mlx_audio.min().item(), mlx_audio.max().item()
        
        print(f"PyTorch audio range: [{pt_min:.4f}, {pt_max:.4f}]")
        print(f"MLX audio range: [{mlx_min:.4f}, {mlx_max:.4f}]")
        
        # Calculate correlation if shapes match
        if pt_audio.shape == mlx_audio.shape:
            correlation = torch.corrcoef(torch.stack([pt_audio.flatten(), mlx_audio.flatten()]))[0, 1].item()
            print(f"Audio correlation: {correlation:.4f}")
            
            # Calculate mean squared error
            mse = ((pt_audio - mlx_audio) ** 2).mean().item()
            print(f"Mean squared error: {mse:.6f}")
        else:
            print("Cannot calculate correlation - shape mismatch")
    
    print(f"PyTorch generation time: {pt_time:.4f}s, MLX generation time: {mlx_time:.4f}s")
    print(f"PyTorch real-time factor: {pt_time / (max_length_ms / 1000):.2f}x")
    print(f"MLX real-time factor: {mlx_time / (max_length_ms / 1000):.2f}x")
    
    return pt_audio, mlx_audio


def compare_token_generation(pt_model: Generator, mlx_model: MLXGenerator, 
                           text: str, voice: str = "warm", max_length_ms: int = 5000):
    """Compare token generation specifically, before audio decoding."""
    print("\n=== COMPARING TOKEN GENERATION ===")
    
    # Get speaker ID for the voice
    speaker = pt_model.get_speaker_id(voice)
    
    # For PyTorch, we need to access the internal generation method
    pt_start = time.time()
    text_tokens = pt_model.tokenize(text)
    samples = []
    for i in range(max_length_ms // 240 + 1):
        context = torch.cat([
            text_tokens, 
            torch.tensor([[65632 + speaker]]),  # Adding speaker token
            *[torch.tensor([s]) for s in samples]
        ], dim=1)
        
        with torch.no_grad():
            logits = pt_model._model(context)
            
        # Get only the audio head output for the last token
        audio_head_logits = pt_model._model.audio_head(logits[:, -1:])  
        
        # Select from each codebook
        for j in range(audio_head_logits.shape[1]):
            codebook_logits = audio_head_logits[:, j, :]
            token = pt_model._model.sample_topk(codebook_logits, 50, 1.2)
            samples.append(token)
            
        if len(samples) >= 32 * 300:  # Safety to prevent infinite loops
            break
    
    pt_tokens = torch.stack(samples).permute(1, 2, 0)
    pt_time = time.time() - pt_start
    
    # For MLX, use the generator
    mlx_start = time.time()
    mlx_tokens = mlx_model.generate_tokens(
        text=text,
        speaker=speaker,
        max_audio_length_ms=max_length_ms
    )
    mlx_time = time.time() - mlx_start
    
    # Compare results
    print(f"PyTorch tokens shape: {pt_tokens.shape}")
    print(f"MLX tokens shape: {mlx_tokens.shape}")
    
    # Analyze token distributions
    pt_unique = torch.unique(pt_tokens)
    mlx_unique = torch.unique(mlx_tokens)
    
    print(f"PyTorch unique token count: {len(pt_unique)}")
    print(f"MLX unique token count: {len(mlx_unique)}")
    
    print(f"PyTorch tokens min: {pt_tokens.min().item()}, max: {pt_tokens.max().item()}")
    print(f"MLX tokens min: {mlx_tokens.min().item()}, max: {mlx_tokens.max().item()}")
    
    # Check for problematic tokens (1-31)
    problematic_pt = ((pt_tokens > 0) & (pt_tokens < 32)).sum().item()
    problematic_mlx = ((mlx_tokens > 0) & (mlx_tokens < 32)).sum().item()
    
    print(f"PyTorch problematic tokens (1-31): {problematic_pt}")
    print(f"MLX problematic tokens (1-31): {problematic_mlx}")
    
    print(f"PyTorch token generation time: {pt_time:.4f}s, MLX time: {mlx_time:.4f}s")
    
    return pt_tokens, mlx_tokens


def compare_decoding(pt_model: Generator, mlx_model: MLXGenerator, pt_tokens, mlx_tokens):
    """Compare audio decoding from tokens."""
    print("\n=== COMPARING AUDIO DECODING ===")
    
    # PyTorch decoding
    pt_start = time.time()
    pt_audio = pt_model._audio_tokenizer.decode(pt_tokens)
    pt_time = time.time() - pt_start
    
    # MLX decoding
    mlx_start = time.time()
    mlx_audio = mlx_model._audio_tokenizer.decode(mlx_tokens)
    mlx_time = time.time() - mlx_start
    
    # Compare results
    print(f"PyTorch audio shape: {pt_audio.shape}, MLX audio shape: {mlx_audio.shape}")
    
    if torch.is_tensor(pt_audio) and torch.is_tensor(mlx_audio):
        # Calculate basic statistics
        pt_min, pt_max = pt_audio.min().item(), pt_audio.max().item()
        mlx_min, mlx_max = mlx_audio.min().item(), mlx_audio.max().item()
        
        print(f"PyTorch audio range: [{pt_min:.4f}, {pt_max:.4f}]")
        print(f"MLX audio range: [{mlx_min:.4f}, {mlx_max:.4f}]")
        
        # Check for NaN values
        pt_nans = torch.isnan(pt_audio).sum().item()
        mlx_nans = torch.isnan(mlx_audio).sum().item()
        
        print(f"PyTorch NaN values: {pt_nans}")
        print(f"MLX NaN values: {mlx_nans}")
    
    print(f"PyTorch decode time: {pt_time:.4f}s, MLX decode time: {mlx_time:.4f}s")
    
    return pt_audio, mlx_audio


def compare_sampling_function(batch_size=1, vocab_size=2051, top_k=50, temperature=1.2):
    """Compare MLX and PyTorch sampling functions directly with controlled inputs."""
    print("\n=== COMPARING SAMPLING FUNCTIONS ===")
    
    # Create identical inputs for both functions
    logits = torch.randn(batch_size, vocab_size)
    
    # Apply small fixed offset to prevent identical values
    torch_logits = logits.clone()
    mlx_logits = logits.clone()
    
    # PyTorch sampling
    pt_start = time.time()
    from csm.models.model import Model  # Import here to get the original sampling function
    model = Model()
    pt_samples = []
    for _ in range(1000):  # Sample multiple times for distribution
        pt_samples.append(model.sample_topk(torch_logits, top_k, temperature))
    pt_time = time.time() - pt_start
    
    # MLX sampling
    mlx_start = time.time()
    mlx_samples = []
    for _ in range(1000):
        mlx_samples.append(mlx_sample_topk(mlx_logits, top_k, temperature))
    mlx_time = time.time() - mlx_start
    
    # Analyze results
    pt_samples = torch.cat(pt_samples)
    mlx_samples = torch.cat(mlx_samples)
    
    pt_unique, pt_counts = torch.unique(pt_samples, return_counts=True)
    mlx_unique, mlx_counts = torch.unique(mlx_samples, return_counts=True)
    
    # Convert to numpy for easier analysis
    pt_dist = {u.item(): c.item() for u, c in zip(pt_unique, pt_counts)}
    mlx_dist = {u.item(): c.item() for u, c in zip(mlx_unique, mlx_counts)}
    
    print(f"PyTorch sampling time: {pt_time:.4f}s, MLX sampling time: {mlx_time:.4f}s")
    print(f"PyTorch unique token count: {len(pt_unique)}")
    print(f"MLX unique token count: {len(mlx_unique)}")
    
    # Check for problematic tokens
    problematic_pt = sum(pt_dist.get(i, 0) for i in range(1, 32))
    problematic_mlx = sum(mlx_dist.get(i, 0) for i in range(1, 32))
    
    print(f"PyTorch problematic tokens (1-31): {problematic_pt}")
    print(f"MLX problematic tokens (1-31): {problematic_mlx}")
    
    return pt_dist, mlx_dist


def save_audio(audio, filename, sample_rate=24000):
    """Save audio tensor to WAV file."""
    try:
        import torchaudio
        torchaudio.save(filename, audio.unsqueeze(0), sample_rate)
        print(f"Audio saved to {filename}")
    except Exception as e:
        print(f"Error saving audio: {e}")


def main():
    """Main function to run all comparisons."""
    text = "Hello, this is a test of the CSM speech model."
    voice = "warm"
    max_length_ms = 3000
    
    # Load models
    pt_model, mlx_model = load_models()
    
    # Run comparisons
    pt_tokens, mlx_tokens = compare_tokenization(pt_model, mlx_model, text)
    
    # Compare sampling distributions
    pt_dist, mlx_dist = compare_sampling_function()
    
    # Compare token generation
    pt_tokens, mlx_tokens = compare_token_generation(pt_model, mlx_model, text, voice, max_length_ms)
    
    # Save token tensors for analysis
    torch.save(pt_tokens, "pt_tokens.pt")
    torch.save(mlx_tokens, "mlx_tokens.pt")
    print("Saved token tensors to pt_tokens.pt and mlx_tokens.pt")
    
    # Compare audio decoding
    pt_audio, mlx_audio = compare_decoding(pt_model, mlx_model, pt_tokens, mlx_tokens)
    
    # Compare full generation pipeline
    pt_audio, mlx_audio = compare_generation(pt_model, mlx_model, text, voice, max_length_ms)
    
    # Save audio files
    save_audio(pt_audio, "pt_audio.wav")
    save_audio(mlx_audio, "mlx_audio.wav")


if __name__ == "__main__":
    main()