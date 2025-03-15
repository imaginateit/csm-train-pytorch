#!/usr/bin/env python
"""
Simple token generator testing script that uses hardcoded known-good tokens.
"""

import os
import sys
import time
import torch
import torchaudio
import numpy as np
from huggingface_hub import hf_hub_download
from moshi.models import loaders

# Create known-good tokens
def create_good_tokens():
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
    
    # Create a good pattern
    good_tokens = torch.zeros((1, num_codebooks, sequence_length), dtype=torch.long)
    
    # Fill with a pattern of safe tokens
    for cb in range(num_codebooks):
        token_idx = cb % len(safe_tokens)
        token_value = safe_tokens[token_idx]
        good_tokens[0, cb, :] = token_value
    
    return good_tokens

# Create tokens with problematic values
def create_problematic_tokens():
    good_tokens = create_good_tokens()
    bad_tokens = good_tokens.clone()
    
    # Add a few problematic tokens (values 1-31)
    for i in range(1, 20, 4):
        bad_tokens[0, i % 32, i % 10] = i
        
    return bad_tokens

# Get audio tokenizer
def get_tokenizer():
    device = "cpu"
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(32)
    return mimi

def test_decoding():
    print("Creating token sets...")
    good_tokens = create_good_tokens()
    bad_tokens = create_problematic_tokens()
    
    print("Loading audio tokenizer...")
    tokenizer = get_tokenizer()
    
    print("\nDecoding good tokens...")
    start_time = time.time()
    good_audio = tokenizer.decode(good_tokens)
    good_time = time.time() - start_time
    
    print(f"Good tokens shape: {good_tokens.shape}")
    print(f"Decoded audio shape: {good_audio.shape}")
    print(f"Audio range: {good_audio.min().item():.4f} to {good_audio.max().item():.4f}")
    print(f"Decoding time: {good_time:.2f}s")
    
    # Check for NaN values
    nan_count = torch.isnan(good_audio).sum().item()
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} NaN values in good audio")
    
    # Save good audio
    torchaudio.save("good_audio.wav", good_audio.unsqueeze(0), 24000)
    print("Saved good audio to good_audio.wav")
    
    print("\nDecoding bad tokens...")
    start_time = time.time()
    try:
        bad_audio = tokenizer.decode(bad_tokens)
        bad_time = time.time() - start_time
        
        print(f"Bad tokens shape: {bad_tokens.shape}")
        print(f"Decoded audio shape: {bad_audio.shape}")
        print(f"Audio range: {bad_audio.min().item():.4f} to {bad_audio.max().item():.4f}")
        print(f"Decoding time: {bad_time:.2f}s")
        
        # Check for NaN values
        nan_count = torch.isnan(bad_audio).sum().item()
        if nan_count > 0:
            print(f"WARNING: Found {nan_count} NaN values in bad audio")
            
        # Save bad audio
        torchaudio.save("bad_audio.wav", bad_audio.unsqueeze(0), 24000)
        print("Saved bad audio to bad_audio.wav")
    except Exception as e:
        print(f"Error decoding bad tokens: {e}")

if __name__ == "__main__":
    test_decoding()