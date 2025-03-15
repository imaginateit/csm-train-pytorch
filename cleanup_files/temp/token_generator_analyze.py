#!/usr/bin/env python
"""
Tool to analyze and debug token generation and distribution issues.
"""

import os
import sys
import torch
import numpy as np
import torchaudio
from collections import Counter
import time

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MLX
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available!")

# Import CSM functions
from csm.generator import load_csm_1b
from csm.cli.mlx_embedding import mlx_sample_topk

def create_test_tokens():
    """
    Create a controlled set of tokens for testing the audio codec.
    """
    print("=== Creating test tokens ===")
    
    # Create tokens with various patterns
    # PyTorch shape: [batch_size, num_codebooks, seq_length]
    batch_size = 1
    num_codebooks = 32
    seq_length = 100  # Approximately 3 seconds
    
    # Create test 1: Regular PyTorch tokens (from working model)
    print("Loading PyTorch model...")
    model = load_csm_1b(device="cpu")
    
    print("Generating real tokens with PyTorch...")
    audio_pt = model.generate(
        text="This is a test of the audio codec.",
        speaker=0,
        max_audio_length_ms=3000
    )
    
    # Extract the tokens that made this audio
    pt_tokens = model._last_tokens if hasattr(model, '_last_tokens') else None
    
    if pt_tokens is None:
        print("Could not get tokens from PyTorch model, creating dummy tokens")
        # Create dummy tokens with expected distribution
        pt_tokens = torch.zeros((batch_size, num_codebooks, seq_length), dtype=torch.long)
        # Fill with random values using actual distribution
        for cb in range(num_codebooks):
            # Each codebook has different token distribution
            base = 42 + (cb * 10) % 100
            for s in range(seq_length):
                # Introduce controlled variety
                pt_tokens[0, cb, s] = base + ((s*7) % 40)
    
    # Save these known-good tokens and audio
    torch.save(pt_tokens, "working_tokens.pt")
    torchaudio.save("working_audio.wav", audio_pt.unsqueeze(0), 24000)
    
    # Create test 2: Fixed pattern
    # Create a pattern with tokens that should be safe
    pattern_tokens = torch.zeros((batch_size, num_codebooks, seq_length), dtype=torch.long)
    
    # Using known working token values
    safe_values = [0, 42, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for cb in range(num_codebooks):
        # Each codebook gets a different tone
        tone = safe_values[cb % len(safe_values)]
        for s in range(seq_length):
            # Add small variations
            pattern_tokens[0, cb, s] = tone + (s % 10)
    
    torch.save(pattern_tokens, "pattern_tokens.pt")
    
    # Create test 3: Copy of working tokens with broader range
    broader_tokens = pt_tokens.clone()
    
    # Introduce more variety but stay in safe ranges
    for cb in range(num_codebooks):
        for s in range(seq_length):
            if s % 5 == 0:
                # Every 5th token gets a different value
                broader_tokens[0, cb, s] = (pt_tokens[0, cb, s] + 100) % 2048
    
    torch.save(broader_tokens, "broader_tokens.pt")
    
    # Create test 4: Mix of zeros and pattern
    mixed_tokens = torch.zeros((batch_size, num_codebooks, seq_length), dtype=torch.long)
    
    for cb in range(num_codebooks):
        base_val = 100 * (cb % 10 + 1)
        for s in range(seq_length):
            # Only set values for even indices, leave zeros for odd
            if s % 2 == 0:
                mixed_tokens[0, cb, s] = base_val
    
    torch.save(mixed_tokens, "mixed_tokens.pt")
    
    # Create test 5: Problematic token test
    problem_tokens = torch.zeros((batch_size, num_codebooks, seq_length), dtype=torch.long)
    
    # Fill with values that might cause issues
    for cb in range(num_codebooks):
        for s in range(seq_length):
            # Create patterns of potentially problematic tokens
            if cb % 4 == 0:
                # Include some silence (0s)
                problem_tokens[0, cb, s] = 0
            elif cb % 4 == 1:
                # Include some known good tokens
                problem_tokens[0, cb, s] = 42 + (s % 10)
            elif cb % 4 == 2:
                # Include borderline tokens
                problem_tokens[0, cb, s] = 31 + (s % 4)
            else:
                # Include potentially problematic range
                problem_tokens[0, cb, s] = 1 + (s % 31)
    
    torch.save(problem_tokens, "problem_tokens.pt")
    
    print("Created and saved test token sets")
    return pt_tokens, pattern_tokens, broader_tokens, mixed_tokens, problem_tokens

def decode_test_tokens(model):
    """
    Try to decode each of the test token sets.
    """
    print("\n=== Testing Audio Decoding with Different Token Sets ===")
    
    tokenizer = model._audio_tokenizer
    
    # Load each token set and try to decode
    for name in ["working_tokens", "pattern_tokens", "broader_tokens", "mixed_tokens", "problem_tokens"]:
        try:
            print(f"\nTesting decoding for {name}...")
            tokens = torch.load(f"{name}.pt")
            
            # Get information about tokens
            token_min = tokens.min().item()
            token_max = tokens.max().item()
            unique_count = len(torch.unique(tokens))
            
            print(f"Token shape: {tokens.shape}")
            print(f"Token range: {token_min} to {token_max}")
            print(f"Unique token count: {unique_count}")
            
            # Check for problematic tokens
            problematic = ((tokens > 0) & (tokens < 32)).sum().item()
            if problematic > 0:
                print(f"WARNING: Contains {problematic} problematic tokens (1-31)")
            
            # Try to decode
            start_time = time.time()
            try:
                audio = tokenizer.decode(tokens)
                decode_time = time.time() - start_time
                
                # Check audio quality
                if isinstance(audio, torch.Tensor):
                    audio_min = audio.min().item()
                    audio_max = audio.max().item()
                    nan_count = torch.isnan(audio).sum().item()
                    
                    print(f"Decoded audio shape: {audio.shape}")
                    print(f"Audio range: {audio_min:.4f} to {audio_max:.4f}")
                    if nan_count > 0:
                        print(f"WARNING: Contains {nan_count} NaN values")
                    
                    # Save the decoded audio
                    torchaudio.save(f"{name}_audio.wav", audio.unsqueeze(0), 24000)
                    print(f"Saved decoded audio to {name}_audio.wav")
                    print(f"Decoding took {decode_time:.4f} seconds")
                else:
                    print(f"Warning: Decoder returned non-tensor {type(audio)}")
                    
            except Exception as e:
                decode_time = time.time() - start_time
                print(f"Error decoding {name}: {e}")
                print(f"Failed after {decode_time:.4f} seconds")
                
                # Try to identify problematic codebooks
                try:
                    for cb in range(tokens.shape[1]):
                        cb_tokens = tokens[:, cb:cb+1, :]
                        try:
                            # Embed in try block to catch errors
                            tokenizer.decode(cb_tokens)
                            # If we get here, this codebook decoded successfully
                        except Exception as cb_error:
                            print(f"Codebook {cb} fails: {cb_error}")
                            # Check token stats for this codebook
                            cb_values = tokens[0, cb, :].unique().tolist()
                            print(f"  Values: {cb_values[:10]}... (total: {len(cb_values)})")
                except Exception as debug_error:
                    print(f"Error during debug: {debug_error}")
                    
        except Exception as e:
            print(f"Error loading or processing {name}: {e}")

def generate_and_test_mlx_tokens():
    """
    Create tokens from both MLX and PyTorch sampling.
    """
    if not MLX_AVAILABLE:
        print("MLX not available, skipping MLX token generation")
        return
    
    print("\n=== Generating Tokens with MLX Sampling ===")
    
    # Prepare the same input
    batch_size = 1
    vocab_size = 2051
    
    np.random.seed(42)
    logits_np = np.random.randn(batch_size, vocab_size).astype(np.float32)
    
    # Make some tokens more likely
    for token in [0, 42, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        logits_np[:, token] += 5.0
    
    # Make problematic tokens less likely
    for token in range(1, 32):
        logits_np[:, token] -= 10.0
    
    # Convert to tensors
    logits_mx = mx.array(logits_np)
    
    # Sample large number of tokens with MLX function
    print("Sampling 3200 tokens with MLX (to simulate 32 codebooks x 100 frames)")
    mlx_tokens = []
    start_time = time.time()
    
    for i in range(3200):
        sample = mlx_sample_topk(logits_mx, topk=50, temperature=1.2, seed=42+i)
        mlx_tokens.append(sample.item())
    
    end_time = time.time()
    
    # Analyze the MLX token distribution
    token_counter = Counter(mlx_tokens)
    unique_count = len(token_counter)
    
    print(f"MLX sampling took {end_time - start_time:.4f} seconds")
    print(f"Unique tokens: {unique_count} out of 2051 possible")
    
    # Check for problematic tokens
    problematic = sum(1 for t in mlx_tokens if 1 <= t <= 31)
    if problematic > 0:
        print(f"WARNING: Generated {problematic} problematic tokens (1-31)")
    
    # Most common tokens
    print("\nMost common tokens:")
    for token, count in token_counter.most_common(10):
        print(f"  Token {token}: {count} times ({count/len(mlx_tokens)*100:.2f}%)")
    
    # Reshape to match expected format [batch_size, num_codebooks, seq_length]
    num_codebooks = 32
    seq_length = 100
    
    # Create proper tensor
    tokens_tensor = torch.zeros((batch_size, num_codebooks, seq_length), dtype=torch.long)
    idx = 0
    for cb in range(num_codebooks):
        for s in range(seq_length):
            if idx < len(mlx_tokens):
                tokens_tensor[0, cb, s] = mlx_tokens[idx]
                idx += 1
    
    # Save the tokens
    torch.save(tokens_tensor, "mlx_generated_tokens.pt")
    print(f"Saved MLX-generated tokens to mlx_generated_tokens.pt")
    
    return tokens_tensor

def main():
    # Load the model
    print("Loading PyTorch model...")
    model = load_csm_1b(device="cpu")
    print("Model loaded successfully")
    
    # Create test token sets
    test_tokens = create_test_tokens()
    
    # Test decoding with various token sets
    decode_test_tokens(model)
    
    # Generate and test MLX tokens
    mlx_tokens = generate_and_test_mlx_tokens()
    
    # Test our MLX-generated tokens too
    if mlx_tokens is not None:
        print("\n=== Testing MLX-generated tokens ===")
        try:
            audio = model._audio_tokenizer.decode(mlx_tokens)
            if isinstance(audio, torch.Tensor):
                torchaudio.save("mlx_generated_audio.wav", audio.unsqueeze(0), 24000)
                print(f"Saved MLX-generated audio to mlx_generated_audio.wav")
            else:
                print(f"Warning: Decoder returned non-tensor {type(audio)}")
        except Exception as e:
            print(f"Error decoding MLX-generated tokens: {e}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()