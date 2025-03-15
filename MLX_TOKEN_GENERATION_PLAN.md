# MLX Token Generation Debugging and Fixing Plan

This document outlines a systematic approach to debug and fix the token generation in the MLX implementation of CSM to achieve matching quality with the PyTorch implementation.

## Background

The CSM model's MLX acceleration for Apple Silicon currently uses a hybrid approach where PyTorch handles token generation and MLX handles the rest of the pipeline. This produces high-quality audio, but to achieve a pure MLX implementation, we need to fix the token generation in MLX.

## Phase 1: Comparative Analysis

1. **Create a token capture utility**
   - Instrument both PyTorch and MLX pipelines to capture tokens at each step
   - Store intermediate tokens in a structured format for analysis
   - Focus on capturing: 
     - Raw logits before sampling
     - Token distributions after sampling
     - Tokens after any post-processing

2. **Analyze token distributions**
   - Compare token frequency histograms between PyTorch and MLX
   - Identify statistical differences in distribution patterns
   - Look for patterns in which tokens appear in PyTorch but not MLX

3. **Trace the divergence point**
   - Identify exactly where the token generation starts to differ
   - Compare logits, probabilities, and sampling outputs step by step
   - Focus on numerical differences in the sampling computation

## Phase 2: MLX Sampling Implementation

1. **Reimplement PyTorch sampling logic exactly in MLX**
   - Create a new `mlx_sample_exact` function that closely mirrors PyTorch code
   - Focus on reproducing the exact numerical operations
   - Pay special attention to:
     - Temperature scaling
     - Top-k filtering algorithm
     - Categorical sampling implementation

2. **Add comprehensive test cases**
   - Test with controlled inputs and random seeds
   - Verify that outputs match PyTorch with identical inputs
   - Test edge cases (extreme temperatures, different topk values)

3. **Fix MLX-specific numerical issues**
   - Address any floating-point precision differences
   - Ensure consistent random number generation
   - Fix reshape and tensor operation differences

## Phase 3: Integration and Validation

1. **Implement a switching mechanism**
   - Add a parameter to toggle between sampling implementations
   - Keep our hybrid solution as a fallback
   - Add the new pure-MLX implementation as the primary option

2. **Create a validation framework**
   - Generate audio with both implementations
   - Compare token distributions statistically
   - Create objective metrics for audio similarity

3. **Perform exhaustive testing**
   - Test with various text inputs
   - Test with different voice presets and parameters
   - Run stress tests with longer generation

## Phase 4: Optimization and Refinement

1. **Profile and optimize performance**
   - Identify any bottlenecks in the MLX implementation
   - Apply MLX-specific optimizations
   - Compare performance with hybrid solution

2. **Fine-tune for audio quality**
   - Adjust sampling parameters for best audio quality
   - Implement any necessary post-processing specific to MLX
   - Optimize for both quality and speed

3. **Document the implementation**
   - Create detailed documentation of the solution
   - Note any MLX-specific considerations
   - Provide examples of expected token distributions

## Implementation Tools

### Token Analyzer

```python
# token_analyzer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Capture token info from both implementations
def capture_tokens(pt_model, mlx_model, text, verbose=True):
    """Capture token generation from both implementations."""
    results = {}
    
    # Generate with PyTorch
    if verbose:
        print("Generating with PyTorch...")
    
    pt_audio = pt_model.generate(text=text, speaker=0)
    pt_tokens = pt_model._last_tokens
    
    # Generate with MLX hybrid (our current working solution)
    if verbose:
        print("Generating with MLX hybrid...")
    
    mlx_audio = mlx_model.generate_speech(text=text, speaker=0)
    mlx_tokens = mlx_model._last_tokens
    
    # Store results
    results['pytorch'] = {
        'tokens': pt_tokens,
        'audio': pt_audio
    }
    
    results['mlx'] = {
        'tokens': mlx_tokens,
        'audio': mlx_audio
    }
    
    # Analyze distributions
    if verbose:
        analyze_distributions(pt_tokens, mlx_tokens)
    
    return results

def analyze_distributions(pt_tokens, mlx_tokens):
    """Analyze token distributions."""
    # Flatten tensors
    pt_flat = pt_tokens.flatten().tolist()
    mlx_flat = mlx_tokens.flatten().tolist()
    
    # Count tokens
    pt_counter = Counter(pt_flat)
    mlx_counter = Counter(mlx_flat)
    
    print(f"\nPyTorch: {len(pt_counter)} unique tokens")
    print(f"MLX: {len(mlx_counter)} unique tokens")
    
    # Top tokens
    print("\nPyTorch top tokens:")
    for token, count in pt_counter.most_common(10):
        print(f"  {token}: {count} times ({count/len(pt_flat)*100:.2f}%)")
        
    print("\nMLX top tokens:")
    for token, count in mlx_counter.most_common(10):
        print(f"  {token}: {count} times ({count/len(mlx_flat)*100:.2f}%)")
        
    # Calculate overlap
    pt_set = set(pt_counter.keys())
    mlx_set = set(mlx_counter.keys())
    common = pt_set.intersection(mlx_set)
    
    print(f"\nToken overlap: {len(common)} tokens ({len(common)/len(pt_set)*100:.2f}% of PyTorch tokens)")
    
    # Distribution similarity
    similarity = 0
    for token in common:
        pt_prob = pt_counter[token] / len(pt_flat)
        mlx_prob = mlx_counter[token] / len(mlx_flat)
        # Use min to measure overlap (Jaccard similarity)
        similarity += min(pt_prob, mlx_prob)
    
    print(f"Distribution similarity: {similarity*100:.2f}%")
```

### Exact PyTorch-Matching MLX Sampling Function

```python
def mlx_sample_exact(logits: mx.array, topk: int = 5, temperature: float = 1.0, seed: int = 42) -> mx.array:
    """
    MLX implementation that exactly matches PyTorch sampling behavior.
    
    Args:
        logits: Raw logits [batch_size, vocab_size]
        topk: Number of top tokens to consider
        temperature: Temperature for sampling
        seed: Random seed
        
    Returns:
        Sampled token indices [batch_size, 1]
    """
    # Ensure proper shape
    if len(logits.shape) == 1:
        logits = logits.reshape(1, -1)
        
    batch_size, vocab_size = logits.shape
    
    # Apply temperature scaling
    scaled_logits = logits / (temperature + 1e-10)
    
    # Apply top-k filtering
    samples = mx.zeros((batch_size, 1), dtype=mx.int32)
    
    # Get random key
    key = mx.random.key(seed)
    
    for b in range(batch_size):
        # Get batch logits
        batch_logits = scaled_logits[b]
        
        # THIS IS CRITICAL: Match PyTorch's top-k implementation exactly
        # 1. Sort values
        sorted_logits, sorted_indices = mx.sort(batch_logits, descending=True)
        
        # 2. Get top-k threshold value
        k = min(topk, vocab_size)
        threshold = sorted_logits[k-1]
        
        # 3. Apply mask using exactly matching operations
        #    IMPORTANT: PyTorch uses a different masking approach than our previous implementation
        mask = batch_logits < threshold
        filtered_logits = mx.where(mask, mx.array(-float('inf')), batch_logits)
        
        # 4. Apply softmax
        probs = mx.softmax(filtered_logits)
        
        # 5. Match PyTorch's categorical sampling with Gumbel-max trick
        key, subkey = mx.random.split(key)
        gumbel_noise = -mx.log(-mx.log(mx.random.uniform(key=subkey, shape=probs.shape) + 1e-10) + 1e-10)
        gumbel_logits = mx.log(probs + 1e-10) + gumbel_noise
        
        # Get sample
        sample_idx = mx.argmax(gumbel_logits)
        
        # Safety check: never return problematic tokens
        if 1 <= sample_idx < 32:
            # Use silence token instead
            sample_idx = mx.array(0)
            
        # Set result
        samples_list = samples.tolist()
        samples_list[b][0] = sample_idx.item()
        samples = mx.array(samples_list)
    
    return samples
```

### Implementation Comparison Tool

```python
def compare_sampling_implementations(text="Hello, this is a test of the CSM speech model."):
    """Compare different sampling implementations."""
    # Load models
    from csm.generator import load_csm_1b
    from csm.cli.mlx_components.generator import MLXGenerator
    from csm.cli.mlx_components.config import MLXConfig
    
    # Load PyTorch model
    pt_model = load_csm_1b(device="cpu")
    
    # Load MLX model
    mlx_config = MLXConfig()
    mlx_model = MLXGenerator(model=pt_model._model, tokenizer=pt_model._tokenizer, debug=True)
    
    # Generate with current implementation (hybrid)
    print("=== Current hybrid implementation ===")
    hybrid_results = capture_tokens(pt_model, mlx_model, text)
    
    # Save the current tokens
    torch.save(hybrid_results, "hybrid_tokens.pt")
    
    # Now patch the MLX model to use our exact sampling implementation
    from csm.cli.mlx_embedding import mlx_sample_topk
    
    # Temporarily backup original function
    original_sample_topk = mlx_sample_topk
    
    # Patch with our exact implementation
    import sys
    sys.modules['csm.cli.mlx_embedding'].mlx_sample_topk = mlx_sample_exact
    
    print("\n=== Exact matching implementation ===")
    exact_results = capture_tokens(pt_model, mlx_model, text)
    
    # Save the exact tokens
    torch.save(exact_results, "exact_tokens.pt")
    
    # Restore original function
    sys.modules['csm.cli.mlx_embedding'].mlx_sample_topk = original_sample_topk
    
    # Print summary
    print("\n=== Summary ===")
    
    hybrid_similarity = distribution_similarity(
        hybrid_results['pytorch']['tokens'],
        hybrid_results['mlx']['tokens']
    )
    
    exact_similarity = distribution_similarity(
        exact_results['pytorch']['tokens'],
        exact_results['mlx']['tokens']
    )
    
    print(f"Hybrid implementation similarity: {hybrid_similarity*100:.2f}%")
    print(f"Exact implementation similarity: {exact_similarity*100:.2f}%")
    
    # Return all results for further analysis
    return {
        'hybrid': hybrid_results,
        'exact': exact_results
    }
```

## Next Steps

1. Implement token capture and analysis
2. Create the exact MLX sampling implementation
3. Test and validate with various inputs
4. Integrate into the main codebase
5. Optimize performance and audio quality