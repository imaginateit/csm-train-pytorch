# MLX Token Generation Utilities

This document provides information about the MLX token generation utilities implemented for CSM (Conversational Speech Model).

## Overview

The MLX token generation system includes:

1. **Token Analyzer**: Utilities for capturing and analyzing token distributions
2. **Exact MLX Sampling**: Implementation that exactly matches PyTorch's sampling behavior
3. **Integration Utilities**: Tools for switching between sampling implementations

## Components

### Token Analyzer (`token_analyzer.py`)

The token analyzer provides utilities for comparing token generation between PyTorch and MLX:

```python
from csm.cli.token_analyzer import capture_tokens, analyze_distributions

# Capture tokens from both implementations
results = capture_tokens(pt_model, mlx_model, "Hello world")

# Analyze distributions
pt_tokens = results['pytorch']['tokens']
mlx_tokens = results['mlx']['tokens']
analyze_distributions(pt_tokens, mlx_tokens)
```

### Exact MLX Sampling (`mlx_sample_exact.py`)

The exact MLX sampling implementation provides a direct replacement for the default MLX sampling:

```python
from csm.cli.mlx_sample_exact import mlx_sample_exact

# Sample tokens from logits
tokens = mlx_sample_exact(logits, topk=50, temperature=0.9, seed=42)
```

### Integration Utilities (`use_exact_sampling.py`)

Utilities for patching the MLX module to use exact sampling:

```python
from csm.cli.use_exact_sampling import patch_mlx_sampling

# Enable exact sampling
original_func = patch_mlx_sampling(enable=True, verbose=True)

# Disable and restore original implementation
patch_mlx_sampling(enable=False)
```

## CLI Options

The `csm-generate-mlx` command-line tool supports the following options for token generation:

```bash
# Use the exact PyTorch-matching sampling implementation
csm-generate-mlx --text "Hello world" --use-exact-sampling

# Force hybrid mode (PyTorch token generation + MLX pipeline)
csm-generate-mlx --text "Hello world" --pytorch-tokens

# Enable debugging output
csm-generate-mlx --text "Hello world" --debug
```

## Testing

To run token generation tests:

```bash
# Run all tests with the test script
./test_tokens.sh

# Run specific tests directly
python -m src.csm.cli.test_token_generation --test-sampling
python -m src.csm.cli.test_token_generation --text "Test text" --use-exact
```

## Sampling Strategies

The system supports three sampling strategies:

1. **Hybrid Mode**: Uses PyTorch for token generation and MLX for the rest of the pipeline (default fallback)
2. **Exact MLX Mode**: Uses MLX implementation that directly mimics PyTorch's sampling behavior
3. **Pure MLX Mode**: Uses MLX's native sampling implementation 

## Performance and Quality Considerations

- **Hybrid Mode**: Highest quality with perfect match to PyTorch generation, but requires PyTorch conversion overhead
- **Exact MLX Mode**: Nearly identical quality to PyTorch with pure MLX execution, slightly slower than pure MLX
- **Pure MLX Mode**: Fastest execution with all-MLX implementation, can produce high-quality audio with proper tuning

The common perception that MLX sampling must be lower quality than PyTorch is incorrect. With appropriate implementation of the sampling functions (particularly the Gumbel-max trick and proper numerical stability), MLX can produce audio quality comparable to PyTorch while maintaining the performance benefits of pure MLX execution.

## Troubleshooting

If you encounter audio quality issues:

1. Try `--use-exact-sampling` for a better MLX implementation
2. Try `--pytorch-tokens` for the highest quality output
3. Use `--debug` to see which sampling strategy is being used

If you're getting numerical errors in token generation:

1. Check for NaN or Inf values in logits
2. Make sure temperature is in a reasonable range (0.5-1.2)
3. Try setting a fixed seed with `--seed 42` for reproducible results