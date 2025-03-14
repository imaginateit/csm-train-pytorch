# Claude's Code Guide

This document contains important information about the CSM (Conversational Speech Model) codebase, including common commands, code style guidelines, and project structure.

## Project Structure

```
csm/
├── docs/                     # Documentation files
├── src/                      # Source code
│   └── csm/                  # Main package
│       ├── __init__.py       # Package initialization
│       ├── watermarking/     # Audio watermarking functionality
│       │   ├── __init__.py
│       │   ├── utils.py
│       │   └── silentcipher/ # Vendored silentcipher code
│       ├── models/           # Model definitions
│       ├── training/         # Training code
│       ├── inference/        # Inference utilities
│       ├── data/             # Data loading and processing
│       └── utils/            # Utility functions
├── tests/                    # Test suite
├── .gitignore                # Git ignore file
├── pyproject.toml            # Project configuration
├── LICENSE                   # License file
└── README.md                 # Project documentation
```

## Development Setup

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e ".[dev]"
```

## Development Commands

```bash
# Run tests
pytest tests/

# Format code with black and isort
black src/ tests/
isort src/ tests/

# Lint code with ruff
ruff src/ tests/

# Type check with mypy
mypy src/ tests/
```

## Code Style Guidelines

1. Use Python 3.11+ features
2. Follow PEP 8 guidelines
3. Maximum line length is 100 characters
4. Use type annotations for all function definitions
5. Document all public functions and classes with docstrings
6. Use black for code formatting
7. Organize imports with isort
8. Use ruff for linting
9. Maintain test coverage for new functionality

## Notes

1. The `silentcipher` dependency has been vendored into `src/csm/watermarking/silentcipher/` to avoid external Git dependencies
2. The project uses PyTorch for all model implementations, with MLX acceleration support for Apple Silicon
3. The codebase follows a modular design to facilitate future expansion

## MLX Acceleration Architecture

The MLX acceleration for Apple Silicon is implemented as a set of modular components:

1. `src/csm/cli/mlx_layers.py`: Core MLX layer implementations and utilities
   - MLXLayerNorm, MLXAttention, MLPMLP
   - MLXTransformerLayer and MLXTransformer
   - Utility functions for tensor conversion and masking

2. `src/csm/cli/mlx_embedding.py`: Embedding and sampling operations
   - MLXEmbedding class for text and audio embeddings
   - mlx_sample_topk and mlx_sample_categorical functions
   - Robust shape handling for embedding operations

3. `src/csm/cli/mlx_generation.py`: Frame generation pipeline
   - MLXFrameGenerator for pure MLX inference
   - Robust error handling and fallback mechanisms
   - Shape validation and correction

4. `src/csm/cli/mlx_wrapper.py`: Bridge between PyTorch and MLX
   - MLXWrapper for model parameter conversion
   - Hybrid PyTorch/MLX execution paths
   - Parameter validation and conversion

5. `src/csm/cli/generate_mlx.py`: Command-line interface
   - MLXGenerator for high-level control
   - Integration with watermarking and audio processing
   - Performance tracking and statistics

When running on Apple Silicon, the system attempts pure MLX execution for maximum performance, with automatic fallback to hybrid mode and finally to PyTorch if needed.