"""Command-line interface for CSM."""

from .generate import main as generate_main
try:
    from .generate_mlx import main as generate_mlx_main
except ImportError:
    generate_mlx_main = None
from .verify import main as verify_main
