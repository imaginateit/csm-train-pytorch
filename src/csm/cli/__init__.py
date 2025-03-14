"""Command-line interface for CSM."""

from .generate import main as generate_main
from .generate_cpu import main as generate_cpu_main
from .generate_mlx import main as generate_mlx_main
from .verify import main as verify_main