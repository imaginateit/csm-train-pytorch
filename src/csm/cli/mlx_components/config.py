"""
Configuration and constants for MLX acceleration.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Voice presets
VOICE_PRESETS = {
    "standard": {
        "temperature": 1.0,
        "topk": 25,
        "speaker": 0,
    },
    "warm": {
        "temperature": 1.2,
        "topk": 50,
        "speaker": 1,
    },
    "deep": {
        "temperature": 1.1,
        "topk": 40,
        "speaker": 2,
    },
    "bright": {
        "temperature": 1.2,
        "topk": 60,
        "speaker": 3,
    },
    "soft": {
        "temperature": 0.9,
        "topk": 30,
        "speaker": 4,
    },
    "energetic": {
        "temperature": 1.3,
        "topk": 70,
        "speaker": 5,
    },
    "calm": {
        "temperature": 0.8,
        "topk": 20,
        "speaker": 6,
    },
    "clear": {
        "temperature": 1.0,
        "topk": 45,
        "speaker": 7,
    },
    "resonant": {
        "temperature": 1.1,
        "topk": 35,
        "speaker": 8,
    },
    "authoritative": {
        "temperature": 1.0,
        "topk": 25,
        "speaker": 9,
    },
    "neutral": {
        "temperature": 1.0,
        "topk": 25,
        "speaker": 0,
    },
    "expressive": {
        "temperature": 1.3,
        "topk": 70,
        "speaker": 5,
    },
    "low": {
        "temperature": 0.8,
        "topk": 15,
        "speaker": 2,
    },
    "precise": {
        "temperature": 0.5,
        "topk": 5,
        "speaker": 0,
    }
}

@dataclass
class MLXConfig:
    """
    Configuration for MLX acceleration.
    
    Attributes:
        max_seq_len: Maximum sequence length
        rope_theta: RoPE theta parameter
        use_bfloat16: Whether to use bfloat16 precision
        epsilon: Small constant for numerical stability
        verbose: Whether to output verbose logs
        debug: Whether to output debug logs
        temperature: Sampling temperature
        topk: Top-k sampling parameter
        no_watermark: Whether to disable audio watermarking
    """
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    use_bfloat16: bool = False
    epsilon: float = 1e-5
    verbose: bool = False
    debug: bool = False
    temperature: Optional[float] = None
    topk: Optional[int] = None
    no_watermark: bool = False
    
    def __post_init__(self):
        # Initialize from environment variables if not set
        if self.verbose is False:
            self.verbose = os.environ.get("MLX_VERBOSE", "0") == "1"
        if self.debug is False:
            self.debug = os.environ.get("MLX_DEBUG", "0") == "1"

def get_voice_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get the parameters for a specific voice preset.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Dictionary with preset parameters
    """
    if preset_name not in VOICE_PRESETS:
        print(f"Warning: Unknown voice preset '{preset_name}', falling back to 'standard'")
        preset_name = "standard"
    
    return VOICE_PRESETS[preset_name].copy()