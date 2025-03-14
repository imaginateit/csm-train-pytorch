"""
Configuration and constants for MLX acceleration.
"""

import os
from typing import Dict, Any

# Voice presets
VOICE_PRESETS = {
    "standard": {
        "temperature": 1.0,
        "topk": 25,
    },
    "warm": {
        "temperature": 1.2,
        "topk": 50,
    },
    "expressive": {
        "temperature": 1.3,
        "topk": 70,
    },
    "low": {
        "temperature": 0.8,
        "topk": 15,
    },
    "precise": {
        "temperature": 0.5,
        "topk": 5,
    }
}

# MLX acceleration parameters
MLX_CONFIG = {
    "max_seq_len": 2048,
    "rope_theta": 10000.0,
    "use_bfloat16": False,
    "epsilon": 1e-5,
    "verbose": os.environ.get("MLX_VERBOSE", "0") == "1",
    "debug": os.environ.get("MLX_DEBUG", "0") == "1",
}

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