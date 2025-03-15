"""
Data handling utilities for CSM training.

This module provides dataset classes and functions for preparing
training data for CSM models.
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Create a dummy MLX module
    class DummyMX:
        def __getattr__(self, name):
            raise ImportError("MLX is not available")

    mx = DummyMX()

logger = logging.getLogger(__name__)


class MLXDataset:
    """
    Dataset class for MLX-based training.
    
    This class prepares audio and transcript data for training with MLX,
    handling batching, tokenization, and formatting.
    """
    
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        max_seq_len: int = 2048,
        batch_size: int = 2
    ):
        """
        Initialize the MLX dataset.
        
        Args:
            examples: List of examples with audio and transcript data
            max_seq_len: Maximum sequence length for training
            batch_size: Batch size for training
        """
        self.examples = examples
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        # Initialize internal state
        self._batches = self._prepare_batches()
        
        logger.info(f"Created MLXDataset with {len(self.examples)} examples, {len(self._batches)} batches")
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self._batches)
    
    def _prepare_batches(self) -> List[Dict[str, Any]]:
        """
        Prepare batches from examples.
        
        This is a stub implementation that creates dummy batches for testing.
        In a real implementation, this would process audio files, tokenize text,
        and create properly formatted batches.
        
        Returns:
            List of batches, each containing input and target tensors
        """
        # Create a minimal batch structure for testing
        dummy_batches = []
        
        batch_count = (len(self.examples) + self.batch_size - 1) // self.batch_size
        for i in range(batch_count):
            # Create dummy tensors
            batch_size = min(self.batch_size, len(self.examples) - i * self.batch_size)
            
            dummy_batch = {
                "input_tokens": mx.zeros((batch_size, self.max_seq_len, 3), dtype=mx.int32),
                "input_masks": mx.ones((batch_size, self.max_seq_len, 3), dtype=mx.float32),
                "target_audio_tokens": mx.zeros((batch_size, self.max_seq_len, 2), dtype=mx.int32)
            }
            
            dummy_batches.append(dummy_batch)
        
        return dummy_batches
    
    def get_batch(self, batch_idx: int, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a specific batch by index.
        
        Args:
            batch_idx: Batch index
            batch_size: Override batch size (ignored in stub implementation)
            
        Returns:
            Batch dictionary with input and target tensors
        """
        if batch_idx < 0 or batch_idx >= len(self._batches):
            raise IndexError(f"Batch index {batch_idx} out of range (0-{len(self._batches)-1})")
        
        return self._batches[batch_idx]


class PyTorchDataset:
    """
    Dataset class for PyTorch-based training.
    
    This class prepares audio and transcript data for training with PyTorch,
    implementing the standard PyTorch Dataset interface.
    """
    
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        max_seq_len: int = 2048
    ):
        """
        Initialize the PyTorch dataset.
        
        Args:
            examples: List of examples with audio and transcript data
            max_seq_len: Maximum sequence length for training
        """
        self.examples = examples
        self.max_seq_len = max_seq_len
        
        logger.info(f"Created PyTorchDataset with {len(self.examples)} examples")
    
    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a specific example by index.
        
        This is a stub implementation that returns dummy tensors for testing.
        In a real implementation, this would process audio files, tokenize text,
        and create properly formatted tensors.
        
        Args:
            idx: Example index
            
        Returns:
            Dictionary with input and target tensors
        """
        # Return a dummy item for testing
        import torch
        
        return {
            "input_tokens": torch.zeros((self.max_seq_len, 3), dtype=torch.long),
            "input_masks": torch.ones((self.max_seq_len, 3), dtype=torch.float),
            "target_audio_tokens": torch.zeros((self.max_seq_len, 2), dtype=torch.long)
        }


def prepare_training_data(
    audio_dir: str,
    transcript_dir: str,
    alignment_dir: Optional[str] = None,
    speaker_id: int = 0,
    val_split: float = 0.1,
    max_seq_len: int = 2048,
    context_turns: int = 2
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Prepare training and validation data from audio and transcript files.
    
    Args:
        audio_dir: Directory containing audio files
        transcript_dir: Directory containing transcript files
        alignment_dir: Optional directory containing alignment files
        speaker_id: Speaker ID to use
        val_split: Validation split ratio
        max_seq_len: Maximum sequence length
        context_turns: Number of context turns to include
        
    Returns:
        Tuple of (train_examples, val_examples)
    """
    # Stub implementation that creates dummy examples
    logger.info(f"Preparing training data from {audio_dir} and {transcript_dir}")
    
    # Get list of audio files
    audio_files = list(Path(audio_dir).glob("*.wav"))
    
    # Create dummy examples
    examples = []
    for i, audio_file in enumerate(audio_files):
        # Get matching transcript file
        transcript_file = Path(transcript_dir) / f"{audio_file.stem}.txt"
        
        # Skip if transcript doesn't exist
        if not transcript_file.exists():
            logger.warning(f"No transcript found for {audio_file}, skipping")
            continue
        
        # Create dummy example
        example = {
            "audio_path": str(audio_file),
            "transcript_path": str(transcript_file),
            "speaker_id": speaker_id,
            "id": f"example_{i}"
        }
        
        examples.append(example)
    
    # Split into train and validation
    random.shuffle(examples)
    val_size = int(len(examples) * val_split)
    train_examples = examples[val_size:]
    val_examples = examples[:val_size]
    
    logger.info(f"Created {len(train_examples)} training and {len(val_examples)} validation examples")
    
    return train_examples, val_examples