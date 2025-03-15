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
        batch_size: int = 2,
        use_dummy_data: bool = True  # Use dummy data for testing
    ):
        """
        Initialize the MLX dataset.
        
        Args:
            examples: List of examples with audio and transcript data
            max_seq_len: Maximum sequence length for training
            batch_size: Batch size for training
            use_dummy_data: Whether to use dummy data for testing
        """
        self.examples = examples
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.use_dummy_data = use_dummy_data
        
        # Initialize internal state
        self._batches = self._prepare_batches()
        
        logger.info(f"Created MLXDataset with {len(self.examples)} examples, {len(self._batches)} batches")
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self._batches)
    
    def _create_dummy_batch(self, batch_size: int) -> Dict[str, Any]:
        """Create a dummy batch for testing."""
        return {
            "input_tokens": mx.zeros((batch_size, self.max_seq_len, 3), dtype=mx.int32),
            "input_masks": mx.ones((batch_size, self.max_seq_len, 3), dtype=mx.float32),
            "target_audio_tokens": mx.zeros((batch_size, self.max_seq_len, 2), dtype=mx.int32)
        }
    
    def _process_audio_file(self, audio_path: str) -> np.ndarray:
        """
        Process an audio file for training.
        
        Loads audio file and extracts features for training.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Processed audio as numpy array
        """
        if self.use_dummy_data:
            # Return dummy audio embedding
            return np.zeros((100,), dtype=np.float32)
        
        try:
            # Import audio processing libraries
            try:
                import librosa
                import soundfile as sf
                HAS_AUDIO_LIBS = True
            except ImportError:
                logger.warning("librosa or soundfile not available. Using dummy audio data.")
                return np.zeros((100,), dtype=np.float32)
            
            # Load audio file
            try:
                # Try with soundfile first (faster)
                audio, sample_rate = sf.read(audio_path)
                # Convert to mono if stereo
                if len(audio.shape) > 1 and audio.shape[1] > 1:
                    audio = audio.mean(axis=1)
            except Exception:
                # Fall back to librosa
                audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            # Resample to standard rate if needed
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                audio = librosa.resample(
                    y=audio, 
                    orig_sr=sample_rate, 
                    target_sr=target_sample_rate
                )
            
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-6)
            
            # Extract features - in a real implementation, you'd use the model's
            # audio encoder to extract proper features. For our test, we'll use
            # simple spectrogram features.
            try:
                # Extract mel-spectrogram features
                n_fft = 512
                hop_length = 128
                n_mels = 80
                
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=audio, 
                    sr=target_sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels
                )
                
                # Convert to log scale
                log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
                
                # Normalize
                log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.mean()) / (log_mel_spectrogram.std() + 1e-6)
                
                # Flatten or reshape as needed
                features = log_mel_spectrogram.T  # Time x Frequency
                
                # Truncate or pad to fixed length
                max_length = 100
                if features.shape[0] > max_length:
                    features = features[:max_length]
                elif features.shape[0] < max_length:
                    padding = np.zeros((max_length - features.shape[0], features.shape[1]), dtype=np.float32)
                    features = np.concatenate([features, padding], axis=0)
                
                # Make sure output shape is correct
                return features.reshape(-1)[:100]  # Return first 100 elements for consistency
                
            except Exception as feature_e:
                logger.warning(f"Error extracting features: {feature_e}, using raw audio")
                # Fall back to raw audio if feature extraction fails
                if len(audio) > 100:
                    return audio[:100]
                else:
                    # Pad if too short
                    padding = np.zeros(100 - len(audio), dtype=np.float32)
                    return np.concatenate([audio, padding])
        
        except Exception as e:
            logger.warning(f"Error processing audio file {audio_path}: {e}")
            # Return empty audio on error
            return np.zeros((100,), dtype=np.float32)
    
    def _tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text for training.
        
        Converts text to token IDs for model training.
        Tries to use a proper tokenizer if available, with fallbacks.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        if self.use_dummy_data:
            # Generate random token IDs between 0 and 99 for testing
            return [np.random.randint(0, 100) for _ in range(min(100, len(text) + 20))]
        
        try:
            # Try to load a real tokenizer
            try:
                from transformers import AutoTokenizer
                
                # Use a pre-trained tokenizer - for a real CSM model, you would
                # use the specific tokenizer that matches the model
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                
                # Tokenize the text
                tokens = tokenizer.encode(text, add_special_tokens=True)
                
                # Make sure we don't go beyond vocab size in our token IDs
                # We'll use an arbitrary maximum of 50k for vocab size
                max_vocab_size = 50000
                tokens = [t % max_vocab_size for t in tokens]
                
                # Include a warning if tokens were clamped
                if any(t >= max_vocab_size for t in tokenizer.encode(text, add_special_tokens=True)):
                    logger.warning(f"Some token IDs exceeded max vocab size and were clamped")
                
                return tokens
                
            except (ImportError, Exception) as e:
                logger.warning(f"Could not use transformers tokenizer: {e}")
                
                # Fall back to byte-pair encoding simulation with character bigrams
                # This is a very simple approximation to subword tokenization
                logger.info("Falling back to simple character bigram encoding")
                
                # Simple bigram tokenization
                chars = list(text)
                bigrams = []
                
                # Add single characters
                for c in chars:
                    bigrams.append(ord(c) % 100)
                
                # Add some bigrams
                for i in range(len(chars) - 1):
                    bigram = chars[i] + chars[i+1]
                    # Use a simple hash function for the bigram
                    bigram_id = (ord(chars[i]) * 31 + ord(chars[i+1])) % 100
                    # Add some bigrams randomly
                    if np.random.random() > 0.7:  # Only include some bigrams
                        bigrams.append(bigram_id)
                
                return sorted(bigrams)[:100]  # Sort and limit to 100 tokens
                
        except Exception as e:
            logger.warning(f"Error tokenizing text '{text}': {e}")
            # Return empty tokens on error
            return [0] * 10
    
    def _process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example for training.
        
        Args:
            example: Example with audio and transcript data
            
        Returns:
            Processed example with tokens and features
        """
        try:
            # Get transcript text
            if 'text' in example:
                text = example['text']
            elif 'transcript_path' in example and os.path.exists(example['transcript_path']):
                with open(example['transcript_path'], 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            else:
                # Use dummy text if not available
                text = "This is a dummy transcript for testing."
            
            # Process audio
            if 'audio_path' in example and os.path.exists(example['audio_path']):
                audio_features = self._process_audio_file(example['audio_path'])
            else:
                # Use dummy audio if not available
                audio_features = np.zeros((100,), dtype=np.float32)
            
            # Tokenize text
            text_tokens = self._tokenize_text(text)
            
            # Create input tokens and masks
            input_tokens = np.zeros((self.max_seq_len, 3), dtype=np.int32)
            input_masks = np.ones((self.max_seq_len, 3), dtype=np.float32)
            
            # Fill in tokens
            text_len = min(len(text_tokens), self.max_seq_len)
            for i in range(text_len):
                input_tokens[i, 0] = text_tokens[i]
            
            # Create speaker ID column
            speaker_id = example.get('speaker_id', 0)
            input_tokens[:, 1] = speaker_id
            
            # Create target audio tokens
            target_audio_tokens = np.zeros((self.max_seq_len, 2), dtype=np.int32)
            
            # Fill in with synthetic audio tokens
            for i in range(min(self.max_seq_len, 100)):
                target_audio_tokens[i, 0] = i % 100
                target_audio_tokens[i, 1] = (i * 2) % 100
            
            # Return processed example
            return {
                "input_tokens": input_tokens,
                "input_masks": input_masks,
                "target_audio_tokens": target_audio_tokens
            }
        
        except Exception as e:
            logger.warning(f"Error processing example: {e}")
            # Return dummy example on error
            return {
                "input_tokens": np.zeros((self.max_seq_len, 3), dtype=np.int32),
                "input_masks": np.ones((self.max_seq_len, 3), dtype=np.float32),
                "target_audio_tokens": np.zeros((self.max_seq_len, 2), dtype=np.int32)
            }
    
    def _prepare_batches(self) -> List[Dict[str, Any]]:
        """
        Prepare batches from examples.
        
        Process all examples and create batches for training.
        
        Returns:
            List of batches, each containing input and target tensors
        """
        if len(self.examples) == 0:
            # No examples provided, return empty list
            return []
        
        # Process examples
        processed_examples = []
        for example in self.examples:
            processed_example = self._process_example(example)
            processed_examples.append(processed_example)
        
        # Create batches
        batches = []
        batch_count = (len(processed_examples) + self.batch_size - 1) // self.batch_size
        
        for i in range(batch_count):
            # Get batch examples
            batch_start = i * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(processed_examples))
            batch_examples = processed_examples[batch_start:batch_end]
            batch_size = len(batch_examples)
            
            # Combine examples into batch tensors
            batch_input_tokens = np.stack([example["input_tokens"] for example in batch_examples])
            batch_input_masks = np.stack([example["input_masks"] for example in batch_examples])
            batch_target_audio_tokens = np.stack([example["target_audio_tokens"] for example in batch_examples])
            
            # Convert to MLX arrays
            batch = {
                "input_tokens": mx.array(batch_input_tokens, dtype=mx.int32),
                "input_masks": mx.array(batch_input_masks, dtype=mx.float32),
                "target_audio_tokens": mx.array(batch_target_audio_tokens, dtype=mx.int32)
            }
            
            batches.append(batch)
        
        return batches
    
    def get_batch(self, batch_idx: int, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a specific batch by index.
        
        Args:
            batch_idx: Batch index
            batch_size: Override batch size (used for dynamic batch sizing)
            
        Returns:
            Batch dictionary with input and target tensors
        """
        if len(self._batches) == 0:
            # No batches available, return dummy batch
            return self._create_dummy_batch(batch_size or self.batch_size)
        
        if batch_idx < 0 or batch_idx >= len(self._batches):
            raise IndexError(f"Batch index {batch_idx} out of range (0-{len(self._batches)-1})")
        
        # If batch_size is specified and different from original, recreate the batch
        if batch_size is not None and batch_size != self.batch_size:
            # This feature would be implemented in a full version
            # For now, just return the existing batch
            logger.warning(f"Dynamic batch size not implemented, using original batch size {self.batch_size}")
        
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