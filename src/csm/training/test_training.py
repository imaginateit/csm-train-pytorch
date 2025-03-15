"""Test suite for CSM training functionality."""

import os
import tempfile
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

from csm.models.model import Model, ModelArgs
from csm.training.trainer import CSMTrainer
from csm.training.data import (
    CSMDataset, 
    TrainingExample,
    CSMDataProcessor
)
from csm.training.utils import compute_loss


class MockTextTokenizer:
    """Mock text tokenizer for testing."""
    
    def encode(self, text):
        """Mock encode method."""
        # Return a list of integers with length proportional to text length
        return [1] * (len(text) // 5 + 1)


class MockAudioTokenizer:
    """Mock audio tokenizer for testing."""
    
    def __init__(self):
        """Initialize the mock tokenizer."""
        self.sample_rate = 24000
    
    def encode(self, audio):
        """Mock encode method."""
        # Return a tensor of shape [32, T] where T is proportional to audio length
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 1:
                t = audio.shape[0] // 1000 + 1
            else:
                t = audio.shape[-1] // 1000 + 1
        else:
            t = 5  # Default size for testing
        return torch.ones(32, t, dtype=torch.long)


def create_mock_model() -> Model:
    """Create a minimal mock CSM model for testing."""
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    
    model = Model(model_args)
    
    # Initialize with minimal parameters for testing
    # This won't be a functional model, but will allow us to test the training flow
    model.text_embeddings = torch.nn.Embedding(model_args.text_vocab_size, 2048)
    model.audio_embeddings = torch.nn.Embedding(
        model_args.audio_vocab_size * model_args.audio_num_codebooks, 2048
    )
    model.codebook0_head = torch.nn.Linear(2048, model_args.audio_vocab_size)
    model.audio_head = torch.nn.Parameter(
        torch.empty(model_args.audio_num_codebooks - 1, 1024, model_args.audio_vocab_size)
    )
    
    # Create backbone and decoder with just enough functionality to pass through the pipeline
    class MockTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2048, 2048)
            self.max_seq_len = 2048
        
        def setup_caches(self, max_batch_size, dtype, decoder_max_seq_len=None):
            pass
        
        def reset_caches(self):
            pass
        
        def caches_are_enabled(self):
            return True
        
        def forward(self, x, input_pos=None, mask=None):
            return self.linear(x)
    
    model.backbone = MockTransformer()
    model.decoder = MockTransformer()
    model.projection = torch.nn.Linear(2048, 1024)
    
    # Register buffer for causal mask
    model.register_buffer("backbone_causal_mask", torch.tril(torch.ones(10, 10, dtype=torch.bool)))
    model.register_buffer("decoder_causal_mask", torch.tril(torch.ones(32, 32, dtype=torch.bool)))
    
    model._index_causal_mask = lambda mask, input_pos: mask
    
    return model


def create_test_dataset() -> tuple:
    """Create a small test dataset."""
    # Create sample training examples
    examples = []
    for i in range(5):
        # Create a dummy audio tensor (1 second)
        audio = torch.randn(24000)
        text = f"This is test sentence {i}."
        examples.append(TrainingExample(text=text, audio=audio, speaker_id=0))
    
    # Create mock tokenizers
    text_tokenizer = MockTextTokenizer()
    audio_tokenizer = MockAudioTokenizer()
    
    # Convert to contextual examples
    contextual_examples = [{"context": [], "target": ex} for ex in examples]
    
    # Create dataset
    dataset = CSMDataset(
        contextual_examples,
        text_tokenizer,
        audio_tokenizer,
        max_seq_len=10  # Small for testing
    )
    
    return dataset, examples


def create_sample_audio_files(temp_dir: Path) -> Dict[str, Path]:
    """Create sample audio files and transcripts for testing."""
    audio_dir = temp_dir / "audio"
    transcript_dir = temp_dir / "transcripts"
    
    audio_dir.mkdir(exist_ok=True)
    transcript_dir.mkdir(exist_ok=True)
    
    # Create 2 sample files
    paths = {}
    for i in range(2):
        # Create dummy audio file (1 second)
        audio_path = audio_dir / f"sample{i}.wav"
        audio_tensor = torch.zeros(1, 24000)
        
        # Save audio file using torchaudio
        import torchaudio
        torchaudio.save(str(audio_path), audio_tensor, 24000)
        
        # Create transcript file
        transcript_path = transcript_dir / f"sample{i}.txt"
        with open(transcript_path, "w") as f:
            f.write(f"This is test sentence {i}.")
        
        paths[f"audio_{i}"] = audio_path
        paths[f"transcript_{i}"] = transcript_path
    
    paths["audio_dir"] = audio_dir
    paths["transcript_dir"] = transcript_dir
    
    return paths


def test_data_processor():
    """Test the CSMDataProcessor class."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        paths = create_sample_audio_files(temp_dir)
        
        # Create a data processor
        data_processor = CSMDataProcessor()
        
        # Process the first file
        examples = data_processor.prepare_from_audio_file(
            paths["audio_0"],
            paths["transcript_0"],
            speaker_id=0
        )
        
        # Verify we got examples
        assert len(examples) > 0
        assert isinstance(examples[0], TrainingExample)
        assert examples[0].text == "This is test sentence 0."
        assert examples[0].speaker_id == 0
        assert isinstance(examples[0].audio, torch.Tensor)
        assert examples[0].audio.shape[0] > 0


def test_dataset():
    """Test the CSMDataset class."""
    dataset, _ = create_test_dataset()
    
    # Verify dataset length
    assert len(dataset) == 5
    
    # Test __getitem__
    item = dataset[0]
    assert "input_tokens" in item
    assert "input_masks" in item
    assert "target_audio_tokens" in item
    
    # Verify shapes
    assert item["input_tokens"].dim() == 2  # [seq_len, 33]
    assert item["input_masks"].dim() == 2  # [seq_len, 33]
    assert item["target_audio_tokens"].dim() == 2  # [seq_len, 32]


def test_compute_loss():
    """Test the compute_loss function."""
    # Create a mock model
    model = create_mock_model()
    
    # Create sample inputs
    batch_size = 2
    seq_len = 5
    input_tokens = torch.randint(0, 100, (batch_size, seq_len, 33))
    input_masks = torch.ones((batch_size, seq_len, 33), dtype=torch.bool)
    target_audio_tokens = torch.randint(0, 100, (batch_size, seq_len, 32))
    
    # Compute loss
    loss, loss_components = compute_loss(
        model,
        input_tokens,
        input_masks,
        target_audio_tokens
    )
    
    # Verify we got a loss value
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0  # Loss should be positive
    
    # Verify loss components
    assert "semantic_loss" in loss_components
    assert isinstance(loss_components["semantic_loss"], torch.Tensor)


def test_trainer_initialization():
    """Test trainer initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create a mock trainer directly instead of loading from file
        # This avoids state_dict mismatch issues
        from csm.training.trainer import CSMTrainer
        
        # Create a minimal trainer with mock properties
        trainer = CSMTrainer(
            model_path="",  # Empty path
            output_dir=str(temp_dir),
            device="cpu"
        )
        
        # Set model manually to avoid loading from file
        trainer.model = create_mock_model()
        
        # Verify trainer was initialized
        assert trainer.model is not None
        assert trainer.output_dir == temp_dir
        
        # Test optimizer preparation
        trainer.prepare_optimizer()
        assert trainer.optimizer is not None


def run_tests():
    """Run all tests."""
    print("Testing CSM training functionality...")
    
    print("1. Testing data processor...")
    test_data_processor()
    print("✅ Data processor test passed")
    
    print("2. Testing dataset...")
    test_dataset()
    print("✅ Dataset test passed")
    
    print("3. Testing loss computation...")
    test_compute_loss()
    print("✅ Loss computation test passed")
    
    print("4. Testing trainer initialization...")
    test_trainer_initialization()
    print("✅ Trainer initialization test passed")
    
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()