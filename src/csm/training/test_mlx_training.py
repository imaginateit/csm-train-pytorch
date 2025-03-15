"""
Test suite for the MLX training implementation.

This module provides comprehensive tests for the MLX training implementation
to ensure compatibility, robustness, and correctness.
"""

import os
import sys
import time
import unittest
import numpy as np
import tempfile
import json
import shutil
from pathlib import Path

# Skip tests if MLX is not available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipIf(not HAS_MLX, "MLX not available")
class TestMLXTrainingCore(unittest.TestCase):
    """Test core functionality of MLX training tools."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal MLX model for testing
        class MinimalMLXModel:
            def __init__(self):
                # Create minimal backbone
                class MinimalBackbone:
                    def __init__(self):
                        self.hidden_size = 16
                        self.layers = []
                    
                    def __call__(self, hidden_states, input_pos=None, mask=None):
                        # Simple pass-through
                        return hidden_states
                    
                    def parameters(self):
                        return {"test_param": mx.zeros((1, 1))}
                
                # Create minimal decoder
                class MinimalDecoder:
                    def __init__(self):
                        self.hidden_size = 16
                        self.layers = []
                    
                    def __call__(self, hidden_states, input_pos=None, mask=None):
                        # Simple pass-through
                        return hidden_states
                    
                    def parameters(self):
                        return {"test_param": mx.zeros((1, 1))}
                
                # Create minimal embedding
                class MinimalEmbedding:
                    def __init__(self):
                        self.embed_dim = 16
                    
                    def embed_tokens(self, tokens):
                        # Create dummy embeddings
                        batch_size, seq_len, codebooks = tokens.shape
                        return mx.zeros((batch_size, seq_len, self.embed_dim))
                    
                    def parameters(self):
                        return {"text_embeddings": mx.zeros((1, 16)), "audio_embeddings": mx.zeros((1, 16))}
                
                # Set up model components
                self.backbone = MinimalBackbone()
                self.decoder = MinimalDecoder()
                self.embedding = MinimalEmbedding()
                self.codebook0_head = mx.zeros((16, 16))  # Dummy weight matrix
                self.audio_head = [mx.zeros((16, 16))]    # Dummy weight matrix
                self.projection = mx.zeros((16, 16))      # Dummy weight matrix
                self.args = type('', (), {})()
                self.args.audio_vocab_size = 16
                self.args.audio_num_codebooks = 2
            
            def parameters(self):
                """Get model parameters."""
                params = {}
                
                # Add backbone parameters
                backbone_params = self.backbone.parameters()
                for name, param in backbone_params.items():
                    params[f"backbone.{name}"] = param
                
                # Add decoder parameters
                decoder_params = self.decoder.parameters() 
                for name, param in decoder_params.items():
                    params[f"decoder.{name}"] = param
                
                # Add embedding parameters
                embedding_params = self.embedding.parameters()
                for name, param in embedding_params.items():
                    params[name] = param
                
                # Add other parameters
                params["codebook0_head"] = self.codebook0_head
                params["audio_head.0"] = self.audio_head[0]
                params["projection"] = self.projection
                
                return params
            
            def update(self, params_dict):
                """Update model parameters."""
                # Update backbone parameters
                backbone_params = {}
                for name, param in params_dict.items():
                    if name.startswith("backbone."):
                        backbone_name = name[len("backbone."):]
                        backbone_params[backbone_name] = param
                
                # Update decoder parameters
                decoder_params = {}
                for name, param in params_dict.items():
                    if name.startswith("decoder."):
                        decoder_name = name[len("decoder."):]
                        decoder_params[decoder_name] = param
                
                # Update direct parameters
                for name, param in params_dict.items():
                    if name == "codebook0_head":
                        self.codebook0_head = param
                    elif name == "audio_head.0":
                        self.audio_head[0] = param
                    elif name == "projection":
                        self.projection = param
                    elif name == "text_embeddings":
                        pass  # Would update embedding
                    elif name == "audio_embeddings":
                        pass  # Would update embedding
            
            def _embed_tokens(self, tokens):
                """Embed tokens using the embedding helper."""
                return self.embedding.embed_tokens(tokens)
            
            def embed_tokens(self, tokens):
                """Public wrapper for _embed_tokens."""
                return self._embed_tokens(tokens)
        
        # Create the model instance
        self.model = MinimalMLXModel()
        
        # Create dummy optimizer
        class MinimalOptimizer:
            def __init__(self):
                self.state = {}
                self.learning_rate = 1e-4
            
            def update(self, model, grads):
                # Just a no-op for testing
                pass
        
        self.optimizer = MinimalOptimizer()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_compute_loss_mlx(self):
        """Test compute_loss_mlx with minimal inputs."""
        from csm.training.utils import compute_loss_mlx
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 4
        
        input_tokens = mx.zeros((batch_size, seq_len, 3), dtype=mx.int32)
        input_masks = mx.ones((batch_size, seq_len, 3), dtype=mx.float32)
        target_audio_tokens = mx.zeros((batch_size, seq_len, 2), dtype=mx.int32)
        
        # Compute loss
        total_loss, losses = compute_loss_mlx(
            self.model,
            input_tokens,
            input_masks,
            target_audio_tokens,
            semantic_weight=1.0,
            acoustic_weight=1.0
        )
        
        # Verify loss is a valid MLX array with finite value
        self.assertIsNotNone(total_loss)
        self.assertTrue(hasattr(total_loss, 'shape'))
        
        # Verify losses dictionary contains expected keys
        self.assertIn("semantic_loss", losses)
        self.assertIn("acoustic_loss", losses)
        
        # Verify loss values are finite and valid
        import math
        total_loss_val = float(total_loss.item())
        self.assertFalse(math.isnan(total_loss_val))
        self.assertFalse(math.isinf(total_loss_val))
    
    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        from csm.training.utils import save_checkpoint_mlx, load_checkpoint_mlx
        
        # Create checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.safetensors")
        
        # Save checkpoint
        saved_path = save_checkpoint_mlx(
            self.model,
            self.optimizer,
            epoch=1,
            global_step=100,
            loss=0.5,
            save_dir=self.temp_dir,
            name="test_checkpoint"
        )
        
        # Verify checkpoint was saved
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.exists(saved_path))
        
        # Check metadata file
        metadata_path = saved_path.replace(".safetensors", "_metadata.json")
        self.assertTrue(os.path.exists(metadata_path))
        
        # Load checkpoint
        metadata = load_checkpoint_mlx(
            saved_path,
            self.model,
            self.optimizer
        )
        
        # Verify metadata
        self.assertEqual(metadata["epoch"], 1)
        self.assertEqual(metadata["global_step"], 100)
        self.assertAlmostEqual(metadata["loss"], 0.5, places=3)
        self.assertTrue(metadata.get("model_loaded", False))
    
    def test_custom_mlx_dataset(self):
        """Test the custom MLX dataset implementation."""
        from csm.cli.train_mlx import MLXDataset
        
        # Create example data
        class DummyData:
            def __init__(self, text="Hello", audio=None, speaker_id=0):
                self.text = text
                self.audio = audio or mx.zeros((1000,))
                self.speaker_id = speaker_id
                
        # Create dummy examples
        examples = [
            {"context": [], "target": DummyData("Hello", speaker_id=0)},
            {"context": [DummyData("How are you?", speaker_id=1)], "target": DummyData("I'm fine", speaker_id=0)}
        ]
        
        # Create dataset
        dataset = MLXDataset(examples, max_seq_len=32)
        
        # Verify dataset length
        self.assertEqual(len(dataset), 2)
        
        # Get batch
        batch = dataset.get_batch(0, 2)
        
        # Verify batch structure
        self.assertIn("input_tokens", batch)
        self.assertIn("input_masks", batch)
        self.assertIn("target_audio_tokens", batch)
        
        # Verify batch shapes
        self.assertTrue(hasattr(batch["input_tokens"], 'shape'))
        self.assertTrue(hasattr(batch["input_masks"], 'shape'))
        self.assertTrue(hasattr(batch["target_audio_tokens"], 'shape'))


@unittest.skipIf(not HAS_MLX or not HAS_TORCH, "MLX or PyTorch not available")
class TestMLXTrainer(unittest.TestCase):
    """Test the MLX trainer implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal checkpoint file
        self.safetensors_path = os.path.join(self.temp_dir, "model.safetensors")
        
        # Create minimal safetensors file
        import safetensors.numpy
        dummy_weights = {"test": np.zeros((1, 1))}
        safetensors.numpy.save_file(dummy_weights, self.safetensors_path)
        
        # Create metadata file
        metadata = {
            "epoch": 0,
            "global_step": 0,
            "loss": 1.0,
            "model_path": self.safetensors_path
        }
        
        with open(os.path.join(self.temp_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization with minimal data."""
        from csm.training.mlx_trainer import CSMMLXTrainer
        
        # Initialize trainer
        trainer = CSMMLXTrainer(
            model_path=self.safetensors_path,
            output_dir=self.temp_dir
        )
        
        # Verify trainer attributes
        self.assertEqual(trainer.model_path, self.safetensors_path)
        self.assertEqual(str(trainer.output_dir), self.temp_dir)
        self.assertEqual(trainer.epoch, 0)
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.best_loss, float("inf"))
    
    def test_optimizer_preparation(self):
        """Test optimizer preparation."""
        from csm.training.mlx_trainer import CSMMLXTrainer
        
        # Initialize trainer
        trainer = CSMMLXTrainer(
            model_path=self.safetensors_path,
            output_dir=self.temp_dir
        )
        
        # Prepare optimizer
        trainer.prepare_optimizer()
        
        # Verify optimizer was created
        self.assertIsNotNone(trainer.optimizer)
    
    def test_train_step(self):
        """Test a single training step."""
        from csm.training.mlx_trainer import CSMMLXTrainer
        
        # Initialize trainer
        trainer = CSMMLXTrainer(
            model_path=self.safetensors_path,
            output_dir=self.temp_dir
        )
        
        # Prepare optimizer
        trainer.prepare_optimizer()
        
        # Create dummy batch
        batch = {
            "input_tokens": mx.zeros((2, 4, 3), dtype=mx.int32),
            "input_masks": mx.ones((2, 4, 3), dtype=mx.float32),
            "target_audio_tokens": mx.zeros((2, 4, 2), dtype=mx.int32)
        }
        
        # Execute train step
        loss = trainer.train_step(batch)
        
        # Verify loss is a valid value
        self.assertIsNotNone(loss)
        
        # Verify loss is finite
        import math
        loss_val = float(loss.item()) if hasattr(loss, 'item') else float(loss)
        self.assertFalse(math.isnan(loss_val))
        self.assertFalse(math.isinf(loss_val))


class TestMLXDataProcessor(unittest.TestCase):
    """Test data processing for MLX training."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy audio file
        self.audio_dir = os.path.join(self.temp_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Create dummy transcript file
        self.transcript_dir = os.path.join(self.temp_dir, "transcript")
        os.makedirs(self.transcript_dir, exist_ok=True)
        
        # Create dummy alignment file
        self.alignment_dir = os.path.join(self.temp_dir, "alignment")
        os.makedirs(self.alignment_dir, exist_ok=True)
        
        # Create dummy output dir
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a dummy 1-second WAV file (silence)
        sample_rate = 24000
        dummy_audio = np.zeros(sample_rate, dtype=np.float32)
        
        # Create dummy WAV files
        try:
            import soundfile as sf
            
            # Create test files
            self.audio_files = []
            for i in range(3):
                filename = f"test_{i}.wav"
                filepath = os.path.join(self.audio_dir, filename)
                sf.write(filepath, dummy_audio, sample_rate)
                self.audio_files.append(filepath)
                
                # Create matching transcript
                with open(os.path.join(self.transcript_dir, f"test_{i}.txt"), "w") as f:
                    f.write(f"This is test audio number {i}.")
                
                # Create matching alignment
                with open(os.path.join(self.alignment_dir, f"test_{i}.json"), "w") as f:
                    # Simple mock alignment
                    alignment = {
                        "words": [
                            {"word": "This", "start": 0.0, "end": 0.1},
                            {"word": "is", "start": 0.1, "end": 0.2},
                            {"word": "test", "start": 0.2, "end": 0.3},
                            {"word": "audio", "start": 0.3, "end": 0.4},
                            {"word": f"number", "start": 0.4, "end": 0.5},
                            {"word": f"{i}.", "start": 0.5, "end": 0.6}
                        ]
                    }
                    json.dump(alignment, f)
                    
            self.HAS_AUDIO = True
        except ImportError:
            self.HAS_AUDIO = False
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipIf(not HAS_MLX, "MLX not available")
    def test_data_processor(self):
        """Test the CSMDataProcessor."""
        from csm.data import CSMDataProcessor
        
        # Skip if no audio support
        if not self.HAS_AUDIO:
            self.skipTest("Audio processing libraries not available")
        
        # Create processor
        processor = CSMDataProcessor()
        
        # Process a single file
        audio_file = self.audio_files[0]
        transcript_file = os.path.join(self.transcript_dir, os.path.basename(audio_file).replace(".wav", ".txt"))
        alignment_file = os.path.join(self.alignment_dir, os.path.basename(audio_file).replace(".wav", ".json"))
        
        # Process file
        examples = processor.prepare_from_audio_file(
            audio_file,
            transcript_file,
            speaker_id=0,
            alignment_file=alignment_file
        )
        
        # Check examples
        self.assertGreater(len(examples), 0)
    
    @unittest.skipIf(not HAS_MLX, "MLX not available")
    def test_context_generator(self):
        """Test the context generator."""
        from csm.data import ContextualExampleGenerator
        
        # Create examples
        class DummyData:
            def __init__(self, text="Hello", audio=None, speaker_id=0):
                self.text = text
                self.audio = audio or np.zeros(1000)
                self.speaker_id = speaker_id
                
        # Create example sequence
        examples = [
            DummyData("Hello", speaker_id=0),
            DummyData("How are you?", speaker_id=1),
            DummyData("I'm fine", speaker_id=0),
            DummyData("That's good", speaker_id=1)
        ]
        
        # Create generator
        generator = ContextualExampleGenerator(max_context_turns=2)
        
        # Generate contexts
        contextual_examples = generator.create_contextual_examples(examples)
        
        # Verify contexts
        self.assertEqual(len(contextual_examples), len(examples))
        
        # Check last example has context
        last_example = contextual_examples[-1]
        self.assertGreater(len(last_example["context"]), 0)


# Only run if this module is being executed directly
if __name__ == "__main__":
    unittest.main()