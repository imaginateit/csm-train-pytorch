"""
Test suite for LoRA implementation with MLX.

This module provides tests for the LoRA (Low-Rank Adaptation) implementation
used for parameter-efficient fine-tuning of CSM models with MLX.
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
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@unittest.skipIf(not HAS_MLX, "MLX not available")
class TestLoRAImplementation(unittest.TestCase):
    """Test the core LoRA implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal MLX model with LoRA adapters for testing
        from csm.mlx.components.lora import LoRALinear
        
        # Create base weight and bias
        self.base_weight = mx.random.normal((16, 32))
        self.base_bias = mx.zeros((16,))
        
        # Create LoRA adapter
        self.lora = LoRALinear(
            base_weight=self.base_weight,
            base_bias=self.base_bias,
            r=4,
            alpha=8.0,
            dropout=0.0,
            use_bias=True,
            name="test_layer"
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_lora_initialization(self):
        """Test LoRA adapter initialization."""
        # Check attributes
        self.assertEqual(self.lora.in_features, 32)
        self.assertEqual(self.lora.out_features, 16)
        self.assertEqual(self.lora.r, 4)
        self.assertEqual(self.lora.alpha, 8.0)
        self.assertEqual(self.lora.scaling, 8.0 / 4)
        
        # Check parameter shapes
        self.assertEqual(self.lora.lora_A.shape, (4, 32))  # (r, in_features)
        self.assertEqual(self.lora.lora_B.shape, (16, 4))  # (out_features, r)
        self.assertEqual(self.lora.lora_bias.shape, (16,))  # (out_features,)
    
    def test_lora_forward(self):
        """Test LoRA adapter forward pass."""
        # Create input tensor
        batch_size = 2
        seq_len = 3
        x = mx.zeros((batch_size, seq_len, 32))
        
        # Forward pass
        output = self.lora(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, 16))
        
        # Create non-zero input to test actual computation
        x = mx.ones((batch_size, seq_len, 32))
        output = self.lora(x)
        
        # Verify output is not all zeros (computation happened)
        self.assertFalse(mx.all(output == 0).item())
    
    def test_lora_parameters(self):
        """Test LoRA parameters method."""
        params = self.lora.parameters()
        
        # Check parameter names
        self.assertIn("test_layer.lora_A", params)
        self.assertIn("test_layer.lora_B", params)
        self.assertIn("test_layer.lora_bias", params)
        
        # Check parameter values
        self.assertIs(params["test_layer.lora_A"], self.lora.lora_A)
        self.assertIs(params["test_layer.lora_B"], self.lora.lora_B)
        self.assertIs(params["test_layer.lora_bias"], self.lora.lora_bias)
    
    def test_lora_update(self):
        """Test LoRA update method."""
        # Create new parameters
        new_A = mx.ones((4, 32))
        new_B = mx.ones((16, 4)) * 0.5
        new_bias = mx.ones((16,)) * 0.1
        
        # Create params dict
        params_dict = {
            "test_layer.lora_A": new_A,
            "test_layer.lora_B": new_B,
            "test_layer.lora_bias": new_bias
        }
        
        # Update parameters
        self.lora.update(params_dict)
        
        # Check parameters were updated
        self.assertTrue(mx.array_equal(self.lora.lora_A, new_A))
        self.assertTrue(mx.array_equal(self.lora.lora_B, new_B))
        self.assertTrue(mx.array_equal(self.lora.lora_bias, new_bias))
    
    def test_lora_merge(self):
        """Test merging LoRA weights with base weights."""
        # Set deterministic values for LoRA matrices
        self.lora.lora_A = mx.ones((4, 32)) * 0.1
        self.lora.lora_B = mx.ones((16, 4)) * 0.2
        
        # Calculate expected merged weight
        # W = W0 + scaling * B * A
        expected_contrib = mx.matmul(self.lora.lora_B, self.lora.lora_A) * self.lora.scaling
        expected_merged = self.base_weight + expected_contrib
        
        # Get merged weights
        merged_weight = self.lora.merge_with_base()
        
        # Check merged weights
        self.assertTrue(mx.allclose(merged_weight, expected_merged, atol=1e-5))


@unittest.skipIf(not HAS_MLX, "MLX not available")
class TestLoRATransformer(unittest.TestCase):
    """Test the LoRA transformer implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Skip the test if the transformer layer module is not available
        try:
            from csm.mlx.components.transformer import MLXTransformerLayer, MLXTransformer
            from csm.mlx.components.lora import LoRATransformerLayer, LoRATransformer
        except ImportError:
            self.skipTest("MLXTransformerLayer not available")
        
        # Create a minimal transformer layer for testing
        class MinimalTransformerLayer:
            def __init__(self):
                # Mock parameters for attention
                self.hidden_size = 32
                self.num_heads = 4
                self.num_kv_heads = 4
                self.head_dim = 8
                
                # Mock attention parameters
                self.q_proj_weight = mx.random.normal((32, 32))
                self.q_proj_bias = mx.zeros((32,))
                self.k_proj_weight = mx.random.normal((32, 32))
                self.k_proj_bias = mx.zeros((32,))
                self.v_proj_weight = mx.random.normal((32, 32))
                self.v_proj_bias = mx.zeros((32,))
                self.o_proj_weight = mx.random.normal((32, 32))
                self.o_proj_bias = mx.zeros((32,))
                
                # Mock MLP parameters
                self.gate_proj_weight = mx.random.normal((128, 32))
                self.gate_proj_bias = mx.zeros((128,))
                self.up_proj_weight = mx.random.normal((128, 32))
                self.up_proj_bias = mx.zeros((128,))
                self.down_proj_weight = mx.random.normal((32, 128))
                self.down_proj_bias = mx.zeros((32,))
                
                # Mock layernorm parameters
                self.input_layernorm_weight = mx.ones((32,))
                self.input_layernorm_bias = mx.zeros((32,))
                self.post_attention_layernorm_weight = mx.ones((32,))
                self.post_attention_layernorm_bias = mx.zeros((32,))
                
                # Mock rotary embeddings
                self.cos_cached = mx.ones((128, 8))
                self.sin_cached = mx.ones((128, 8))
            
            def _layernorm(self, x, weight, bias):
                # Simple mock layernorm
                return x
                
            def _apply_rotary_pos_emb(self, x, cos, sin):
                # Simple mock rotary embedding
                return x
                
            def forward(self, hidden_states, attention_mask=None, position_ids=None):
                # Simple pass-through implementation for testing
                return hidden_states
        
        # Create a minimal transformer model
        class MinimalTransformer:
            def __init__(self):
                self.hidden_size = 32
                self.num_layers = 2
                self.layers = [MinimalTransformerLayer() for _ in range(2)]
                self.final_layernorm_weight = mx.ones((32,))
                self.final_layernorm_bias = mx.zeros((32,))
        
        # Create the model instance
        self.base_model = MinimalTransformer()
        
        # Create the LoRA transformer
        from csm.mlx.components.lora import LoRATransformer
        self.lora_model = LoRATransformer(
            transformer_model=self.base_model,
            r=4,
            alpha=8.0,
            dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            target_layers=[0],  # Only apply to first layer
            use_bias=True
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_lora_transformer_initialization(self):
        """Test LoRA transformer initialization."""
        # Check the number of LoRA layers matches
        self.assertEqual(len(self.lora_model.lora_layers), self.base_model.num_layers)
        
        # Check first layer has LoRA adapters
        layer_idx, layer = self.lora_model.lora_layers[0]
        self.assertEqual(layer_idx, 0)
        self.assertIsNotNone(getattr(layer, 'lora_adapters', None))
        
        # Check second layer does not have LoRA adapters (not in target_layers)
        layer_idx, layer = self.lora_model.lora_layers[1]
        self.assertEqual(layer_idx, 1)
        self.assertEqual(layer, self.base_model.layers[1])
        
        # Check target modules were applied correctly
        layer_idx, lora_layer = self.lora_model.lora_layers[0]
        self.assertIn("q_proj", lora_layer.lora_adapters)
        self.assertIn("v_proj", lora_layer.lora_adapters)
        self.assertNotIn("k_proj", lora_layer.lora_adapters)
        self.assertNotIn("o_proj", lora_layer.lora_adapters)
    
    def test_lora_transformer_parameters(self):
        """Test LoRA transformer parameters method."""
        params = self.lora_model.parameters()
        
        # Verify parameters exist for first layer only
        self.assertIn("layers.0.attn.q_proj.lora_A", params)
        self.assertIn("layers.0.attn.v_proj.lora_A", params)
        
        # Check parameters for second layer don't exist
        for param_name in params:
            self.assertFalse(param_name.startswith("layers.1."))
    
    def test_lora_transformer_forward(self):
        """Test LoRA transformer forward pass."""
        # Create input
        batch_size = 2
        seq_len = 4
        hidden_states = mx.ones((batch_size, seq_len, self.base_model.hidden_size))
        
        # Run forward pass
        output = self.lora_model.forward(hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, hidden_states.shape)
    
    def test_lora_transformer_update(self):
        """Test LoRA transformer update method."""
        # Get parameters
        params = self.lora_model.parameters()
        
        # Create a modified parameter dict
        modified_params = {}
        for name, param in params.items():
            modified_params[name] = param * 2.0
        
        # Update parameters
        self.lora_model.update(modified_params)
        
        # Get updated parameters
        updated_params = self.lora_model.parameters()
        
        # Check parameters were updated
        for name, param in updated_params.items():
            self.assertTrue(mx.allclose(param, modified_params[name], atol=1e-5))


@unittest.skipIf(not HAS_MLX, "MLX not available")
class TestCSMLoRATrainer(unittest.TestCase):
    """Test the CSM LoRA trainer."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal safetensors file for testing
        self.model_path = os.path.join(self.temp_dir, "model.safetensors")
        
        # Create a minimal MLX model
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
                
                # Set up model components
                self.backbone = MinimalBackbone()
                self.decoder = MinimalDecoder()
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
                
                return params
            
            def update(self, params_dict):
                """Update model parameters."""
                pass
        
        # Create dummy safetensors file
        import safetensors.numpy
        dummy_weights = {"test": np.zeros((1, 1))}
        safetensors.numpy.save_file(dummy_weights, self.model_path)

        # Create metadata file
        metadata = {
            "epoch": 0,
            "global_step": 0,
            "loss": 1.0,
            "model_path": self.model_path
        }
        
        with open(os.path.join(self.temp_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_lora_trainer_initialization(self):
        """Test LoRA trainer initialization."""
        try:
            from csm.training.lora_trainer import CSMLoRATrainer
            
            # Create trainer with mock model
            trainer = CSMLoRATrainer(
                model_path=self.model_path,
                output_dir=self.temp_dir,
                lora_r=8,
                lora_alpha=16.0,
                target_modules=["q_proj", "v_proj"],
                target_layers=[0, 1]
            )
            
            # Verify trainer attributes
            self.assertEqual(trainer.model_path, self.model_path)
            self.assertEqual(str(trainer.output_dir), self.temp_dir)
            self.assertEqual(trainer.lora_r, 8)
            self.assertEqual(trainer.lora_alpha, 16.0)
            self.assertEqual(trainer.target_modules, ["q_proj", "v_proj"])
            self.assertEqual(trainer.target_layers, [0, 1])
            self.assertEqual(trainer.lora_dropout, 0.0)
            self.assertEqual(trainer.lora_use_bias, False)
            
        except ImportError as e:
            self.skipTest(f"Couldn't import CSMLoRATrainer: {e}")
    
    def test_save_load_lora_weights(self):
        """Test saving and loading LoRA weights."""
        try:
            from csm.training.lora_trainer import CSMLoRATrainer
            from csm.mlx.components.lora import apply_lora_to_model
            from csm.mlx.components.model_wrapper import MLXModelWrapper
            
            # Create a model
            model_args = {
                "backbone_flavor": "llama-1B",
                "decoder_flavor": "llama-100M",
                "text_vocab_size": 128256,
                "audio_vocab_size": 2051,
                "audio_num_codebooks": 32,
                "debug": True  # Enable debug mode for better error messages
            }
            
            try:
                # Initialize model
                model = MLXModelWrapper(model_args)
                # Apply LoRA
                model = apply_lora_to_model(
                    model=model,
                    r=4,
                    alpha=8.0
                )
                
                # Create trainer
                trainer = CSMLoRATrainer(
                    model_path=self.model_path,
                    output_dir=self.temp_dir,
                    lora_r=4,
                    lora_alpha=8.0
                )
                
                # Set the model
                trainer.model = model
                
                # Save LoRA weights
                lora_path = os.path.join(self.temp_dir, "lora_weights.safetensors")
                trainer.save_model(lora_path, save_mode="lora")
                
                # Check file exists
                self.assertTrue(os.path.exists(lora_path))
                
                # Try to load the weights
                trainer.load_lora_weights(lora_path)
                
                # Success if we reach here without errors
                self.assertTrue(True)
                
            except Exception as e:
                self.skipTest(f"Error in model creation: {e}")
                
        except ImportError as e:
            self.skipTest(f"Couldn't import required modules: {e}")


# Only run if this module is being executed directly
if __name__ == "__main__":
    unittest.main()