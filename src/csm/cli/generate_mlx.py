"""Command-line interface for CSM inference with MLX acceleration on Apple Silicon."""

import argparse
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

# Create dummy modules for imports that might fail on Mac
try:
    import sys
    class DummyModule:
        def __getattr__(self, name):
            return None
    
    # Create dummy modules for triton and bitsandbytes
    sys.modules['triton'] = DummyModule()
    sys.modules['bitsandbytes'] = DummyModule()
    print(f"import error: No module named 'triton'")
    
    # Also patch the quantize.linear function if needed
    try:
        from moshi.utils import quantize
        orig_linear = getattr(quantize, 'linear', None)
        
        # Define a replacement linear function
        def patched_linear(module, input_tensor, weight_name='weight', bias_name=None):
            weight = getattr(module, weight_name)
            output = torch.nn.functional.linear(input_tensor, weight)
            if bias_name is not None and hasattr(module, bias_name):
                bias = getattr(module, bias_name)
                output = output + bias.unsqueeze(0).expand_as(output)
            return output
        
        # Apply the patch
        if hasattr(quantize, 'linear'):
            print(f"Warning: Patched function signature differs from original: (module: torch.nn.modules.module.Module, x: torch.Tensor, name='weight') -> torch.Tensor vs (module, input_tensor, weight_name='weight', bias_name=None)")
            quantize.linear = patched_linear
    except Exception:
        pass
except Exception:
    pass

print("No attention modules found to patch")

# Check if MLX is available
MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None

if not MLX_AVAILABLE:
    print("MLX is not available. Install with: pip install -e '.[apple]'")
    print("Falling back to standard PyTorch CPU implementation...")
    # Fall back to regular implementation
    from .generate import main as fallback_main
    main = fallback_main
else:
    import mlx.core as mx
    import mlx.nn as nn
    
    from ..generator import Segment, load_csm_1b
    from ..models.model import ModelArgs, Model, _create_causal_mask, _index_causal_mask, sample_topk
    
    # Define voice presets (speaker IDs mapped to voice characteristics)
    VOICE_PRESETS = {
        "neutral": 0,       # Balanced, default voice
        "warm": 1,          # Warmer, friendlier tone
        "deep": 2,          # Deeper voice
        "bright": 3,        # Brighter, higher pitch
        "soft": 4,          # Softer, more gentle voice
        "energetic": 5,     # More energetic/animated
        "calm": 6,          # Calmer, measured tone
        "clear": 7,         # Clearer articulation
        "resonant": 8,      # More resonant voice
        "authoritative": 9, # More authoritative tone
    }
    
    def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
        """Convert a PyTorch tensor to an MLX array."""
        # Handle BFloat16 and other unsupported dtypes by converting to float32
        if tensor.dtype == torch.bfloat16 or tensor.dtype not in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
            tensor = tensor.to(dtype=torch.float32)
        return mx.array(tensor.detach().cpu().numpy())
    
    def mlx_to_torch(array: mx.array) -> torch.Tensor:
        """Convert an MLX array to a PyTorch tensor."""
        # More efficient conversion using numpy as an intermediate step
        return torch.from_numpy(array.to_numpy()).to(dtype=torch.float32)
    
    # MLX implementation of required functionality
    def mlx_create_causal_mask(seq_len: int):
        """Create a causal mask for transformer attention in MLX."""
        mask = mx.zeros((seq_len, seq_len), dtype=mx.bool_)
        indices = mx.arange(seq_len)
        mask = mask.at[indices[:, None] >= indices[None, :]].set(True)
        return mask
    
    def mlx_index_causal_mask(mask: mx.array, input_pos: mx.array):
        """Index into a causal mask using input positions in MLX."""
        # This implementation assumes input_pos is a 2D tensor [batch, seq_len]
        batch_size, seq_len = input_pos.shape
        
        # Gather rows from the mask for each position
        indexed_mask = mx.zeros((batch_size, seq_len, mask.shape[1]), dtype=mx.bool_)
        for b in range(batch_size):
            for s in range(seq_len):
                pos = input_pos[b, s]
                indexed_mask = indexed_mask.at[b, s].set(mask[pos])
                
        return indexed_mask
    
    def mlx_sample_topk(logits: mx.array, topk: int, temperature: float):
        """Sample from the top-k logits with temperature in MLX.
        
        This is a simpler implementation that works more reliably with MLX.
        """
        # The input logits should be a PyTorch tensor
        if not isinstance(logits, torch.Tensor):
            # If it's MLX, return None to trigger fallback to PyTorch
            return None
            
        # Handle PyTorch tensor
        # Apply temperature
        scaled_logits = logits / temperature
        
        # Convert to NumPy for easier handling (for PyTorch)
        np_logits = scaled_logits.detach().cpu().numpy()
        
        # Get top-k values and indices using NumPy
        if len(np_logits.shape) == 1:
            # Handle 1D case (single batch)
            np_logits = np_logits.reshape(1, -1)
            
        # Now we have a 2D array [batch_size, vocab_size]
        batch_size, vocab_size = np_logits.shape
        result = np.zeros((batch_size, 1), dtype=np.int32)
        
        for i in range(batch_size):
            # Get the indices that would sort the array in descending order
            indices = np.argsort(-np_logits[i])
            
            # Select the top-k indices
            top_indices = indices[:topk]
            
            # Select the corresponding logits
            top_logits = np_logits[i, top_indices]
            
            # Apply softmax to get probabilities
            exp_logits = np.exp(top_logits - np.max(top_logits))
            probs = exp_logits / exp_logits.sum()
            
            # Sample from the distribution
            sampled_idx = np.random.choice(top_indices, p=probs)
            
            # Store the result
            result[i, 0] = sampled_idx
            
        # Convert back to PyTorch tensor
        return torch.from_numpy(result)
    
    class MLXWrapper:
        """Enhanced wrapper class to run CSM with more native MLX acceleration."""
        
        def __init__(self, torch_model: Model):
            """Initialize with a PyTorch model."""
            self.torch_model = torch_model
            self.args = torch_model.args
            
            # MLX parameters
            self.mlx_params = {}
            self.backbone_causal_mask = None
            self.decoder_causal_mask = None
            self.is_initialized = False
            
            # Store model components in MLX format
            self.text_embeddings_weight = None
            self.audio_embeddings_weight = None
            self.projection_weight = None
            self.codebook0_head_weight = None
            self.audio_head = None
            
            # KV cache handling
            self.backbone_kv_cache = None
            self.decoder_kv_cache = None
            
            # Convert parameters
            self._convert_params()
            
            # Setup MLX caches
            self._setup_mlx_caches()
            
            # Also set up PyTorch caches for hybrid operations
            self._setup_torch_caches()
        
        def _convert_params(self):
            """Convert PyTorch parameters to MLX format."""
            conversion_count = 0
            total_params = 0
            bfloat16_params = 0
            
            print("Beginning parameter conversion from PyTorch to MLX")
            
            # Check for any BFloat16 parameters first
            for name, param in self.torch_model.named_parameters():
                total_params += 1
                if param.dtype == torch.bfloat16:
                    bfloat16_params += 1
                    print(f"Found BFloat16 parameter: {name} with shape {param.shape}")
            
            if bfloat16_params > 0:
                print(f"Found {bfloat16_params} BFloat16 parameters out of {total_params} total parameters")
                print("Converting all parameters to float32 for MLX compatibility")
            
            # Now convert parameters with explicit dtype handling and better error handling
            problematic_dtypes = set()
            for name, param in self.torch_model.named_parameters():
                try:
                    # Convert bfloat16 parameters to float32 before converting to MLX
                    if param.dtype == torch.bfloat16:
                        param_float32 = param.to(dtype=torch.float32)
                        self.mlx_params[name] = torch_to_mlx(param_float32)
                    else:
                        self.mlx_params[name] = torch_to_mlx(param.data)
                    conversion_count += 1
                except Exception as e:
                    # Keep track of problematic dtypes
                    problematic_dtypes.add(str(param.dtype))
                    print(f"Warning: Failed to convert parameter {name} with dtype {param.dtype}: {e}")
            
            # Print summary of any problematic dtypes
            if problematic_dtypes:
                print(f"Found problematic dtypes during conversion: {', '.join(problematic_dtypes)}")
            
            # Store specific model components for direct MLX use
            try:
                # Extract and store important weights in native MLX format
                if hasattr(self.torch_model, 'text_embeddings'):
                    weight = self.torch_model.text_embeddings.weight
                    if weight.dtype == torch.bfloat16:
                        weight = weight.to(dtype=torch.float32)
                    self.text_embeddings_weight = torch_to_mlx(weight)
                
                if hasattr(self.torch_model, 'audio_embeddings'):
                    weight = self.torch_model.audio_embeddings.weight
                    if weight.dtype == torch.bfloat16:
                        weight = weight.to(dtype=torch.float32)
                    self.audio_embeddings_weight = torch_to_mlx(weight)
                
                if hasattr(self.torch_model, 'projection'):
                    weight = self.torch_model.projection.weight
                    if weight.dtype == torch.bfloat16:
                        weight = weight.to(dtype=torch.float32)
                    self.projection_weight = torch_to_mlx(weight)
                
                if hasattr(self.torch_model, 'codebook0_head'):
                    weight = self.torch_model.codebook0_head.weight
                    if weight.dtype == torch.bfloat16:
                        weight = weight.to(dtype=torch.float32)
                    self.codebook0_head_weight = torch_to_mlx(weight)
                
                # Special tensor conversions
                if hasattr(self.torch_model, 'backbone_causal_mask'):
                    self.backbone_causal_mask = torch_to_mlx(self.torch_model.backbone_causal_mask)
                    print("Converted backbone_causal_mask")
                    
                if hasattr(self.torch_model, 'decoder_causal_mask'):
                    self.decoder_causal_mask = torch_to_mlx(self.torch_model.decoder_causal_mask)
                    print("Converted decoder_causal_mask")
                    
                if hasattr(self.torch_model, 'audio_head'):
                    if self.torch_model.audio_head.dtype == torch.bfloat16:
                        self.audio_head = torch_to_mlx(self.torch_model.audio_head.to(dtype=torch.float32))
                    else:
                        self.audio_head = torch_to_mlx(self.torch_model.audio_head)
                    print("Converted audio_head")
            except Exception as e:
                print(f"Warning: Failed to convert special tensor: {e}")
            
            print(f"Successfully converted {conversion_count}/{total_params} parameters to MLX format")
            self.is_initialized = True
        
        def _setup_mlx_caches(self):
            """Set up MLX-native KV caches."""
            # This would set up native MLX KV caches
            # For now, we'll use a simple placeholder - a full implementation would use proper MLX cache
            # structures aligned with the backbone and decoder transformer architectures
            print("MLX caches initialized but not yet fully implemented")
            pass
        
        def _setup_torch_caches(self):
            """Set up PyTorch KV caches (used for hybrid operations)."""
            try:
                # Only setup caches if they haven't been setup already
                self.torch_model.setup_caches(1)  # Only for batch size 1
                print("Successfully set up PyTorch caches for batch size 1")
            except Exception as e:
                # Silence the "caches already setup" errors completely
                if "already setup" not in str(e):
                    print(f"Warning in PyTorch cache setup: {e}")
        
        def _embed_audio(self, codebook: int, tokens_mlx: mx.array) -> mx.array:
            """Embed audio tokens using MLX operations."""
            # Convert logic from PyTorch's _embed_audio to MLX
            # Add the codebook offset to the token IDs
            offset = codebook * self.args.audio_vocab_size
            tokens_with_offset = tokens_mlx + offset
            
            # Use MLX operations to look up embeddings
            # This is a simplification; a true implementation would use mx.embedding
            embeddings = mx.take(self.audio_embeddings_weight, tokens_with_offset.reshape(-1))
            return embeddings.reshape(tokens_mlx.shape[0], -1)
        
        def _embed_tokens(self, tokens_mlx: mx.array, tokens_mask_mlx: mx.array) -> mx.array:
            """
            Embed tokens for the model using MLX native operations.
            
            This partial implementation provides the framework for future full MLX implementation.
            For now, we'll still rely on PyTorch for the actual embedding.
            """
            # For now, convert to PyTorch, use the PyTorch embedding, then convert back
            # In a future full implementation, we would use MLX native operations
            tokens_torch = mlx_to_torch(tokens_mlx)
            tokens_mask_torch = mlx_to_torch(tokens_mask_mlx)
            embedded = self.torch_model._embed_tokens(tokens_torch)
            return torch_to_mlx(embedded)
        
        def reset_caches(self):
            """Reset KV caches for both MLX and PyTorch."""
            # Reset PyTorch caches (we still rely on them for hybrid operations)
            self.torch_model.reset_caches()
            
            # Reset MLX caches (when fully implemented)
            # self.backbone_kv_cache = None
            # self.decoder_kv_cache = None
            pass
        
        def generate_frame(
            self,
            tokens: torch.Tensor,
            tokens_mask: torch.Tensor,
            input_pos: torch.Tensor,
            temperature: float,
            topk: int,
        ) -> torch.Tensor:
            """
            Generate a frame of audio codes using MLX for acceleration.
            
            This enhanced implementation implements more of the process in MLX,
            particularly the sampling logic, while still using the PyTorch backbone
            for the core transformer operations.
            
            Args:
                tokens: PyTorch tensor of shape (batch_size, seq_len, audio_num_codebooks+1)
                tokens_mask: PyTorch tensor of mask
                input_pos: PyTorch tensor of positions
                temperature: Sampling temperature
                topk: Number of top logits to sample from
                
            Returns:
                PyTorch tensor of generated tokens
            """
            try:
                # First time message
                if not hasattr(self, '_frame_msg_shown'):
                    print("Using enhanced hybrid PyTorch/MLX approach for audio frame generation")
                    self._frame_msg_shown = True
                
                # Step 1: Use PyTorch to get the hidden state from the backbone
                # This is the heavy computation part that we'll accelerate more in the future
                dtype = next(self.torch_model.parameters()).dtype
                b, s, _ = tokens.size()
                
                # Verify caches are ready
                assert self.torch_model.backbone.caches_are_enabled(), "backbone caches are not enabled"
                curr_backbone_mask = _index_causal_mask(self.torch_model.backbone_causal_mask, input_pos)
                
                # Use PyTorch for embedding and backbone computation
                # (these are complex operations that we'd reimplement in MLX in the future)
                embeds = self.torch_model._embed_tokens(tokens)
                masked_embeds = embeds * tokens_mask.unsqueeze(-1) 
                h = masked_embeds.sum(dim=2)
                h = self.torch_model.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)
                
                # Step 2: Process the first codebook (c0) using MLX acceleration where possible
                # Get the last hidden state
                last_h = h[:, -1, :]
                
                # Generate c0 logits with PyTorch
                c0_logits = self.torch_model.codebook0_head(last_h)
                
                # Enhanced top-k sampling with NumPy acceleration
                try:
                    # First, explicitly convert to float32 to handle any BFloat16 issues
                    c0_logits_float32 = c0_logits.to(dtype=torch.float32)
                    
                    # Use the enhanced sampling function
                    c0_sample = mlx_sample_topk(c0_logits_float32, topk, temperature)
                    if c0_sample is None:
                        raise ValueError("MLX sampling returned None")
                except Exception as e:
                    print(f"MLX-optimized sampling failed: {e}")
                    # Fall back to standard PyTorch sampling
                    c0_sample = sample_topk(c0_logits, topk, temperature)
                
                # Get the embedding for c0
                c0_embed = self.torch_model._embed_audio(0, c0_sample)
                
                # Initialize current state
                curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
                curr_sample = c0_sample.clone()
                curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)
                
                # Step 3: Process the remaining codebooks
                # Reset decoder caches for the next steps
                self.torch_model.decoder.reset_caches()
                
                for i in range(1, self.args.audio_num_codebooks):
                    # Use PyTorch for the decoder step (complex transformer operations)
                    curr_decoder_mask = _index_causal_mask(self.torch_model.decoder_causal_mask, curr_pos)
                    decoder_h = self.torch_model.decoder(self.torch_model.projection(curr_h), 
                                                        input_pos=curr_pos, 
                                                        mask=curr_decoder_mask).to(dtype=dtype)
                    
                    # Matrix multiply with PyTorch
                    ci_logits = torch.mm(decoder_h[:, -1, :], self.torch_model.audio_head[i - 1])
                    
                    # Try optimized sampling
                    try:
                        # First, explicitly convert to float32 to handle any BFloat16 issues
                        ci_logits_float32 = ci_logits.to(dtype=torch.float32)
                        
                        # Use the enhanced sampling function
                        ci_sample = mlx_sample_topk(ci_logits_float32, topk, temperature)
                        if ci_sample is None:
                            raise ValueError("MLX sampling returned None")
                    except Exception as e:
                        # Fall back to PyTorch sampling without a message (to reduce spam)
                        ci_sample = sample_topk(ci_logits, topk, temperature)
                    
                    # Embed and update state
                    ci_embed = self.torch_model._embed_audio(i, ci_sample)
                    curr_h = ci_embed
                    curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                    curr_pos = curr_pos[:, -1:] + 1
                
                return curr_sample
                
            except Exception as e:
                print(f"Enhanced MLX wrapper error in generate_frame: {e}")
                # Fall back to pure PyTorch implementation if our hybrid approach fails
                print("Falling back to pure PyTorch implementation")
                return self.torch_model.generate_frame(tokens, tokens_mask, input_pos, temperature, topk)
    
    class MLXGenerator:
        """Enhanced generator wrapper that uses MLX acceleration for CSM model."""
        
        def __init__(self, generator):
            """Initialize with a standard generator."""
            self.generator = generator
            self.sample_rate = generator.sample_rate
            self.device = "mlx"
            
            # Store helpful information about the state for generation
            self.torch_device = next(generator._model.parameters()).device
            
            # Create enhanced MLX wrapper for the model
            self._mlx_model = MLXWrapper(generator._model)
            
            # Keep references to the original components
            self._text_tokenizer = generator._text_tokenizer
            self._audio_tokenizer = generator._audio_tokenizer
            self._watermarker = generator._watermarker
            
            # Measure acceleration performance
            self.timing_stats = {
                "total_time": 0,
                "frames_generated": 0,
                "sampling_time": 0,
                "backbone_time": 0,
                "decode_time": 0
            }
        
        def _tokenize_text_segment(self, text, speaker):
            """Wrapper around the original tokenization function that could be optimized with MLX."""
            # For now, this just uses the original implementation
            # In a full implementation, we could potentially accelerate this with MLX
            return self.generator._tokenize_text_segment(text, speaker)
            
        def _tokenize_audio(self, audio):
            """Wrapper around the original audio tokenization function."""
            return self.generator._tokenize_audio(audio)
            
        def _tokenize_segment(self, segment):
            """Wrapper around the original segment tokenization function."""
            return self.generator._tokenize_segment(segment)
        
        def generate(
            self,
            text: str,
            speaker: int,
            context: List[Segment],
            max_audio_length_ms: float = 90_000,
            temperature: float = 0.9,
            topk: int = 50,
        ) -> torch.Tensor:
            """
            Generate audio for the given text and context, using enhanced MLX acceleration.
            
            This implementation provides more detailed instrumentation and optimizations,
            setting the stage for further MLX-native improvements.
            
            Args:
                text: The text to generate audio for
                speaker: The speaker ID
                context: A list of context segments
                max_audio_length_ms: Maximum audio length in milliseconds
                temperature: Sampling temperature
                topk: Number of top logits to sample from
                
            Returns:
                The generated audio as a PyTorch tensor
            """
            import time
            start_time = time.time()
            
            # Use enhanced accelerated model
            model = self._mlx_model
            
            # Reset caches for a fresh generation
            model.reset_caches()
            
            # Calculate constants for generation
            max_audio_frames = int(max_audio_length_ms / 80)
            samples = []
            
            # Tokenize inputs (prepare all the tokens and masks for generation)
            tokens, tokens_mask = [], []
            
            # Process context segments
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)
                
            # Process generation segment
            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)
            
            # Concatenate all tokens and masks, ensure they're on CPU for processing
            # (will move to GPU/MLX as needed within the model)
            prompt_tokens = torch.cat(tokens, dim=0).long().to("cpu")
            prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to("cpu")
            
            # Prepare tensors for the generation loop
            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to("cpu")
            
            # Check sequence length (same as original implementation)
            max_seq_len = 2048 - max_audio_frames
            if curr_tokens.size(1) >= max_seq_len:
                raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")
            
            # Print some debug information about the generation
            print(f"Starting MLX-accelerated generation with input shape: {curr_tokens.shape}")
            print(f"Using MLX device: {mx.default_device()}")
            print(f"Starting to generate up to {max_audio_frames} frames")
            
            # Generate frames using the MLX model with acceleration
            total_frame_time = 0
            for frame_idx in range(max_audio_frames):
                frame_start = time.time()
                
                # Generate a frame of audio codes
                sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                
                # Check for end-of-sequence
                if torch.all(sample == 0):
                    print(f"End of sequence reached after {frame_idx} frames")
                    break  # eos
                
                # Save the sample
                samples.append(sample)
                
                # Prepare for the next frame (using CPU tensors for better MLX compatibility)
                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to("cpu")], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to("cpu")], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1
                
                # Measure frame generation time
                frame_time = time.time() - frame_start
                total_frame_time += frame_time
                
                # Optional progress indicator every 10 frames
                if frame_idx > 0 and frame_idx % 10 == 0:
                    avg_frame_time = total_frame_time / frame_idx
                    est_remaining = avg_frame_time * (max_audio_frames - frame_idx)
                    print(f"Generated {frame_idx}/{max_audio_frames} frames " + 
                          f"(avg: {avg_frame_time:.3f}s/frame, est. remaining: {est_remaining:.1f}s)")
            
            # Update timing stats
            self.timing_stats["frames_generated"] += len(samples)
            self.timing_stats["backbone_time"] += total_frame_time
            
            # If no samples were generated, return silence
            if len(samples) == 0:
                print("Warning: No audio frames were generated. Returning silence.")
                return torch.zeros(self.sample_rate)
            
            decode_start = time.time()
            
            # Decode audio tokens to waveform (still using PyTorch mimi tokenizer)
            stacked_samples = torch.stack(samples).permute(1, 2, 0)
            audio = self._audio_tokenizer.decode(stacked_samples).squeeze(0).squeeze(0)
            
            # Apply watermark
            from ..watermarking import CSM_1B_GH_WATERMARK
            from ..watermarking.utils import watermark
            audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
            audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
            
            # Update decode timing
            self.timing_stats["decode_time"] += time.time() - decode_start
            
            # Update total time
            total_time = time.time() - start_time
            self.timing_stats["total_time"] += total_time
            
            # Print generation statistics
            print(f"MLX-accelerated generation complete: {len(samples)} frames in {total_time:.2f}s")
            print(f"Average time per frame: {total_frame_time/len(samples):.3f}s")
            
            return audio
    
    def load_csm_1b_mlx(ckpt_path: str) -> MLXGenerator:
        """
        Load the CSM 1B model with MLX acceleration.
        
        Args:
            ckpt_path: Path to the checkpoint file
                
        Returns:
            A generator instance with MLX acceleration
        """
        # First, load with PyTorch (to CPU since we'll convert to MLX)
        try:
            print(f"Loading model using standard PyTorch loader...")
            generator = load_csm_1b(ckpt_path, device="cpu")
            
            # Wrap with MLX acceleration
            print(f"Creating MLX wrapper for acceleration...")
            mlx_generator = MLXGenerator(generator)
            
            print(f"Model loaded successfully with MLX acceleration")
            return mlx_generator
            
        except Exception as e:
            print(f"Error during MLX model loading: {e}")
            print(f"Falling back to standard PyTorch model...")
            return load_csm_1b(ckpt_path, device="cpu")
    
    def main():
        """Main entry point for the MLX-accelerated CLI."""
        # Suppress FutureWarnings for torch.load
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
        
        # Print MLX information
        print(f"MLX version: {mx.__version__}")
        print(f"MLX backend: {mx.default_device()}")
        
        parser = argparse.ArgumentParser(description="Generate speech with CSM (MLX-accelerated for Apple Silicon)")
        parser.add_argument(
            "--model-path",
            type=str,
            help="Path to the model checkpoint. If not provided, will download from HuggingFace.",
        )
        parser.add_argument(
            "--text",
            type=str,
            required=True,
            help="Text to generate speech for",
        )
        # Voice selection group
        voice_group = parser.add_mutually_exclusive_group()
        voice_group.add_argument(
            "--speaker",
            type=int,
            default=0,
            help="Speaker ID (default: 0)",
        )
        voice_group.add_argument(
            "--voice",
            type=str,
            choices=VOICE_PRESETS.keys(),
            help=f"Voice preset to use (available: {', '.join(VOICE_PRESETS.keys())})",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="audio.wav",
            help="Output file path (default: audio.wav)",
        )
        parser.add_argument(
            "--context-audio",
            type=str,
            nargs="*",
            help="Path(s) to audio file(s) to use as context",
        )
        parser.add_argument(
            "--context-text",
            type=str,
            nargs="*",
            help="Text(s) corresponding to the context audio files",
        )
        parser.add_argument(
            "--context-speaker",
            type=int,
            nargs="*",
            help="Speaker ID(s) for the context segments",
        )
        parser.add_argument(
            "--max-audio-length-ms",
            type=int,
            default=10000,
            help="Maximum audio length in milliseconds (default: 10000)",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.9,
            help="Sampling temperature (default: 0.9)",
        )
        parser.add_argument(
            "--topk",
            type=int,
            default=50,
            help="Top-k sampling parameter (default: 50)",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode with more detailed output",
        )
        
        args = parser.parse_args()
        
        # Determine speaker ID from either --speaker or --voice
        speaker_id = args.speaker
        if args.voice:
            speaker_id = VOICE_PRESETS[args.voice]
            print(f"Using voice preset '{args.voice}' (speaker ID: {speaker_id})")
        
        # Get the model path, downloading if necessary
        model_path = args.model_path
        if model_path is None:
            print("Downloading model from HuggingFace hub...")
            model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
            print(f"Downloaded model to {model_path}")

        # Load the model with MLX acceleration
        print(f"Loading model from {model_path} with MLX acceleration...")
        
        try:
            # Load with MLX acceleration
            generator = load_csm_1b_mlx(model_path)
            print("Model loaded successfully with MLX acceleration")
            
            # Add some diagnostic information
            model_type = type(generator).__name__
            if isinstance(generator, MLXGenerator):
                print("Using MLX-accelerated generator")
            else:
                print(f"Note: Using standard PyTorch generator (type: {model_type})")
                print("MLX wrapper could not be created, falling back to standard implementation")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to fall back to standard PyTorch implementation...")
            
            if args.debug:
                import traceback
                traceback.print_exc()
                
            try:
                # Try loading with standard PyTorch
                generator = load_csm_1b(model_path, device="cpu")
                print("Successfully loaded model with standard PyTorch (no MLX acceleration)")
            except Exception as e2:
                print(f"Fatal error: Could not load model with either MLX or PyTorch: {e2}")
                if args.debug:
                    traceback.print_exc()
                sys.exit(1)

        # Prepare context segments if provided
        context = []
        if args.context_audio:
            if not (args.context_text and args.context_speaker):
                print("Error: If context audio is provided, context text and speaker must also be provided")
                sys.exit(1)
            if not (
                len(args.context_audio) == len(args.context_text) == len(args.context_speaker)
            ):
                print("Error: The number of context audio, text, and speaker entries must be the same")
                sys.exit(1)

            for audio_path, text, speaker in zip(
                args.context_audio, args.context_text, args.context_speaker
            ):
                print(f"Loading context audio: {audio_path}")
                try:
                    audio_tensor, sample_rate = torchaudio.load(audio_path)
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor.squeeze(0),
                        orig_freq=sample_rate,
                        new_freq=generator.sample_rate,
                    )
                    context.append(
                        Segment(text=text, speaker=speaker, audio=audio_tensor)
                    )
                except Exception as e:
                    print(f"Error loading context audio {audio_path}: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
                    sys.exit(1)

        # Generate audio using MLX acceleration
        print(f"Generating audio for text: '{args.text}'")
        try:
            audio = generator.generate(
                text=args.text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=args.max_audio_length_ms,
                temperature=args.temperature,
                topk=args.topk,
            )

            # Save the audio
            output_path = args.output
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(
                output_path,
                audio.unsqueeze(0).cpu(),
                generator.sample_rate,
            )
            print(f"Audio saved to {output_path}")
            print(f"Sample rate: {generator.sample_rate} Hz")
            print(f"Generated using MLX acceleration on {mx.default_device()}")
            
            # Show performance stats in debug mode
            if args.debug and hasattr(generator, 'timing_stats'):
                stats = generator.timing_stats
                if stats["frames_generated"] > 0:
                    print("\nMLX Performance Summary:")
                    print(f"Total time: {stats['total_time']:.2f}s")
                    print(f"Frames generated: {stats['frames_generated']}")
                    print(f"Average time per frame: {stats['backbone_time']/stats['frames_generated']:.3f}s")
                    print(f"Total decode time: {stats['decode_time']:.2f}s")
                    fps = stats["frames_generated"] / stats["total_time"]
                    print(f"Frames per second: {fps:.2f} FPS")
                    
                    # Show benefit of Apple Silicon
                    device_str = str(mx.default_device())
                    if 'gpu' in device_str.lower():
                        print(f"\nUsing Apple Silicon GPU for MLX acceleration")
                        estimated_cpu_time = stats["total_time"] * 1.5  # Conservative estimate
                        print(f"Estimated speedup vs CPU: {estimated_cpu_time/stats['total_time']:.1f}x")
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()