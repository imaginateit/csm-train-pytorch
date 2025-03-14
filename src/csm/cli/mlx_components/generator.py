"""
MLX Generator for CSM.
"""

import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch

from csm.cli.mlx_wrapper import MLXWrapper
from csm.cli.mlx_components.config import get_voice_preset
from csm.cli.mlx_components.utils import measure_time, is_mlx_available

# Import MLX if available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    # Create a dummy module
    class DummyMX:
        def __getattr__(self, name):
            raise ImportError("MLX is not available")
    mx = DummyMX()

class MLXGenerator:
    """
    MLX-accelerated speech generator that handles the entire generation process.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
        debug: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.voice = None  # Initialize voice attribute
        self.text = None   # Initialize text attribute
        self.sample_rate = 24000  # Default sample rate
        self._last_audio = None  # Store last audio for direct access
        self._last_samples = None  # Store raw samples from generate_frame
        
        # Check if MLX is available
        self.mlx_available = is_mlx_available()
        if self.mlx_available:
            # Initialize MLX wrapper
            try:
                # Create argument holder
                import argparse
                args = argparse.Namespace()
                
                # Copy any arguments from the model if available
                if hasattr(model, 'args'):
                    model_args = model.args
                    args.audio_vocab_size = getattr(model_args, 'audio_vocab_size', 2051)
                    args.audio_num_codebooks = getattr(model_args, 'audio_num_codebooks', 32)
                else:
                    # Default values
                    args.audio_vocab_size = 2051
                    args.audio_num_codebooks = 32
                    
                # Set debug flag
                args.debug = debug
                
                # Create MLX wrapper
                self.mlx_wrapper = MLXWrapper(model, args)
                if self.debug:
                    print("MLX wrapper initialized successfully")
            except Exception as e:
                print(f"Error initializing MLX wrapper: {e}")
                self.mlx_wrapper = None
                self.mlx_available = False
        else:
            self.mlx_wrapper = None
            
    @measure_time
    def generate_speech(
        self,
        text: str,
        voice: str = "standard",
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            voice: Voice preset to use
            temperature: Temperature for sampling (overrides preset)
            topk: Top-k value for sampling (overrides preset)
            progress_callback: Optional callback for progress updates
            
        Returns:
            NumPy array with audio data
        """
        # Store the voice and text for use in generate_audio_tokens_torch
        self.voice = voice
        self.text = text  # Store the original text
        
        # Get voice preset parameters
        preset = get_voice_preset(voice)
        
        # Override with explicit parameters if provided
        if temperature is not None:
            preset["temperature"] = temperature
        if topk is not None:
            preset["topk"] = topk
            
        if self.debug:
            print(f"Using voice preset: {voice}")
            print(f"Temperature: {preset['temperature']}, Top-k: {preset['topk']}")
            
        # Track whether we're using MLX or not
        using_mlx = self.mlx_available and self.mlx_wrapper is not None
        
        # Tokenize the text
        tokenized = self.tokenize_text(text)
        
        # Generate audio frames
        audio_tokens = self.generate_audio_tokens(
            tokenized, 
            temperature=preset["temperature"],
            topk=preset["topk"],
            progress_callback=progress_callback
        )
        
        # Convert audio tokens to audio waveform
        audio = self.decode_audio_tokens(audio_tokens)
        
        return audio
        
    def tokenize_text(self, text: str) -> torch.Tensor:
        """
        Tokenize text input.
        
        Args:
            text: Input text
            
        Returns:
            Tokenized text as PyTorch tensor
        """
        if self.debug:
            print(f"Tokenizing text: {text}")
            
        # Tokenize the text
        if self.tokenizer is None:
            # If no tokenizer, create a simple dummy tokenizer that returns empty tensor
            if self.debug:
                print("WARNING: No tokenizer available, using empty tensor")
            tokens = torch.zeros((1, 0), dtype=torch.long, device=self.device)
        elif hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        elif hasattr(self.tokenizer, "__call__"):
            tokens = self.tokenizer(text)
        else:
            if self.debug:
                print("WARNING: Unknown tokenizer type, using empty tensor")
            tokens = torch.zeros((1, 0), dtype=torch.long, device=self.device)
            
        # Convert to tensor
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, device=self.device)
        elif isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens).to(self.device)
            
        # Add batch dimension if needed
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
            
        if self.debug:
            print(f"Tokenized shape: {tokens.shape}")
            
        return tokens
        
    def generate_audio_tokens(
        self,
        text_tokens: torch.Tensor,
        temperature: float = 1.0,
        topk: int = 25,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        """
        Generate audio tokens from text tokens.
        
        Args:
            text_tokens: Tokenized text input
            temperature: Temperature for sampling
            topk: Top-k value for sampling
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated audio tokens
        """
        with torch.no_grad():
            if self.mlx_available and self.mlx_wrapper is not None:
                # Use MLX acceleration
                if self.debug:
                    print("Using MLX acceleration for audio token generation")
                return self.generate_audio_tokens_mlx(
                    text_tokens, 
                    temperature=temperature,
                    topk=topk,
                    progress_callback=progress_callback
                )
            else:
                # Fall back to pure PyTorch
                if self.debug:
                    print("Using PyTorch for audio token generation")
                return self.generate_audio_tokens_torch(
                    text_tokens, 
                    temperature=temperature,
                    topk=topk,
                    progress_callback=progress_callback
                )
    
    def generate_audio_tokens_mlx(
        self,
        text_tokens: torch.Tensor,
        temperature: float = 1.0,
        topk: int = 25,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        """
        Generate audio tokens using MLX acceleration.
        
        Args:
            text_tokens: Tokenized text input
            temperature: Temperature for sampling
            topk: Top-k value for sampling
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated audio tokens
        """
        # MLX wrapper should handle the generation details
        # Check the parameter names
        import inspect
        
        if hasattr(self.model, 'generate'):
            try:
                sig = inspect.signature(self.model.generate)
                param_names = [param for param in sig.parameters]
                
                if self.debug:
                    print(f"MLX Model.generate parameters: {param_names}")
                
                # Create parameters based on what the model's generate method accepts
                generate_kwargs = {}
                
                # Handle text parameter
                if 'text' in param_names and self.text is not None:
                    # Always prioritize actual text over tokenized text
                    generate_kwargs['text'] = self.text
                elif 'prompt' in param_names and self.text is not None:
                    # Some models might use 'prompt' instead of 'text'
                    generate_kwargs['prompt'] = self.text
                    
                # Handle text_tokens or tokens
                if 'text_tokens' in param_names:
                    generate_kwargs['text_tokens'] = text_tokens
                elif 'tokens' in param_names:
                    generate_kwargs['tokens'] = text_tokens
                    
                # Handle temperature parameter
                if 'temperature' in param_names:
                    generate_kwargs['temperature'] = temperature
                    
                # Handle topk/top_k difference
                if 'topk' in param_names:
                    generate_kwargs['topk'] = topk
                elif 'top_k' in param_names:
                    generate_kwargs['top_k'] = topk
                    
                # Handle callback/progress_callback
                if 'callback' in param_names and progress_callback is not None:
                    generate_kwargs['callback'] = progress_callback
                elif 'progress_callback' in param_names and progress_callback is not None:
                    generate_kwargs['progress_callback'] = progress_callback
                    
                # Handle other common parameters
                if 'use_mlx' in param_names:
                    generate_kwargs['use_mlx'] = True
                    
                # Handle speaker parameter
                if 'speaker' in param_names:
                    # If we used a string voice name, try to convert to int
                    if isinstance(self.voice, str) and self.voice.isdigit():
                        generate_kwargs['speaker'] = int(self.voice)
                    else:
                        # Default to speaker 0
                        generate_kwargs['speaker'] = 0
                        
                if self.debug:
                    print(f"Calling MLX generate with kwargs: {generate_kwargs}")
                    
                # Call the generate method with the appropriate arguments
                audio_tokens = self.model.generate(**generate_kwargs)
                return audio_tokens
                
            except Exception as e:
                if self.debug:
                    print(f"Error in MLX generate: {e}")
                raise
        else:
            raise ValueError("MLX model doesn't have generate method")
    
    def generate_audio_tokens_torch(
        self,
        text_tokens: torch.Tensor,
        temperature: float = 1.0,
        topk: int = 25,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        """
        Generate audio tokens using pure PyTorch.
        
        Args:
            text_tokens: Tokenized text input
            temperature: Temperature for sampling
            topk: Top-k value for sampling
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated audio tokens
        """
        # Try to directly access the audio tokenizer from the original model
        # This is a more direct approach to get proper audio tokens
        if hasattr(self.model, '_audio_tokenizer') and self.text is not None:
            try:
                if self.debug:
                    print("Using model's _audio_tokenizer directly")
                    
                # Extract important components
                audio_tokenizer = self.model._audio_tokenizer
                text_tokenizer = getattr(self.model, '_text_tokenizer', None)
                
                # Try to use the model's _tokenize_text_segment method if available
                if hasattr(self.model, '_tokenize_text_segment') and isinstance(self.voice, (int, str)):
                    speaker_id = int(self.voice) if isinstance(self.voice, str) and self.voice.isdigit() else 0
                    text_tokens, text_masks = self.model._tokenize_text_segment(self.text, speaker_id)
                    
                    # Now try to generate with the model directly
                    if hasattr(self.model, '_model'):
                        if hasattr(self.model._model, 'generate_frame'):
                            if self.debug:
                                print("Using model._model.generate_frame")
                                
                            # Following the same pattern as in the original generator:
                            # We already have tokenized text from _tokenize_text_segment
                            # Get the maximum sequence length from the model
                            max_audio_frames = 125  # Default ~10 seconds at 80ms per frame
                            if hasattr(self.model._model, 'backbone') and hasattr(self.model._model.backbone, 'max_seq_len'):
                                max_seq_len = self.model._model.backbone.max_seq_len
                                max_audio_frames = min(max_audio_frames, max_seq_len // 2)
                                
                            # Reset the model caches
                            if hasattr(self.model._model, 'reset_caches'):
                                self.model._model.reset_caches()
                                
                            # Generate the frames one by one
                            samples = []
                            curr_tokens = text_tokens.unsqueeze(0)  # Add batch dimension
                            curr_tokens_mask = text_masks.unsqueeze(0)  # Add batch dimension
                            curr_pos = torch.arange(0, text_tokens.size(0)).unsqueeze(0).to(text_tokens.device)
                            
                            # Generate frames
                            for _ in range(max_audio_frames):
                                sample = self.model._model.generate_frame(
                                    curr_tokens, curr_tokens_mask, curr_pos, 
                                    temperature=temperature, topk=topk
                                )
                                if torch.all(sample == 0):
                                    break  # EOS
                                    
                                samples.append(sample)
                                
                                # Update context for next token
                                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(text_tokens.device)], dim=1).unsqueeze(1)
                                curr_tokens_mask = torch.cat(
                                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(text_tokens.device)], dim=1
                                ).unsqueeze(1)
                                curr_pos = curr_pos[:, -1:] + 1
                            
                            # Store the samples for decoder to use directly
                            self._last_samples = samples
                            
                            # Return samples directly - decoder will use them or reshape
                            return torch.stack(samples)
                            
                        elif hasattr(self.model._model, 'generate'):
                            if self.debug:
                                print("Using model._model.generate with tokenized text")
                            # Use the underlying model's generate method
                            samples = self.model._model.generate(
                                text_tokens, 
                                text_masks, 
                                temperature=temperature, 
                                topk=topk
                            )
                            # Format the samples as expected
                            if isinstance(samples, list):
                                audio_tokens = torch.stack(samples).permute(1, 2, 0)
                            else:
                                audio_tokens = samples
                            return audio_tokens
            except Exception as e:
                if self.debug:
                    print(f"Direct tokenizer approach failed: {e}")
                # Fall back to standard approach

        # Use the model's native generation method
        # Examine the model to figure out its API
        if hasattr(self.model, 'generate'):
            # Check the parameter names
            import inspect
            sig = inspect.signature(self.model.generate)
            param_names = [param for param in sig.parameters]
            
            if self.debug:
                print(f"Model.generate parameters: {param_names}")
            
            # Create parameters based on what the model's generate method accepts
            generate_kwargs = {}
            
            # Handle text parameter
            if 'text' in param_names and self.text is not None:
                # Always prioritize actual text over tokenized text
                generate_kwargs['text'] = self.text
            elif 'prompt' in param_names and self.text is not None:
                # Some models might use 'prompt' instead of 'text'
                generate_kwargs['prompt'] = self.text
                
            # Handle text_tokens or tokens if we don't have text
            if self.text is None or 'text' not in param_names:
                if 'text_tokens' in param_names:
                    generate_kwargs['text_tokens'] = text_tokens
                elif 'tokens' in param_names:
                    generate_kwargs['tokens'] = text_tokens
                elif 'input_tokens' in param_names:
                    generate_kwargs['input_tokens'] = text_tokens
                    
            # Handle other common parameters
            if 'temperature' in param_names:
                generate_kwargs['temperature'] = temperature
                
            # Handle topk/top_k difference
            if 'topk' in param_names:
                generate_kwargs['topk'] = topk
            elif 'top_k' in param_names:
                generate_kwargs['top_k'] = topk
                
            # Handle callback/progress_callback
            if 'callback' in param_names and progress_callback is not None:
                generate_kwargs['callback'] = progress_callback
            elif 'progress_callback' in param_names and progress_callback is not None:
                generate_kwargs['progress_callback'] = progress_callback
                
            # Handle other parameters
            if 'use_mlx' in param_names:
                generate_kwargs['use_mlx'] = False
                
            # The model might need a speaker or speaker_id
            if 'speaker' in param_names:
                # If we used a string voice name, try to convert to int
                if isinstance(self.voice, str) and self.voice.isdigit():
                    generate_kwargs['speaker'] = int(self.voice)
                else:
                    # Default to speaker 0
                    generate_kwargs['speaker'] = 0
                    
            # Some models need context
            if 'context' in param_names:
                generate_kwargs['context'] = []
                
            # Some models need max_audio_length_ms
            if 'max_audio_length_ms' in param_names:
                generate_kwargs['max_audio_length_ms'] = 10000  # 10 seconds default
                
            if self.debug:
                print(f"Calling generate with kwargs: {generate_kwargs}")
                
            # Call the generate method with the appropriate arguments
            result = self.model.generate(**generate_kwargs)
            
            # Handle different return types
            if isinstance(result, list) and len(result) > 0 and hasattr(result[0], 'audio'):
                # List of Segment objects
                audio_tokens = None
                for segment in result:
                    if hasattr(segment, 'audio'):
                        # Store it for later decoding
                        self._last_audio = segment.audio
                        break
                if audio_tokens is None:
                    # Create dummy tokens
                    audio_tokens = torch.zeros((1, 32, 10), dtype=torch.int64)
            elif isinstance(result, torch.Tensor):
                # Direct tensor result
                audio_tokens = result
            else:
                # Unknown result type
                print(f"Unexpected result type: {type(result)}")
                audio_tokens = torch.zeros((1, 32, 10), dtype=torch.int64)
        else:
            # No generate method - create a dummy output
            print("ERROR: Model has no generate method")
            audio_tokens = torch.zeros((1, 32, 10), dtype=torch.int64)  # Dummy output
        
        return audio_tokens
    
    def decode_audio_tokens(self, audio_tokens: torch.Tensor) -> np.ndarray:
        """
        Decode audio tokens to audio waveform.
        
        Args:
            audio_tokens: Generated audio tokens
            
        Returns:
            Audio waveform as NumPy array
        """
        # Check if we have direct audio available (captured from Segment)
        if self._last_audio is not None:
            if self.debug:
                print("Using direct audio from generator output")
            audio = self._last_audio
            self._last_audio = None  # Clear it for next time
            
            # Convert to NumPy if needed
            if isinstance(audio, torch.Tensor):
                return audio.detach().cpu().numpy()
            elif isinstance(audio, np.ndarray):
                return audio
                
        # Try to use the _last_samples directly if available
        if self._last_samples is not None and hasattr(self.model, '_audio_tokenizer'):
            try:
                if self.debug:
                    print("Using _last_samples with _audio_tokenizer.decode")
                
                # Format samples as in original generator.py: torch.stack(samples).permute(1, 2, 0)
                audio_tokens = torch.stack(self._last_samples).permute(1, 2, 0)
                
                # Reset for next use
                samples = self._last_samples
                self._last_samples = None
                
                if self.debug:
                    print(f"Samples shape for decode: {audio_tokens.shape}")
                
                # Call decode with exactly the format used in original generator
                audio = self.model._audio_tokenizer.decode(audio_tokens).squeeze(0).squeeze(0)
                
                # Apply watermarking if available
                if hasattr(self.model, '_watermarker') and audio is not None:
                    try:
                        from csm.watermarking.utils import watermark
                        from csm.watermarking import CSM_1B_GH_WATERMARK
                        
                        if self.debug:
                            print("Applying watermark to audio")
                            
                        # Apply watermark with the same key as the original generator
                        audio, wm_sample_rate = watermark(
                            self.model._watermarker, 
                            audio, 
                            self.sample_rate, 
                            CSM_1B_GH_WATERMARK
                        )
                        
                        # Resample if needed
                        if wm_sample_rate != self.sample_rate:
                            import torchaudio
                            audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
                    except Exception as e:
                        if self.debug:
                            print(f"Error applying watermark: {e}")
                
                # Return the final audio
                return audio.detach().cpu().numpy()
            
            except Exception as e:
                if self.debug:
                    print(f"Error using _last_samples: {e}")
        
        # Try to use the audio tokenizer directly from the generator
        if hasattr(self.model, '_audio_tokenizer') and audio_tokens is not None:
            try:
                if self.debug:
                    print("Using model._audio_tokenizer.decode directly")
                
                # Need to handle different input formats based on generator output
                # Original format in generator.py: torch.stack(samples).permute(1, 2, 0)
                # samples is a list of tensors with shape [batch_size, audio_num_codebooks]
                
                # First make sure we have a tensor
                if isinstance(audio_tokens, list):
                    if self.debug:
                        print(f"Converting list of tensors to tensor")
                    try:
                        audio_tokens = torch.stack(audio_tokens)
                    except:
                        if self.debug:
                            print(f"Could not stack tokens, using first item")
                        audio_tokens = audio_tokens[0]
                
                # Check the shape and permute/reshape as needed
                if self.debug:
                    print(f"Input audio tokens shape: {audio_tokens.shape}")
                    
                # Handle different shapes
                if len(audio_tokens.shape) == 1:
                    # Single token vector, needs to be shaped for the tokenizer
                    audio_tokens = audio_tokens.reshape(1, -1, 1)
                elif len(audio_tokens.shape) == 2:
                    # This could be [batch, tokens] or [codebooks, frames]
                    # For mimi decoder we need [batch, codebooks, frames]
                    if audio_tokens.shape[0] == 32 or (hasattr(self.model, '_model') and 
                                                      hasattr(self.model._model, 'args') and 
                                                      audio_tokens.shape[0] == self.model._model.args.audio_num_codebooks):
                        # This is [codebooks, frames]
                        audio_tokens = audio_tokens.unsqueeze(0)  # Add batch dim
                    else:
                        # This is probably [batch, tokens] or some other format
                        # Reshape to ensure 3 dimensions
                        shape0 = audio_tokens.shape[0]
                        shape1 = audio_tokens.shape[1]
                        if shape1 % 32 == 0:
                            # This might be flattened
                            audio_tokens = audio_tokens.reshape(shape0, 32, -1)
                        else:
                            # Last resort: reshape to standard 32 codebooks
                            audio_tokens = audio_tokens.reshape(1, 32, -1)
                
                # Final check of shape for decode call
                if len(audio_tokens.shape) != 3:
                    if self.debug:
                        print(f"Reshaping tokens to 3D: {audio_tokens.shape}")
                    audio_tokens = audio_tokens.reshape(1, 32, -1)  # Force to 3D with 32 codebooks
                
                if self.debug:
                    print(f"Final tokens shape for decode: {audio_tokens.shape}")
                
                # Call decode with the properly shaped tokens
                audio = self.model._audio_tokenizer.decode(audio_tokens)
                
                # Handle different return shapes
                if isinstance(audio, torch.Tensor):
                    if len(audio.shape) > 1:
                        audio = audio.squeeze(0)
                        if len(audio.shape) > 1:
                            audio = audio.squeeze(0)
                    return audio.detach().cpu().numpy()
            except Exception as e:
                if self.debug:
                    print(f"Error using model._audio_tokenizer.decode: {e}")
                # Continue to other approaches
        
        # Use the model's native decoding method
        if hasattr(self.model, "decode_audio"):
            if self.debug:
                print("Using model.decode_audio method")
            audio = self.model.decode_audio(audio_tokens)
        elif hasattr(self.model, "vocoder"):
            # Try using a vocoder if available
            if self.debug:
                print("Using model.vocoder method")
            audio = self.model.vocoder(audio_tokens)
        elif hasattr(self.model, "decode") and callable(self.model.decode):
            # Try using a generic decode method
            if self.debug:
                print("Using model.decode method")
            audio = self.model.decode(audio_tokens)
        elif hasattr(self.model, "codec") and hasattr(self.model.codec, "decode"):
            # Try using a codec's decode method
            if self.debug:
                print("Using model.codec.decode method")
            audio = self.model.codec.decode(audio_tokens)
        elif audio_tokens is not None:
            # Try to use any methods available in the model to decode audio
            if hasattr(self.model, "codec") and hasattr(self.model.codec, "model") and hasattr(self.model.codec.model, "decode"):
                # Try accessing codec's internal model decode method
                if self.debug:
                    print("Using codec.model.decode method for audio decoding")
                audio = self.model.codec.model.decode(audio_tokens)
            elif hasattr(self.model, "get_audio"):
                # Try a get_audio method
                if self.debug:
                    print("Using model.get_audio method for decoding")
                audio = self.model.get_audio(audio_tokens)
            elif hasattr(self.model, '_model') and hasattr(self.model._model, 'decode'):
                # Try using the underlying model's decode method
                if self.debug:
                    print("Using model._model.decode method")
                audio = self.model._model.decode(audio_tokens)
            else:
                # Finally, as a last resort, create a dummy sine wave for debugging
                print("WARNING: No decoding method found, returning dummy audio")
                # Create a more complex sine wave as dummy audio
                sample_rate = 24000
                duration = 3.0  # seconds
                frequencies = [440, 550, 660]  # Multiple frequencies for a more interesting sound
                t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                audio = np.zeros_like(t)
                for i, freq in enumerate(frequencies):
                    audio += 0.2 * np.sin(2 * np.pi * freq * t + i*0.5)
                # Apply fade in/out
                fade_samples = int(0.1 * sample_rate)
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                audio[:fade_samples] *= fade_in
                audio[-fade_samples:] *= fade_out
        else:
            raise ValueError("Model must have decode_audio, vocoder, decode method, or codec.decode")
                
        # Convert to NumPy array if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
            
        return audio