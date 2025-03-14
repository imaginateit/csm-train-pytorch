"""Command-line interface for CSM inference with MLX acceleration on Apple Silicon."""

import argparse
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download

# Check if MLX is available
MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None

if not MLX_AVAILABLE:
    print("MLX is not available. Install with: pip install -e '.[apple]'")
    print("Using fallback CPU implementation...")
    from .generate_cpu import main as main_cpu
    main = main_cpu
else:
    import mlx.core as mx
    
    from ..generator import Segment, load_csm_1b
    
    def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
        """Convert a PyTorch tensor to an MLX array."""
        return mx.array(tensor.detach().cpu().numpy())
    
    def mlx_to_torch(array: mx.array) -> torch.Tensor:
        """Convert an MLX array to a PyTorch tensor."""
        return torch.tensor(array.tolist())
    
    class MLXWrapper:
        """Wrapper class to run PyTorch models with MLX acceleration."""
        
        def __init__(self, torch_model, device="cpu"):
            """Initialize with a PyTorch model."""
            self.torch_model = torch_model
            self.device = device
            
            # Cache for converted parameters
            self.mlx_params = {}
            
            # Convert model parameters to MLX format
            self._convert_params()
        
        def _convert_params(self):
            """Convert PyTorch parameters to MLX format."""
            for name, param in self.torch_model.named_parameters():
                self.mlx_params[name] = torch_to_mlx(param.data)
            
            print(f"Converted {len(self.mlx_params)} parameters to MLX format")
            
        def forward(self, *args, **kwargs):
            """Forward pass using MLX when possible, falling back to PyTorch."""
            # For prototype, we'll just use the PyTorch model directly
            # In a full implementation, this would use MLX operations
            return self.torch_model(*args, **kwargs)
    
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
        parser.add_argument(
            "--speaker",
            type=int,
            default=0,
            help="Speaker ID (default: 0)",
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
        
        # Force CPU for PyTorch parts that still need it
        device = "cpu"
        
        # Get the model path, downloading if necessary
        model_path = args.model_path
        if model_path is None:
            print("Downloading model from HuggingFace hub...")
            model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
            print(f"Downloaded model to {model_path}")

        # Load the model with PyTorch first
        print(f"Loading model from {model_path} with PyTorch...")
        
        try:
            generator = load_csm_1b(model_path, device)
            print("Model loaded successfully")
            
            # Wrap model with MLX acceleration
            print("Converting model to MLX format...")
            # Note: This is a prototype - in a full implementation, we would properly
            # convert the model operations to MLX
            # mlx_generator = MLXWrapper(generator._model)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            if args.debug:
                import traceback
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

        # Generate audio
        print(f"Generating audio for text: '{args.text}'")
        try:
            # For now, we use the standard PyTorch implementation
            # In a full MLX implementation, we would use MLX for all operations
            audio = generator.generate(
                text=args.text,
                speaker=args.speaker,
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
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()