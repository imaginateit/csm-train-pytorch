"""Command-line interface for generation with CSM on CPU."""

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

# Add patch for moshi to work without bitsandbytes
def patch_moshi():
    """Apply patch to moshi to work without GPU dependencies."""
    import moshi.utils.quantize as quantize
    
    # Save the original linear function
    original_linear = getattr(quantize, "linear", None)
    
    # Define a new linear function that skips bitsandbytes
    def patched_linear(module, input_tensor, weight_name, bias_name=None):
        weight = getattr(module, weight_name)
        bias = getattr(module, bias_name) if bias_name is not None else None
        return torch.nn.functional.linear(input_tensor, weight, bias)
    
    # Replace the linear function with our patched version
    setattr(quantize, "linear", patched_linear)

# Apply the patch before importing from generator
patch_moshi()

from ..generator import Segment, load_csm_1b


def main():
    """Main entry point for the CPU generation CLI."""
    # Suppress FutureWarnings for torch.load
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
    
    parser = argparse.ArgumentParser(description="Generate speech with CSM (CPU version)")
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

    args = parser.parse_args()

    # Force CPU for this version
    device = "cpu"
    
    # Get the model path, downloading if necessary
    model_path = args.model_path
    if model_path is None:
        print("Downloading model from HuggingFace hub...")
        model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
        print(f"Downloaded model to {model_path}")

    # Load the model
    print(f"Loading model from {model_path} to {device}...")
    
    try:
        generator = load_csm_1b(model_path, device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
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
                sys.exit(1)

    # Generate audio
    print(f"Generating audio for text: '{args.text}'")
    try:
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
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torchaudio.save(
            output_path,
            audio.unsqueeze(0).cpu(),
            generator.sample_rate,
        )
        print(f"Audio saved to {output_path}")
        print(f"Sample rate: {generator.sample_rate} Hz")
    except Exception as e:
        print(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()