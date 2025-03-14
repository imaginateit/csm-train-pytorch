"""Command-line interface for generation with CSM."""

import argparse
import os
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from ..generator import Segment, load_csm_1b

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


def main():
    """Main entry point for the generation CLI."""
    parser = argparse.ArgumentParser(description="Generate speech with CSM")
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=f"Device to run inference on (default: {'cuda' if torch.cuda.is_available() else 'cpu'})",
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
        model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
        print(f"Downloaded model to {model_path}")

    # Load the model
    print(f"Loading model from {model_path} to {args.device}...")
    generator = load_csm_1b(model_path, args.device)
    print("Model loaded successfully")

    # Prepare context segments if provided
    context = []
    if args.context_audio:
        if not (args.context_text and args.context_speaker):
            raise ValueError(
                "If context audio is provided, context text and speaker must also be provided"
            )
        if not (
            len(args.context_audio) == len(args.context_text) == len(args.context_speaker)
        ):
            raise ValueError(
                "The number of context audio, text, and speaker entries must be the same"
            )

        for audio_path, text, speaker in zip(
            args.context_audio, args.context_text, args.context_speaker
        ):
            print(f"Loading context audio: {audio_path}")
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0),
                orig_freq=sample_rate,
                new_freq=generator.sample_rate,
            )
            context.append(
                Segment(text=text, speaker=speaker, audio=audio_tensor)
            )

    # Generate audio
    print(f"Generating audio for text: '{args.text}'")
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
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torchaudio.save(
        output_path,
        audio.unsqueeze(0).cpu(),
        generator.sample_rate,
    )
    print(f"Audio saved to {output_path}")
    print(f"Sample rate: {generator.sample_rate} Hz")


if __name__ == "__main__":
    main()