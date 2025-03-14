import argparse

import torch
import torchaudio

from .silentcipher import get_model
from . import CSM_1B_GH_WATERMARK


def load_watermarker(device: str = "cuda") -> object:
    """
    Load the watermarking model.
    
    Args:
        device (str): The device to load the model on.
        
    Returns:
        The watermarking model.
    """
    model = get_model(
        model_type="44.1k",
        device=device,
    )
    return model


@torch.inference_mode()
def watermark(
    watermarker: object,
    audio_array: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> tuple[torch.Tensor, int]:
    """
    Apply a watermark to an audio array.
    
    Args:
        watermarker: The watermarking model.
        audio_array: The audio array to watermark.
        sample_rate: The sample rate of the audio array.
        watermark_key: The watermark key to use.
        
    Returns:
        A tuple of the watermarked audio array and the output sample rate.
    """
    audio_array_44khz = torchaudio.functional.resample(audio_array, orig_freq=sample_rate, new_freq=44100)
    encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=36)

    output_sample_rate = min(44100, sample_rate)
    encoded = torchaudio.functional.resample(encoded, orig_freq=44100, new_freq=output_sample_rate)
    return encoded, output_sample_rate


@torch.inference_mode()
def verify(
    watermarker: object,
    watermarked_audio: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> bool:
    """
    Verify if an audio array has a watermark.
    
    Args:
        watermarker: The watermarking model.
        watermarked_audio: The audio array to check.
        sample_rate: The sample rate of the audio array.
        watermark_key: The watermark key to check for.
        
    Returns:
        True if the audio array has the watermark, False otherwise.
    """
    watermarked_audio_44khz = torchaudio.functional.resample(watermarked_audio, orig_freq=sample_rate, new_freq=44100)
    result = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=True)

    is_watermarked = result["status"]
    if is_watermarked:
        is_csm_watermarked = result["messages"][0] == watermark_key
    else:
        is_csm_watermarked = False

    return is_watermarked and is_csm_watermarked


def check_audio_from_file(audio_path: str) -> None:
    """
    Check if an audio file has a watermark.
    
    Args:
        audio_path: The path to the audio file.
    """
    watermarker = load_watermarker(device="cuda")

    audio_array, sample_rate = load_audio(audio_path)
    is_watermarked = verify(watermarker, audio_array, sample_rate, CSM_1B_GH_WATERMARK)

    outcome = "Watermarked" if is_watermarked else "Not watermarked"
    print(f"{outcome}: {audio_path}")


def load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    """
    Load an audio file.
    
    Args:
        audio_path: The path to the audio file.
        
    Returns:
        A tuple of the audio array and sample rate.
    """
    audio_array, sample_rate = torchaudio.load(audio_path)
    audio_array = audio_array.mean(dim=0)
    return audio_array, int(sample_rate)


def cli() -> None:
    """Command line interface for checking audio watermarks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    args = parser.parse_args()

    check_audio_from_file(args.audio_path)


if __name__ == "__main__":
    cli()