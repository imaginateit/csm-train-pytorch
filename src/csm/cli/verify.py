"""Command-line interface for watermark verification."""

import argparse

from ..watermarking.utils import check_audio_from_file


def main():
    """Main entry point for the watermark verification CLI."""
    parser = argparse.ArgumentParser(description="Verify audio watermarks in CSM-generated files")
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to the audio file to verify",
    )

    args = parser.parse_args()
    check_audio_from_file(args.audio_path)


if __name__ == "__main__":
    main()