# Command-Line Interface

CSM provides a command-line interface for generating speech and verifying audio watermarks.

## Installation

To use the CSM command-line tools, install the package:

```bash
pip install -e .
```

For Apple Silicon acceleration (recommended for Mac users):

```bash
pip install -e ".[apple]"
```

This will make the command-line tools available in your environment.

## Generating Speech

CSM provides two commands for generating speech from text:

- `csm-generate`: Standard version (works on all platforms, CUDA-optimized on NVIDIA GPUs)
- `csm-generate-mlx`: MLX-accelerated version for Apple Silicon Macs

### Basic Usage

Generate speech from text on Mac with Apple Silicon (recommended for Mac users):

```bash
csm-generate-mlx --text "Hello, this is a test."
```

Generate speech on other platforms:

```bash
csm-generate --text "Hello, this is a test."
```

These commands will:
1. Download the model checkpoint from HuggingFace (on first run)
2. Generate audio for the provided text
3. Save the audio to `audio.wav` in the current directory

### Options

```
--model-path PATH         Path to the model checkpoint
--text TEXT               Text to generate speech for
--speaker ID              Speaker ID (default: 0)
--output PATH             Output file path (default: audio.wav)
--max-audio-length-ms MS  Maximum audio length in milliseconds (default: 10000)
--temperature TEMP        Sampling temperature (default: 0.9)
--topk K                  Top-k sampling parameter (default: 50)
--debug                   Enable debug mode with more detailed output
```

The `csm-generate` command also accepts a `--device` parameter to select between "cuda" and "cpu".

### Using Context

You can provide context to the model to make the generated speech more natural:

```bash
csm-generate-mlx \
  --text "I'm doing well, thank you." \
  --context-audio utterance_1.wav \
  --context-text "Hello, how are you doing today?" \
  --context-speaker 1
```

You can provide multiple context segments:

```bash
csm-generate-mlx \
  --text "Me too, this is some cool stuff huh?" \
  --speaker 1 \
  --context-audio utterance_1.wav utterance_2.wav utterance_3.wav \
  --context-text "Hey how are you doing." "Pretty good, pretty good." "I'm great." \
  --context-speaker 0 1 0
```

## Verifying Watermarks

The `csm-verify` command allows you to check if an audio file contains the CSM watermark.

### Usage

```bash
csm-verify --audio-path audio.wav
```

This will analyze the audio file and report whether it contains a CSM watermark.

## Platform-Specific Information

### Mac Users

Mac users have two options, in order of preference:

1. **MLX Acceleration** (recommended for Apple Silicon): Install with `pip install -e ".[apple]"` and use `csm-generate-mlx` for best performance
2. **PyTorch**: Use standard `csm-generate` (slower)

### Linux/Windows with NVIDIA GPU

Use the standard `csm-generate` command with `--device cuda` for best performance.