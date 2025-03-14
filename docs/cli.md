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

Generate speech with a specific voice preset:

```bash
csm-generate-mlx --text "Hello, this is a test." --voice warm
```

Generate speech on other platforms:

```bash
csm-generate --text "Hello, this is a test."
```

Generate speech with a specific voice preset on CUDA:

```bash
csm-generate --text "Hello, this is a test." --voice deep --device cuda
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
--voice PRESET            Voice preset (alternative to --speaker)
--output PATH             Output file path (default: audio.wav)
--max-audio-length-ms MS  Maximum audio length in milliseconds (default: 10000)
--temperature TEMP        Sampling temperature (default: 0.9)
--topk K                  Top-k sampling parameter (default: 50)
--debug                   Enable debug mode with more detailed output
```

The `csm-generate` command also accepts a `--device` parameter to select between "cuda" and "cpu".

### Voice Presets

Instead of using numeric speaker IDs, you can select a voice using the `--voice` parameter. This works with both `csm-generate` and `csm-generate-mlx`:

```bash
csm-generate-mlx --text "Hello, this is a test." --voice warm
csm-generate --text "Hello, this is a test." --voice deep
```

Available voice presets:
- `neutral` - Balanced, default voice
- `warm` - Warmer, friendlier tone
- `deep` - Deeper voice
- `bright` - Brighter, higher pitch
- `soft` - Softer, more gentle voice
- `energetic` - More energetic/animated
- `calm` - Calmer, measured tone
- `clear` - Clearer articulation
- `resonant` - More resonant voice
- `authoritative` - More authoritative tone

### Using Context

You can provide context to the model to make the generated speech more natural:

```bash
csm-generate-mlx \
  --text "I'm doing well, thank you." \
  --voice warm \
  --context-audio utterance_1.wav \
  --context-text "Hello, how are you doing today?" \
  --context-speaker 1
```

You can provide multiple context segments:

```bash
csm-generate-mlx \
  --text "Me too, this is some cool stuff huh?" \
  --voice deep \
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
   - This implementation uses MLX to accelerate model inference on Apple Silicon GPUs
   - MLX provides optimized tensor operations specifically designed for Apple devices
   - Automatically falls back to the PyTorch implementation if MLX is not available
2. **PyTorch**: Use standard `csm-generate` (slower on Mac)

### Linux/Windows with NVIDIA GPU

Use the standard `csm-generate` command with `--device cuda` for best performance.

### Known Issues

- **Warning messages**: You may see FutureWarning messages related to torch.load. These are harmless and will be addressed in a future PyTorch version.
- **MLX fallback**: If MLX is not available, the MLX command will automatically fall back to the standard implementation.