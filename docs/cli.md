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

For fine-tuning capabilities, install with training dependencies:

```bash
# Basic training (all platforms)
pip install -e ".[train]"

# Training with Apple Silicon acceleration
pip install -e ".[train,apple]"
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
   - Implements a full MLX transformer architecture for maximum performance
   - Has three acceleration modes that automatically activate based on your device:
     - **Pure MLX Mode**: Full MLX transformer for maximum speed (Apple Silicon only)
     - **Hybrid Mode**: MLX for embedding and sampling operations with PyTorch transformers
     - **PyTorch Mode**: Automatic fallback when MLX is not available
   - Up to 2-3x faster on Apple Silicon compared to PyTorch
   - Show detailed performance metrics with `--debug` flag
2. **PyTorch**: Use standard `csm-generate` (slower on Mac)

### Linux/Windows with NVIDIA GPU

Use the standard `csm-generate` command with `--device cuda` for best performance.

### Fine-Tuning Models

CSM provides two commands for fine-tuning models:

- `csm-train`: Standard training on all platforms (CPU or GPU)
- `csm-train-mlx`: MLX-accelerated training for Apple Silicon Macs

### Data Preparation

For fine-tuning, you'll need:

1. **Audio files**: .wav format, ideally 24kHz sample rate
2. **Transcript files**: .txt files with matching filenames to audio
3. **Optional alignments**: .json files with word-level timings

Directory structure:
```
data/
  ├── audio/
  │   ├── sample1.wav
  │   ├── sample2.wav
  │   └── ...
  ├── transcripts/
  │   ├── sample1.txt
  │   ├── sample2.txt
  │   └── ...
  └── alignments/
      ├── sample1.json  # optional
      ├── sample2.json  # optional
      └── ...
```

### Basic Training Usage

Start training with PyTorch (all platforms):

```bash
csm-train \
  --model-path /path/to/csm_1b.pt \
  --audio-dir data/audio \
  --transcript-dir data/transcripts \
  --output-dir my_finetuned_model \
  --speaker-id 5
```

Start training with MLX (Apple Silicon only):

```bash
csm-train-mlx \
  --model-path /path/to/csm_1b.pt \
  --audio-dir data/audio \
  --transcript-dir data/transcripts \
  --output-dir my_finetuned_model \
  --speaker-id 5 \
  --autotune
```

### Training Options

#### Data Options
- `--model-path PATH`: Path to CSM model checkpoint (required)
- `--audio-dir DIR`: Directory with audio files (required)
- `--transcript-dir DIR`: Directory with transcript files (required)
- `--alignment-dir DIR`: Directory with word-level alignments
- `--speaker-id ID`: Speaker ID to assign (0-9, default: 0)
- `--val-split RATIO`: Portion of data for validation (0.0-1.0, default: 0.1)

#### Training Configuration
- `--learning-rate RATE`: Base learning rate (default: 1e-5)
- `--epochs NUM`: Number of training epochs (default: 5)
- `--batch-size SIZE`: Batch size for training (default: 2)
- `--accumulation-steps STEPS`: Gradient accumulation steps (default: 4)
- `--semantic-weight VAL`: Weight for semantic token loss (default: 100.0)
- `--acoustic-weight VAL`: Weight for acoustic token loss (default: 1.0)
- `--freeze-backbone`: Freeze backbone parameters
- `--freeze-decoder`: Freeze decoder parameters
- `--freeze-embeddings`: Freeze embedding parameters
- `--resume-from PATH`: Path to checkpoint to resume from

#### MLX-Specific Options
- `--autotune`: Enable MLX kernel autotuning
- `--num-threads NUM`: Number of threads for MLX operations (default: 4)

### Fine-Tuning Strategies

#### Voice Adaptation

```bash
csm-train \
  --model-path /path/to/csm_1b.pt \
  --audio-dir my_voice_samples \
  --transcript-dir my_voice_transcripts \
  --output-dir my_voice_model \
  --speaker-id 5 \
  --freeze-backbone \
  --learning-rate 2e-5 \
  --semantic-weight 20.0 \
  --epochs 10
```

#### Style Tuning

```bash
csm-train \
  --model-path /path/to/csm_1b.pt \
  --audio-dir excited_style_samples \
  --transcript-dir excited_style_transcripts \
  --output-dir excited_style_model \
  --speaker-id 3 \
  --learning-rate 1e-5 \
  --semantic-weight 50.0 \
  --epochs 5
```

## Known Issues

- **Warning messages**: You may see FutureWarning messages related to torch.load. These are harmless and will be addressed in a future PyTorch version.
- **MLX fallback**: If MLX is not available, the MLX command will automatically fall back to the standard implementation.
- **MLX training implementation**: The MLX training module is in early development. Performance may vary across different Apple Silicon devices.