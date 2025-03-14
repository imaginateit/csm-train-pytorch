# Command-Line Interface

CSM provides a command-line interface for generating speech and verifying audio watermarks.

## Installation

To use the CSM command-line tools, install the package:

```bash
pip install -e .
```

This will make the command-line tools available in your environment.

## Generating Speech

CSM provides two commands for generating speech from text:

- `csm-generate`: Standard version (requires CUDA GPU)
- `csm-generate-cpu`: CPU-compatible version for Mac and systems without specialized GPU libraries

### Basic Usage

Generate speech from text (CPU version):

```bash
csm-generate-cpu --text "Hello, this is a test."
```

This will:
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
```

The `csm-generate` command also accepts a `--device` parameter to select between "cuda" and "cpu", but `csm-generate-cpu` always uses the CPU.

### Using Context

You can provide context to the model to make the generated speech more natural:

```bash
csm-generate-cpu \
  --text "I'm doing well, thank you." \
  --context-audio utterance_1.wav \
  --context-text "Hello, how are you doing today?" \
  --context-speaker 1
```

You can provide multiple context segments:

```bash
csm-generate-cpu \
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

## Troubleshooting

### Mac/CPU Users

If you encounter errors related to missing CUDA or GPU libraries, use the `csm-generate-cpu` command instead of `csm-generate`. This version includes patches to work without GPU-specific libraries like `triton` and `bitsandbytes`.

### Common Errors

- **ModuleNotFoundError: No module named 'triton'** or **No module named 'bitsandbytes'**: Use the `csm-generate-cpu` command instead.
- **Error loading model**: Ensure your internet connection is working to download the model from HuggingFace.
- **CUDA/GPU errors**: If you're on a Mac or system without compatible GPU, use the CPU version with `csm-generate-cpu`.