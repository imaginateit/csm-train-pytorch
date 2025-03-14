# Command-Line Interface

CSM provides a command-line interface for generating speech and verifying audio watermarks.

## Installation

To use the CSM command-line tools, install the package:

```bash
pip install -e .
```

This will make the `csm-generate` and `csm-verify` commands available in your environment.

## Generating Speech

The `csm-generate` command allows you to generate speech from text using the CSM model.

### Basic Usage

Generate speech from text:

```bash
csm-generate --text "Hello, this is a test."
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
--device DEVICE           Device to run inference on (default: cuda if available, else cpu)
```

### Using Context

You can provide context to the model to make the generated speech more natural:

```bash
csm-generate \
  --text "I'm doing well, thank you." \
  --context-audio utterance_1.wav \
  --context-text "Hello, how are you doing today?" \
  --context-speaker 1
```

You can provide multiple context segments:

```bash
csm-generate \
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