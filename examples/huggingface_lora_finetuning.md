# Fine-tuning CSM with LoRA using Hugging Face Datasets

This example demonstrates how to fine-tune a CSM (Conversational Speech Model) using LoRA with voice data from the Hugging Face Hub. It provides a complete end-to-end workflow from data download to fine-tuning and sample generation.

## Requirements

- Apple Silicon Mac (M1/M2/M3 series)
- Python 3.11+
- MLX installed (`pip install mlx`)
- Hugging Face datasets library (`pip install datasets`)
- PyTorch and torchaudio for audio processing (`pip install torch torchaudio`)
- A pre-trained CSM model (safetensors format)

## Installation

Ensure you have all the required packages:

```bash
pip install mlx datasets huggingface_hub torchaudio librosa soundfile
```

## Usage

```bash
python huggingface_lora_finetune.py \
  --model-path /path/to/model.safetensors \
  --output-dir ./hf_finetuned_model \
  --dataset mozilla-foundation/common_voice_16_0 \
  --language en \
  --num-samples 100 \
  --lora-r 8 \
  --lora-alpha 16.0 \
  --batch-size 2 \
  --epochs 5
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--model-path` | Path to pretrained CSM model (safetensors format) |
| `--output-dir` | Directory to save fine-tuned model and outputs |
| `--dataset` | HuggingFace dataset to use (default: mozilla-foundation/common_voice_16_0) |
| `--language` | Language to filter the dataset (default: en) |
| `--num-samples` | Number of samples to use (default: 100) |
| `--speaker-id` | Speaker ID to use for training (default: 0) |
| `--lora-r` | LoRA rank (default: 8) |
| `--lora-alpha` | LoRA alpha scaling factor (default: 16.0) |
| `--batch-size` | Batch size (default: 2) |
| `--epochs` | Number of epochs (default: 5) |
| `--keep-data` | Keep downloaded data after training |
| `--log-level` | Logging level: debug, info, warning, error, critical (default: info) |

## How It Works

1. **Data Download**: Downloads speech data from the specified Hugging Face dataset.
2. **Data Preparation**: Processes the audio and transcript files into the required format.
3. **LoRA Fine-tuning**: Applies LoRA adapters to the base model and fine-tunes with the processed data.
4. **Sample Generation**: Creates a sample audio file with the fine-tuned model.

## Recommended Datasets

These Hugging Face datasets work well for fine-tuning speech models:

- `mozilla-foundation/common_voice_16_0`: Multi-language crowd-sourced voice dataset
- `openslr/librispeech_asr`: English audiobook recordings
- `facebook/voxpopuli`: Multi-language European Parliament recordings
- `jonatasgrosman/ljspeech_format`: Converted audiobooks in LJSpeech format

## Performance Tips

- Start with a small number of samples (50-100) to test the workflow
- For Apple Silicon Macs:
  - M1: Use batch size 1-2, rank 4-8
  - M2: Use batch size 2-4, rank 8-16
  - M3: Use batch size 4-8, rank 16-32
- Using a lower number of samples from a cleaner dataset often gives better results than using more samples from a noisy dataset

## What's Next

After fine-tuning, you can:

1. Use the fine-tuned model for voice generation with the provided CSM tools
2. Further customize the fine-tuning by targeting different modules with LoRA
3. Explore multi-speaker fine-tuning with `finetune_lora_multi.py`

## Troubleshooting

- If you encounter out-of-memory errors, reduce the batch size or LoRA rank
- For best results, ensure your audio files have consistent recording conditions
- If fine-tuning does not reflect the voice characteristics well, try increasing the number of epochs or targeting additional modules with LoRA

## Example Outputs

After successful fine-tuning, you'll find:

1. The fine-tuned model files in `output_dir`
2. A sample audio file (`sample.wav`) generated with the fine-tuned model
3. A log file with detailed training information