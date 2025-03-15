# LoRA Fine-tuning with MLX

This guide explains how to fine-tune CSM (Conversational Speech Model) models using LoRA (Low-Rank Adaptation) with MLX on Apple Silicon.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that significantly reduces the number of trainable parameters by adding small, trainable "adapter" modules to existing weights in a model while keeping the original weights frozen.

Benefits of LoRA:
- Reduces memory requirements (only updating a small fraction of parameters)
- Faster training and inference
- Multiple fine-tuned versions can be stored compactly
- Can be merged with original weights for deployment
- State-of-the-art performance for voice adaptation

## Requirements

- Apple Silicon Mac (M1/M2/M3 series)
- Python 3.11+
- MLX installed (`pip install mlx`)
- A pre-trained CSM model (safetensors or PyTorch format)
- Audio data for fine-tuning

## Installation

Ensure you have the CSM package installed with development dependencies:

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e ".[dev]"
```

## Preparing Your Data

LoRA fine-tuning requires:
1. Audio files (WAV format)
2. Transcript files (matching filenames with .txt extension)
3. Optional alignment files (matching filenames with .json extension)

Directory structure example:
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
└── alignments/  (optional)
    ├── sample1.json
    ├── sample2.json
    └── ...
```

Each transcript file should contain the text corresponding to the audio file. Alignment files provide word-level timing information, which can improve the quality of fine-tuning but are optional.

## Basic Usage

The simplest way to fine-tune a CSM model with LoRA is using the provided CLI:

```bash
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_model \
  --audio-dir /path/to/audio \
  --transcript-dir /path/to/transcripts \
  --speaker-id 0 \
  --batch-size 2 \
  --epochs 5
```

This will fine-tune the model using default LoRA parameters (rank=8, alpha=16) and save the resulting model in the specified output directory.

## Advanced Configuration

### LoRA Parameters

Customize the LoRA adapter configuration:

```bash
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_model \
  --audio-dir /path/to/audio \
  --transcript-dir /path/to/transcripts \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules q_proj v_proj o_proj \
  --target-layers 0 1 2 3
```

Parameters explained:
- `lora-r`: LoRA rank (higher means more capacity but more parameters)
- `lora-alpha`: LoRA scaling factor (typically 2×r)
- `lora-dropout`: Dropout probability for LoRA layers
- `target-modules`: Which modules to apply LoRA to (options: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- `target-layers`: Layer indices to apply LoRA to (default: all layers)
- `lora-bias`: Add trainable bias terms to LoRA layers

### Training Parameters

Fine-tune the training process:

```bash
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_model \
  --audio-dir /path/to/audio \
  --transcript-dir /path/to/transcripts \
  --learning-rate 5e-5 \
  --semantic-weight 120.0 \
  --acoustic-weight 1.0 \
  --weight-decay 0.01 \
  --batch-size 4 \
  --epochs 10 \
  --val-split 0.1 \
  --val-every 50 \
  --save-every 200 \
  --max-grad-norm 1.0
```

Parameters explained:
- `learning-rate`: Learning rate for Adam optimizer
- `semantic-weight`: Weight for semantic token loss (codebook 0)
- `acoustic-weight`: Weight for acoustic token loss (other codebooks)
- `weight-decay`: Weight decay for optimizer
- `batch-size`: Batch size for training
- `epochs`: Number of epochs to train
- `val-split`: Validation split ratio
- `val-every`: Validate every N steps
- `save-every`: Save checkpoint every N steps
- `max-grad-norm`: Maximum gradient norm for clipping

### Data Processing

Configure data processing options:

```bash
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_model \
  --audio-dir /path/to/audio \
  --transcript-dir /path/to/transcripts \
  --alignment-dir /path/to/alignments \
  --speaker-id 1 \
  --max-seq-len 1024 \
  --context-turns 3
```

Parameters explained:
- `alignment-dir`: Directory containing alignment files (optional)
- `speaker-id`: Speaker ID to use for training (default: 0)
- `max-seq-len`: Maximum sequence length (default: 2048)
- `context-turns`: Number of context turns to include (default: 2)

### Output Options

Configure saving options:

```bash
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_model \
  --audio-dir /path/to/audio \
  --transcript-dir /path/to/transcripts \
  --save-mode both \
  --generate-samples \
  --sample-prompt "This is a test of my new voice model." \
  --log-level debug
```

Parameters explained:
- `save-mode`: How to save the fine-tuned model (options: lora, full, both)
  - `lora`: Save only LoRA parameters (default, smallest files)
  - `full`: Save the full model with merged weights
  - `both`: Save both LoRA parameters and merged model
- `generate-samples`: Generate audio samples after training
- `sample-prompt`: Prompt for sample generation
- `log-level`: Logging level (options: debug, info, warning, error, critical)
- `debug`: Enable debug mode with more verbose logging

## Resuming Training

To resume training from a checkpoint:

```bash
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_model \
  --audio-dir /path/to/audio \
  --transcript-dir /path/to/transcripts \
  --resume-from ./fine_tuned_model/checkpoint_latest.safetensors
```

## Using LoRA Models in Python

You can load and use LoRA fine-tuned models programmatically:

```python
import mlx.core as mx
from csm.mlx.components.model_wrapper import MLXModelWrapper
from csm.mlx.components.lora import apply_lora_to_model
from csm.training.lora_trainer import CSMLoRATrainer

# Load base model
model_args = {
    "backbone_flavor": "llama-1B",
    "decoder_flavor": "llama-100M",
    "text_vocab_size": 128256,
    "audio_vocab_size": 2051,
    "audio_num_codebooks": 32
}
model = MLXModelWrapper(model_args)

# Create trainer and load LoRA weights
trainer = CSMLoRATrainer(
    model_path="/path/to/base_model.safetensors",
    output_dir="./output",
    lora_r=8,
    lora_alpha=16.0
)

# Load LoRA weights
trainer.load_lora_weights("/path/to/lora_weights.safetensors")

# Generate audio with the fine-tuned model
trainer.generate_sample(
    text="This is a test of the fine-tuned voice model.",
    speaker_id=0,
    output_path="./sample.wav"
)
```

## Advanced Use Cases

### Multiple Speaker Adaptation

To fine-tune for multiple speakers, organize your data by speaker and run separate fine-tuning jobs:

```bash
# Fine-tune for speaker 1
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./speaker1_model \
  --audio-dir /path/to/speaker1/audio \
  --transcript-dir /path/to/speaker1/transcripts \
  --speaker-id 1

# Fine-tune for speaker 2
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./speaker2_model \
  --audio-dir /path/to/speaker2/audio \
  --transcript-dir /path/to/speaker2/transcripts \
  --speaker-id 2
```

### Style Transfer

For style transfer, prepare data that exemplifies the target style:

```bash
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./emotional_style \
  --audio-dir /path/to/emotional/audio \
  --transcript-dir /path/to/emotional/transcripts \
  --speaker-id 0 \
  --lora-r 32 \  # Higher rank for better style capture
  --lora-alpha 64 \
  --target-modules q_proj v_proj o_proj  # Include more modules for style
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Reduce LoRA rank (--lora-r)
   - Target fewer layers (--target-layers)
   - Target fewer modules (--target-modules)

2. **Poor Audio Quality**
   - Increase LoRA rank
   - Include more target modules
   - Increase training epochs
   - Ensure audio quality is good in training data

3. **Loss Not Decreasing**
   - Adjust learning rate
   - Check data quality
   - Increase semantic and acoustic weights
   - Ensure alignment between audio and transcripts

4. **Import Errors**
   - Ensure MLX is installed and up to date
   - Check you're running on Apple Silicon
   - Verify all dependencies are installed

### Performance Tips

1. Use a higher batch size if memory allows (--batch-size 4 or higher)
2. Target only the most important modules (q_proj and v_proj work well)
3. For memory constraints, target only upper layers (--target-layers 16 17 18 19)
4. Balance lora-r and lora-alpha (typically alpha = 2×r)

## Future Improvements

Future versions of the LoRA implementation may include:
- QLoRA support for even more memory-efficient fine-tuning
- Multi-speaker training in a single run
- Automatic hyperparameter optimization
- Integration with external tools for better alignment
- Prompt-conditioned LoRA for style control

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [CSM Documentation](https://github.com/conversationalspeechmodel/csm)