# CSM Fine-Tuning Examples

This directory contains example scripts for fine-tuning CSM (Conversational Speech Model) models using LoRA with MLX on Apple Silicon.

## Available Examples

### Basic Examples

- **[basic_finetuning.sh](./basic_finetuning.sh)**: Simple LoRA fine-tuning with default parameters
- **[advanced_finetuning.sh](./advanced_finetuning.sh)**: Advanced fine-tuning with custom parameters
- **[resume_finetuning.sh](./resume_finetuning.sh)**: Resuming training from a checkpoint

### Specialized Examples

- **[expressive_style_transfer.sh](./expressive_style_transfer.sh)**: Fine-tuning for expressive style transfer
- **[low_resource_finetuning.sh](./low_resource_finetuning.sh)**: Fine-tuning with limited data and compute
- **[multi_speaker_finetuning.sh](./multi_speaker_finetuning.sh)**: Training multiple speakers in a single run
- **[huggingface_lora_finetune.py](./huggingface_lora_finetune.py)**: Fine-tuning with data from Hugging Face Hub

### Performance Optimization

- **[benchmark_and_optimize.sh](./benchmark_and_optimize.sh)**: Benchmarking and optimizing LoRA configurations
- **[test_lora.py](./test_lora.py)**: Simple test script for LoRA functionality

## Configuration Files

- **[speakers_config.json](./speakers_config.json)**: Example configuration file for multi-speaker fine-tuning
- **[huggingface_lora_finetuning.md](./huggingface_lora_finetuning.md)**: Documentation for Hugging Face integration

## Example Types

### Shell Scripts (*.sh)

These are ready-to-use shell scripts that demonstrate various fine-tuning scenarios. You can modify the parameters directly in the scripts or run them with your own command-line arguments.

### Python Scripts (*.py)

These are more comprehensive Python scripts that provide deeper customization and integration with external tools and datasets. They can be run directly with Python and support various command-line arguments.

### Documentation Files (*.md)

These provide detailed explanations and usage instructions for the example scripts.

## Running the Examples

### Shell Scripts

1. Make the script executable: `chmod +x example_script.sh`
2. Run the script: `./example_script.sh`

### Python Scripts

Run with Python:
```bash
python huggingface_lora_finetune.py --model-path /path/to/model.safetensors --output-dir ./output
```

## Full Example Workflow

For a complete fine-tuning workflow:

1. First benchmark different LoRA configurations:
   ```bash
   ./benchmark_and_optimize.sh
   ```

2. Choose the optimal configuration and run basic fine-tuning:
   ```bash
   ./basic_finetuning.sh
   ```

3. For more control, use the advanced fine-tuning script:
   ```bash
   ./advanced_finetuning.sh
   ```

4. For multiple speakers, prepare a speakers_config.json file and run:
   ```bash
   ./multi_speaker_finetuning.sh
   ```

5. To use data from external sources, use the Hugging Face integration:
   ```bash
   python huggingface_lora_finetune.py --model-path /path/to/model.safetensors --dataset example/dataset
   ```

## Advanced Usage

For advanced use cases such as custom adaptation of specific model components or integration with external pipelines, refer to the Python API documentation in the `docs/` directory.

## Usage

1. Make sure you have the CSM package installed with all dependencies
2. Modify the example scripts to use your model and data paths
3. Make the scripts executable: `chmod +x *.sh`
4. Run an example: `./basic_finetuning.sh`

## Data Structure

The examples expect data organized as follows:

```
data/
├── speaker1/
│   ├── audio/
│   │   ├── sample1.wav
│   │   ├── sample2.wav
│   │   └── ...
│   ├── transcripts/
│   │   ├── sample1.txt
│   │   ├── sample2.txt
│   │   └── ...
│   └── alignments/  (optional)
│       ├── sample1.json
│       ├── sample2.json
│       └── ...
├── speaker2/
│   └── ...
└── ...
```

## Multi-Speaker Configuration

The `speakers_config.json` file contains configuration for multiple speakers. You can customize each speaker's parameters:

```json
[
  {
    "name": "speaker1",
    "speaker_id": 1,
    "audio_dir": "./data/speaker1/audio",
    "transcript_dir": "./data/speaker1/transcripts",
    "alignment_dir": "./data/speaker1/alignments",
    "lora_r": 8,
    "learning_rate": 1e-4,
    "epochs": 5
  },
  {
    "name": "speaker2",
    ...
  }
]
```

## Performance Tips

1. Start with a low LoRA rank (r=8) and only include q_proj and v_proj in target modules
2. For expressive speech, increase rank and include more attention modules
3. Use the benchmark script to find optimal settings for your hardware
4. Memory usage scales with batch size and sequence length - adjust as needed
5. For Apple Silicon Macs:
   - M1: Use batch size 1-2, rank 4-8
   - M2: Use batch size 2-4, rank 8-16 
   - M3: Use batch size 4-8, rank 16-32

## Additional Documentation

For more detailed information on LoRA fine-tuning options, see the [LoRA fine-tuning documentation](../docs/reference/sesame_csm/lora_finetuning.md).