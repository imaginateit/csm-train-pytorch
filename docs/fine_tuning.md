# CSM Fine-Tuning Architecture

This document outlines the fine-tuning capabilities for CSM (Conversational Speech Model), providing a comprehensive framework for customizing the model for specific voices, styles, and domains.

## 1. Introduction and Design Goals

The fine-tuning system for CSM enables the following key capabilities:

1. **Voice Adaptation**: Fine-tune the model on custom voices to produce personalized speech synthesis
2. **Style Tuning**: Adjust speaking style characteristics (pace, emotion, formality)
3. **Domain Specialization**: Adapt the model for specific industries or content areas
4. **Multi-Speaker Management**: Maintain consistency across multiple speakers
5. **Resource Efficiency**: Enable fine-tuning with modest compute resources

## 2. System Architecture Overview

The fine-tuning system is structured in four progressive stages, each building on the previous:

```
1. Base Model → 2. Voice Adaptation → 3. Style Tuning → 4. Domain Specialization
```

Each stage can be performed independently depending on user needs.

## 3. Fine-Tuning Approaches

### 3.1 Voice Adaptation

Voice adaptation fine-tunes the model to mimic a specific voice's characteristics:

- **Data Requirements**: 5-10 minutes of high-quality speech from the target voice
- **Training Focus**: Primarily decoder and codebook projections
- **Recommended Settings**:
  - Freeze backbone: True (minimize impact on language capabilities)
  - Learning rate: 2e-5
  - Epochs: 5-10 depending on data quantity
  - Semantic token weight: 20.0
  - Acoustic token weight: 1.0

### 3.2 Style Tuning

Style tuning adjusts prosodic characteristics like emotion, pace, and formality:

- **Data Requirements**: 3-5 minutes of speech in the target style
- **Training Focus**: Both backbone and decoder
- **Recommended Settings**:
  - Freeze backbone: False (style is reflected in semantic tokens)
  - Learning rate: 1e-5
  - Epochs: 3-5
  - Semantic token weight: 50.0
  - Acoustic token weight: 1.0

### 3.3 Domain Specialization

Domain specialization improves pronunciation of domain-specific terminology:

- **Data Requirements**: 20+ minutes across multiple speakers
- **Training Focus**: Backbone with minimal decoder changes
- **Recommended Settings**:
  - Freeze decoder: True (maintain voice qualities)
  - Learning rate: 5e-6
  - Epochs: 5-8
  - Semantic token weight: 150.0
  - Acoustic token weight: 1.0

### 3.4 Conversational Tuning

Conversational tuning improves multi-turn dialogue capabilities:

- **Data Requirements**: Multi-speaker conversation recordings
- **Training Focus**: Context handling and responsiveness
- **Recommended Settings**:
  - Max context turns: 3-5
  - Learning rate: 2e-5
  - Epochs: 3
  - Include audio context: True

## 4. Data Preparation

### 4.1 Audio Processing

- **Sample Rate**: 24kHz (matching model's native rate)
- **Segmentation**: 10-second chunks with 2-second overlap
- **Normalization**: Audio normalized to consistent levels
- **Text Alignment**: Word-level alignments improve fine-tuning quality (optional)

### 4.2 Contextual Examples

For conversational fine-tuning, examples include previous turns as context:

- Multi-turn conversations are processed to create contextual examples
- Each example consists of context segments and a target segment
- Context segments include both text and audio when available
- This approach helps the model learn turn-taking and contextual responses

## 5. Training Methodology

### 5.1 Loss Function

The fine-tuning loss combines semantic and acoustic components:

```
Loss = (semantic_weight * semantic_token_loss) + (acoustic_weight * acoustic_token_loss)
```

- **Semantic Token Loss**: Cross-entropy for codebook 0 tokens (weighted heavily)
- **Acoustic Token Loss**: Cross-entropy for codebooks 1+ tokens

### 5.2 Optimization Strategy

Fine-tuning uses parameter-specific learning rates:

- **Backbone**: Lower learning rate (0.1x base LR)
- **Decoder**: Standard learning rate (1.0x base LR)
- **Embeddings**: Intermediate learning rate (0.5x base LR)
- **Optimizer**: AdamW with weight decay 0.01
- **Scheduler**: Cosine decay with warmup

### 5.3 Hardware Requirements

- **GPU**: RTX 3080/4080 or better (16GB+ VRAM)
- **Full Fine-tuning**: 16GB+ VRAM
- **Adapter-based Tuning**: 8GB+ VRAM
- **Training Time**: 2-8 hours depending on dataset size

## 6. Voice Profile Management

The system includes a comprehensive voice profile management system:

- **Profile Storage**: JSON-based profile database
- **Metadata**: Voice characteristics, training details, and inference parameters
- **Sample Generation**: Automated generation of sample audio for each profile
- **CLI Interface**: Commands for listing, creating, and managing voice profiles

## 7. Evaluation

### 7.1 Voice Similarity

- Reference-based comparison using speaker embeddings
- Subjective MOS (Mean Opinion Score) evaluation
- Prosodic feature comparison

### 7.2 Style Consistency

- Speech rate analysis
- Pitch variation measurement
- Energy profile comparison
- Pause pattern analysis

### 7.3 Domain Accuracy

- Term pronunciation verification
- Domain-specific metric improvement

## 8. CLI Usage Examples

### 8.1 Voice Adaptation

```bash
csm-finetune voice-adapt \
  --model-path /path/to/csm_1b.pt \
  --speaker-id 5 \
  --audio-dir /path/to/voice_samples \
  --transcript-dir /path/to/transcripts \
  --output-dir voice_adapted_model \
  --learning-rate 2e-5 \
  --epochs 5 \
  --freeze-backbone
```

### 8.2 Style Tuning

```bash
csm-finetune style-tune \
  --model-path /path/to/csm_1b.pt \
  --speaker-id 5 \
  --style-name "excited" \
  --audio-dir /path/to/style_samples \
  --transcript-dir /path/to/transcripts \
  --output-dir excited_style_model
```

### 8.3 Voice Profile Management

```bash
# List all voice profiles
csm voice list

# Create a new voice profile
csm voice create "John" 3 "Professional male voice" \
  --gender male \
  --age-range "30-40" \
  --accent "North American" \
  --model-path /path/to/fine_tuned_model.pt

# Generate a sample with a profile
csm voice sample 3 \
  --text "Hello, this is a demonstration of my voice."
```

## 9. API Integration

The fine-tuning system integrates with the existing CSM Python API:

```python
from csm.generator import load_csm_1b
from csm.training.finetuner import CSMFineTuner

# Load an existing fine-tuned model
generator = load_csm_1b("fine_tuned_model.pt", device="cuda")

# Generate with the fine-tuned model
audio = generator.generate(
    text="This is generated with my fine-tuned voice.",
    speaker=5,
    context=[],
    temperature=0.9
)

# Create a new fine-tuning job
fine_tuner = CSMFineTuner(
    model_path="csm_1b.pt",
    output_dir="my_custom_voice",
    learning_rate=2e-5
)

# Configure and run fine-tuning
fine_tuner.prepare_optimizer(freeze_backbone=True)
fine_tuner.train(
    train_dataset=my_dataset,
    batch_size=1,
    epochs=5
)
```

## 10. Best Practices

### 10.1 Data Quality

- Use studio-quality recordings when possible
- Ensure accurate transcriptions
- Remove background noise and normalize audio levels
- Include diverse sentence structures and phonetic coverage

### 10.2 Training Parameters

- **Voice Adaptation**: Higher decoder learning rate, lower backbone learning rate
- **Style Tuning**: Balanced learning rates, higher semantic weight
- **Domain Specialization**: Focus on backbone, use longer training

### 10.3 Avoiding Overfitting

- Start with minimal training epochs (3-5)
- Use validation split (10-20% of data)
- Monitor validation loss for early stopping
- Add slight regularization (weight decay)

### 10.4 Model Selection

- Generate samples throughout training to assess progress
- Choose checkpoint with best subjective quality
- Balance similarity to target with naturalness
- Test with out-of-domain text to ensure generalization

## 11. Limitations and Considerations

- Fine-tuning preserves the watermarking system for responsible use
- Voice cloning should only be performed with explicit consent
- Performance varies based on data quality and quantity
- Limited emotional range for extreme styles
- Domain specialization works best for technical terms rather than general topics

## 12. Future Enhancements

- LoRA/QLoRA support for more efficient fine-tuning
- Multi-language fine-tuning capabilities
- Cross-speaker style transfer
- Voice mixing and interpolation
- Fine-tuning directly from remote audio sources