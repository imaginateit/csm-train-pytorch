# Final Considerations for Recreating CSM

Recreating CSM from scratch is a complex but achievable project, given the information and resources now available. Here we outline a step-by-step roadmap and important considerations for an engineer attempting this, as well as the compute requirements and potential challenges.

## Step-by-Step Development Plan

1. **Obtain/Prepare a Large Conversational Speech Dataset**
   
   As a foundation, gather a substantial amount of audio data with transcriptions. Aim for diverse, multi-speaker conversation data. Public datasets can be combined:
   - LibriSpeech/LibriLight for sheer hours of speech
   - Switchboard and Fisher for telephone conversations
   - Common Voice or VoxPopuli for crowd-sourced speech
   - Podcast datasets for natural dialogue
   
   You'll need transcripts and speaker labels. If not readily available, use an ASR like Whisper to transcribe audio and a diarization tool (like Pyannote or SpeakerNet) to label speaker turns.
   
   The goal is to produce a training corpus of token sequences where text and audio tokens alternate by speaker. If replicating at smaller scale, you might start with 1–10k hours instead of 1M, but more data will noticeably improve naturalness and reduce overfitting.
   
   Ensure audio is standardized (e.g., 16 or 24 kHz mono) and segmented into chunks (e.g., 20–120 seconds segments).

2. **Train or Use an Audio Tokenizer (RVQ-VAE)**
   
   CSM relies on the Mimi tokenizer, which you can either reuse or reimplement. The easiest path is to use the **pre-trained Mimi codec** from Hugging Face. It provides an encoder that converts wav audio into a series of discrete codes and a decoder that does the reverse.
   
   If you want to train your own tokenizer, you'd train an autoencoder on audio:
   - Encoder produces a sequence of latent vectors (one per 80 ms frame)
   - Multiple quantization layers (codebooks) progressively quantize the latent with residuals
   - Decoder reconstructs the waveform
   
   Using Mimi, encode all your training audio into token sequences. These tokens will form the "audio portion" of your training samples.

3. **Define the Model Architecture**
   
   Implement the dual-transformer model. You can use an existing Transformer framework (like HuggingFace Transformers) to create a GPT-like model, but you will need to extend it to handle two types of inputs and two transformers:
   
   - Create the **Backbone** model as a decoder-only Transformer with a causal self-attention mask. The backbone's vocabulary is the union of text tokens and audio tokens (code0 specifically).
   - The backbone needs to be able to attend across a long history (so implement positional embeddings for up to 2048 or more positions).
   - Create the **Audio Decoder** model. This can be a smaller Transformer. Have the decoder attend to backbone output and use multiple output heads that directly predict each codebook level.
   - Integrate the two: The backbone will output a probability distribution for code0. In training, you have the true code0 as next token, so backbone loss is cross-entropy on code0 prediction. Then, conditioned on code0, the decoder predicts code1..N-1.
   - Don't forget to prepend special tokens or embeddings for speaker IDs in the input.

4. **Training Setup**
   
   Set up a training loop to feed the data through the model:
   
   - Build batches of padded sequences
   - Use a masking to ensure loss is only computed on real tokens, not padding
   - Use an optimizer like AdamW with a learning rate schedule
   - Implement the **compute amortization**: when computing decoder loss, randomly choose a subset of frames in each sequence to apply decoder loss
   - Start with a smaller model first for faster iteration
   
   As for hyperparameters: 
   - Likely hundreds of thousands of steps
   - Batch size to cover a couple minutes of audio per GPU
   - Use gradient accumulation or micro-batching for large effective batch sizes
   - Set aside development data for validation

5. **Scaling Up**
   
   Once the training pipeline works at smaller scale, scale to larger models gradually:
   
   - Use multiple GPUs with distributed training
   - Utilize mixed precision and gradient checkpointing
   - This stage is the most resource-intensive, potentially requiring weeks on many GPUs

6. **Fine-Tuning and Specialty Training**
   
   After base training, you may fine-tune the model for specific sub-tasks:
   
   - Fine-tune on **emotion-rich data** to enhance expressiveness
   - Fine-tune for **voice consistency** if needed
   - Fine-tune to reduce **verbal filler** if unwanted
   - Integrate with an LLM for appropriate text styling

7. **Inference Pipeline Development**
   
   Develop the generation loop:
   
   - Feed context tokens
   - Use cached attention states for efficiency
   - Generate until an end-of-utterance token is produced
   - Incorporate the Mimi decoder for audio synthesis
   - Consider post-processing like pause insertion
   - Implement streaming mode for reduced latency

8. **Hardware and Optimization**
   
   Optimize for deployment:
   
   - Use quantization (8-bit or 4-bit) to reduce model size
   - Apply high-performance inference libraries
   - Optimize for specific hardware targets
   - Consider multiple model sizes for different use cases

9. **Testing and Evaluation**
   
   Evaluate thoroughly:
   
   - Use objective metrics like WER and MOS
   - Test contextual understanding with homograph disambiguation
   - Evaluate dialogue coherence in multi-turn exchanges
   - Test with different voices and accents

10. **Iterative Improvement**
    
    Identify and address shortcomings:
    
    - Add more data for problematic cases
    - Fine-tune with targeted examples
    - Implement evaluations from Sesame's approach

## Compute and Hardware Requirements

Building CSM from scratch is computationally intensive:

- **Training Scale**: The 8B parameter model on 1M hours requires supercomputer-grade setup
- **GPU Requirements**: Likely 32+ A100 GPUs for several weeks for full scale
- **Realistic Starting Point**: 8 GPUs (24+ GB VRAM each) for a 1B model on 100k hours
- **Storage Needs**: ~100 TB for data and intermediate tokens
- **Inference Hardware**: A single high-end GPU for the final 8B model

## Potential Challenges

1. **Data Quality Issues**
   
   Training on found data includes many potential problems:
   
   - ASR transcript errors
   - Speaker label errors
   - Disfluencies or ASR mistakes being learned
   - Data filtering and preprocessing is crucial

2. **Long-range Dependencies**
   
   Training with 2048-token context presents challenges:
   
   - Some attention heads might ignore far context
   - Context might be confusing with many speakers
   - Managing context relevance requires careful implementation

3. **Balancing Text vs. Audio Learning**
   
   The model must learn two different modalities:
   
   - Audio token prediction might dominate due to higher frequency
   - Text understanding could be sacrificed
   - Loss scaling between modalities may need tuning

4. **Reproducing Expressiveness**
   
   Emotional nuance is challenging:
   
   - A smaller replication might be more monotone
   - More expressive data might be needed
   - Subtle prosody elements require sufficient examples

5. **Deployment Considerations**
   
   When deploying widely:
   
   - Consider model compression techniques
   - Implement ethical safeguards for voice cloning
   - Be mindful of compute efficiency

6. **Multilingual Support**
   
   Supporting languages beyond English:
   
   - CSM was primarily English-trained
   - Multilingual support requires diverse data
   - Language-specific models might be more effective

## Summary

Replicating CSM entails combining expertise in NLP, speech processing, and large-scale distributed training. The publicly released code and model provide a blueprint, and by following the outlined steps, a determined ML engineer or research group can build a system with similar capabilities.

Setting realistic milestones, leveraging existing components like Mimi, and iterating on model quality will be key to success. The final outcome, if successful, would be a powerful conversational speech model that can bring human-like voice interactions to life.

---

## Navigation

* [Back to Index](index.md)
* Previous: [Comparison with Moshi](comparison.md)