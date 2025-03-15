# Training Procedures

Moshi's training pipeline was conducted in **multiple phases**, each with different data and objectives. The model starts from Helium's pretrained weights for the Temporal Transformer, then is gradually taught to handle audio tokens and multi-speaker dialogue, and finally fine-tuned for conversational instruction-following. In parallel, the Mimi codec is trained separately on audio data before being fixed for Moshi.

## Datasets and Preprocessing

### Text Pretraining Data

Helium was trained on 2.1 trillion tokens of English text. High-quality sources included:
- Wikipedia
- StackExchange Q&A
- Scientific publications
- Filtered CommonCrawl web data

The filtering pipeline:
1. Removed duplicative content (using hash fingerprints and deduplication)
2. Filtered non-English text
3. Applied quality scoring using a fastText-based classifier
4. Prioritized content from high-quality domains (STEM, humanities)

The resulting text dataset required distributed training over many GPUs for several weeks, using unsupervised next-token prediction with cross-entropy loss.

### Unsupervised Audio Pretraining Data

For teaching Moshi to model speech audio, Kyutai assembled an **enormous audio dataset (~7 million hours)** of "readily available audio content." This dataset is orders of magnitude larger than typical speech corpora and likely includes:
- Audiobooks
- Podcasts
- Web videos
- Conversational speech datasets

The majority is English content (matching Helium's language training). They call it "unsupervised" because these audio clips are not annotated with transcriptions or speaker labels for the most part.

Audio preprocessing steps included:
- Resampling to 24 kHz (the codec's rate)
- Normalizing volume levels
- Segmenting long recordings into shorter chunks (using random 12-second windows during training)

During this phase, Moshi is exposed to audio data in a _single-stream_ fashion (only one speaker's audio at a time) to learn audio generation and understanding.

### Simulated Multi-Stream Data

After initial audio training, Moshi undergoes a **post-training phase with "simulated multi-stream" audio based on diarization**. This means they created pseudo-conversations from raw audio by:
- Taking single-speaker recordings and mixing or pairing them to form two-channel inputs
- Applying speaker diarization algorithms on multi-speaker recordings to get separate speaker segments
- Treating these as parallel streams for training

The goal in this phase is to introduce Moshi to **two-stream audio** scenarios _without_ relying on transcripts. The model learns to handle two concurrent token streams and the temporal relationships between them (e.g., one stream pausing while the other speaks).

This phase used approximately 100k training steps with 8-hour audio batches.

### Supervised Conversation Data (Fisher Dataset)

To teach Moshi actual conversational dynamics with real transcript grounding, they fine-tuned on the **Fisher English corpus** - approximately 2,000 hours of telephone conversations with transcriptions. Fisher provides:
- Ground-truth two-channel audio (one channel per speaker) aligned with text
- Real examples of overlapping dialogue, turn-taking, and natural conversational flow

Kyutai used this dataset to:
1. Fine-tune Moshi's ability to handle real conversational dynamics (10k training steps)
2. Teach the model to cope with realistic timing, interruptions, and silence in human dialogue
3. Introduce **Inner Monologue text tokens** for Moshi's side using aligned transcripts

Transcript alignment was done using Whisper (medium model) to get accurate word timing. Interestingly, they did **not train Moshi directly on Fisher text** for the user side - the user's words were provided only as audio, not as text. This mimics Moshi's deployment setting where it doesn't receive transcripts of user speech.

### Synthetic Conversation Data (Instruction Fine-tuning)

The final training stage was an **instructional fine-tune** on a large **synthetic dialogue dataset** created by Kyutai. This process involved:

1. Fine-tuning Helium (as a text model) on existing dialogue datasets like OpenChat and OpenHermes
2. Using this model to _generate thousands of new dialogue scripts_ between a user and Moshi
3. Converting these text dialogues into audio with two distinct voices using multi-stream TTS
4. Varying the "user" voice's accent and noise conditions while keeping Moshi's voice consistent

This approach created approximately **20,000 hours of synthetic conversational audio** paired with transcripts - far more than any available real dataset.

The synthetic data included:
- Instruction-following scenarios
- Question-answering exchanges
- Safety conversation scripts (where users ask inappropriate questions and Moshi refuses)

This fine-tuning phase (roughly 30k steps with 2.7-hour audio batches) taught Moshi to act as a helpful conversational agent that can follow instructions, answer questions, and maintain consistency throughout dialogues.

## Summary of Training Phases

According to the documentation, Moshi's training had four distinct phases:

1. **Pre-training (Audio-LM phase)**
   - Data: Unlabeled audio (7M hours)
   - Setup: Temporal Transformer initialized from Helium's weights, Depth Transformer from scratch
   - Process: Predict next audio tokens (for one stream) as a language model
   - Note: Half of the training time included text-only batches to preserve Helium's language skills
   - Duration: ~1M steps, large batch (16 hours audio per batch)
   - Technical detail: Used acoustic delay of 2 frames (semantic token appears 2 frames before acoustic tokens)

2. **Post-training (Dual audio streams, unsupervised)**
   - Data: Simulated two-speaker audio (from diarization/mixing)
   - Process: Learn to handle two parallel audio token streams
   - Duration: ~100k steps (8h audio batch)
   - Technical details: Reduced acoustic delay to 1, text-only batches 10% of the time

3. **Supervised Fine-tuning (Conversational audio)**
   - Data: Real dialogue audio with transcripts (Fisher corpus)
   - Process: Learn from aligned transcripts and real conversations
   - Duration: 10k steps, 40 min audio per batch
   - Key addition: Inner Monologue text tokens aligned with Moshi's speech

4. **Instruction Fine-tuning**
   - Data: Synthetic multi-turn conversations (20k hours)
   - Focus: Instruction following, knowledge Q&A, and safety behavior
   - Duration: ~30k steps, 2.7h batch
   - Result: A model that can carry full-duplex conversation while following instructions

## Loss Functions and Optimization

Moshi (Temporal+Depth) is trained as a **causal language model over a composite vocabulary** (text tokens + audio codec tokens). The primary loss is the cross-entropy of predicting the next token in each stream.

At each time step, the model predicts the next token(s) in the sequence of combined streams. For example, the sequence might be:
```
[User_audio_token(s)_t=1, Moshi_text_token_t=1, Moshi_audio_token(s)_t=1, User_audio_token(s)_t=2, ...]
```

The model is trained to predict each token given all previous tokens. Key optimization strategies included:

1. **Weighted Token Loss**
   - Semantic tokens: weight α=100
   - Acoustic tokens: weight of 1
   - This prioritizes linguistic content over minor acoustic details

2. **Acoustic Token Delay**
   - Semantic token for a time frame is predicted slightly ahead (τ=1) of the acoustic tokens
   - Helps the model plan phonemes first then acoustics
   - Effectively implements the inner monologue approach

3. **Balancing Multi-Modal Training**
   - During unsupervised pre-training, maintained text-only language modeling alongside speech modeling
   - Used separate optimizer states to prevent "catastrophic forgetting" of Helium's knowledge
   - Later phases still included text-only batches (10% in one phase) to preserve textual knowledge

4. **Optimization Parameters**
   - Learning rates: Smaller for Helium's Transformer (~3e-5), higher for new tasks (2e-4)
   - Used AdamW optimizer with cosine decay schedule
   - Likely implemented mixed precision (bfloat16) and sharded data-parallel training

This sophisticated training regimen allowed Moshi to develop both strong language understanding and speech generation capabilities in a single unified model.

---

## Navigation

* [Back to Index](index.md)
* Previous: [Architecture and Components](architecture.md)
* Next: [Inference and Processing](inference.md)