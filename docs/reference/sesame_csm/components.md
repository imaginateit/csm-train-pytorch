# Technical Components of CSM

## Backbone Transformer (LLaMA-based)

The backbone of CSM is a Transformer network based on the LLaMA architecture (a GPT-style decoder-only Transformer). It serves as the main _sequence model_ that reads the combined text and audio token sequence.

### Architecture
- In the largest "Medium" configuration, the backbone has around 8 billion parameters
- Smaller variants with 1B or 3B parameters were also trained for research
- Features multiple self-attention layers with many heads and feed-forward networks
- Optimized for autoregressive token prediction

Sesame did **not** initialize this transformer from any pre-trained text model – it was trained from scratch on the conversational speech data. This means the backbone had to learn linguistic patterns, contextual cues, and basic world knowledge directly from the audio transcripts, which is feasible given the massive 1M hour dataset.

### Token Handling
- Uses a SentencePiece tokenizer (or similar) for text, likely the same as LLaMA's tokenizer
- Audio is tokenized by the Mimi codec (described below)
- Special embeddings or token IDs distinguish between text and audio tokens
- Speaker identities are encoded directly into the text token sequence as markers

The backbone's primary job is to output the correct _zeroth codebook token_ for the next audio frame given all the preceding context. In doing so, it must integrate conversational context (what has been said) and paralinguistic context (how it was said) to decide the appropriate prosody, timing, intonation, etc., for the upcoming speech.

### Context Window
With a context window of 2048 tokens (~2 minutes of audio/text), the backbone can retain conversational memory and style over extended dialogues. This long context capability is crucial for maintaining consistency in things like tone or persona over the course of a chat. The backbone uses positional encodings (likely rotary position embeddings as in LLaMA) to manage such a long sequence.

Overall, this transformer acts as the **brain of CSM**, figuring out _what_ audio should sound like before the finer details are filled in by the decoder.

## Contextual Understanding & Long-Form Dialogue Retention

A standout feature of CSM is its ability to utilize **long conversational context**. Each training sample fed to the model was a sequence of up to 2048 tokens, corresponding to roughly 2 minutes of a conversation.

### Context Construction
- Sequences were built by alternating between speakers
- Format: [Speaker0's text + audio tokens][Speaker1's text + audio tokens][Speaker0's text + audio tokens]...
- This allows the backbone to interpret not just the last user utterance, but the entire history

### Contextual Benefits
By training on such comprehensive context, the model retains important information:
- _Dialogue state_ (has a question been answered already?)
- _Emotional tone_ of each speaker (was the user sounding frustrated or pleased?)
- Speaking quirks or pace
- Relevant information from earlier in the conversation

Mechanisms like self-attention allow the model to attend to relevant parts of the history. For example, if the assistant is about to clarify something, the backbone might attend to the earlier part of the conversation where that topic was first raised, ensuring the clarification tone matches the context.

### Practical Advantages
This goes beyond what typical TTS models do – CSM isn't generating speech in isolation, but as a turn in an ongoing exchange. As a result, it can exhibit behaviors like:
- **Dialogue coherence** (sounding hesitant when appropriate, or using a firmer tone if repeating an answer)
- _Long-form synthesis_ (up to 2 minutes straight) without resetting style

In testing, the Medium CSM model showed that including conversational context significantly improved human preference for its responses' appropriateness. Essentially, by remembering up to 2 minutes of dialogue, CSM provides continuity in multi-turn interactions that makes the voice assistant feel more "present" and aware.

From an implementation perspective, handling 2048 token sequences with an 8B model is non-trivial (it's a lot of computation), but Sesame's training strategies and model optimizations made it feasible.

## Residual Vector Quantization (RVQ) Tokenizer – Mimi

Instead of producing raw audio waveforms, CSM outputs **discrete audio tokens** thanks to a technique called Residual Vector Quantization. The team uses _Mimi_, a state-of-the-art neural audio codec developed by Kyutai (and adopted by Sesame for CSM) to convert audio into tokens and back.

### How Mimi Works
- Encodes 24 kHz audio into a sequence of code vectors at 12.5 Hz (one set of codes every 80 ms)
- Each 80 ms frame is represented by a stack of codebook indices:
  - **1 semantic codebook** (codebook 0) for high-level speech content
  - **N−1 acoustic codebooks** (codebooks 1 through N−1) for finer details

The semantic code (zeroth codebook) is trained to represent the content and broad prosody of speech in a compact way. Mimi's training uses a form of **distillation from a self-supervised speech model (WavLM)** so that codebook 0 tokens carry linguistic/semantic information akin to a transcript, but also some prosodic cues.

The remaining acoustic codebooks encode additional information needed for high-fidelity reconstruction:
- Voice timbre of the speaker
- Background noise
- Subtle inflections
- Other fine audio details

### Advantages for CSM
Using Mimi tokens provides several benefits:

1. **Discrete Representation**: Transforms continuous audio generation into a _discrete sequence prediction_ problem suitable for Transformers

2. **Hierarchical Information**: By splitting information across semantic and acoustic tokens, the model can focus on high-level correctness first and detail second

3. **Efficiency**: Mimi compresses speech down to about 1.1 kbps while preserving quality, resulting in fewer tokens to process (about 100 tokens/sec total)

4. **Streaming Capability**: Mimi is _streaming-capable_ and causal by design, aligning with CSM's real-time goals

By operating on this discrete audio representation, CSM can handle aspects like speaker identity and audio style transfer inherently – if the context contains a certain speaker's tokens, the model learns to continue using similar acoustic tokens to maintain that voice.

The semantic token (codebook 0) is especially important for bridging text and speech: since it captures phonetic content, the backbone essentially learns a mapping from text+context to a pseudo-phoneme representation. The subsequent acoustic tokens then ensure the phonemes are rendered in the correct voice and style.

## Audio Decoder and Multi-Codebook Generation

The Audio Decoder in CSM is a smaller Transformer network dedicated to predicting the acoustic codebooks (levels 1 through N−1) given the backbone's output. In the Medium model, the decoder has about 300 million parameters (much smaller than the 8B backbone).

### Decoder Operation
The decoder is conditioned on two inputs:
1. The backbone's hidden representation or output embedding for the current time step
2. Any already-predicted lower-level codes for the current frame

Sesame's design uses a **distinct linear output head for each codebook level** in the decoder. In other words, the decoder has multiple output projections sharing the same Transformer body – one head is responsible for predicting code1, another for code2, etc.

During inference, the decoder predicts codebooks in sequence:
- First code1
- Takes that prediction as input to predict code2
- And so on, until all acoustic codes for that frame are generated

This sequential approach ensures that each higher codebook can depend on the lower codebooks, capturing the residual nature of RVQ.

### Training and Efficiency
Training of the decoder is done jointly with the backbone: the decoder learns to minimize the error in predicting the true acoustic tokens given the true code0 and any previously predicted lower-level codes.

The decoder's **time dimension** is the number of codebooks, not the length of the utterance – it models a short sequence (the stack of codes for one frame) at a time. This is why it can be much smaller; it doesn't need to model long temporal dependencies, which are handled by the backbone.

By keeping the decoder lightweight, CSM achieves low latency generation: once code0 is available for a frame, the decoder can very quickly produce the rest of the codes.

### Advantages of This Approach
This design makes CSM **single-stage** yet efficient – older two-stage TTS would first generate a representation with one model and then invoke a completely separate model to produce the entire utterance. CSM instead interleaves these steps frame by frame, with a tiny "vocoder" step (the decoder) that's intimately guided by the context-aware backbone.

The multi-codebook approach ensures natural speech characteristics like speaker idiosyncrasies, accents, and emotions are preserved in the output. Because the acoustic tokens explicitly encode voice timbre, the decoder can reproduce the unique sound of the speaker's voice or even mimic a style.

In summary, the audio decoder component can be seen as **CSM's neural vocoder**, one that is tightly integrated into the language modeling process.

## Expressive Speech Modeling and Emotion Classification

One of Sesame's goals was to imbue the model with **emotional intelligence** and expressive range, so that it can react to the subtleties of human speech.

### Emotion Recognition Component
According to Sesame's technical disclosures, CSM features a **6-layer emotion classifier** network as part of its pipeline. This is likely a small Transformer or feed-forward model that:
- Takes as input either the audio tokens or the backbone's hidden state
- Predicts an emotion category (e.g., neutral, happy, sad, angry)

By training this classifier on conversation data, the system gains the ability to **classify the emotion** in a speaker's voice, which can then influence generation. The detected emotion of the user's utterance can be fed into the backbone to adjust its speaking style in response.

### Expressive Generation
CSM's expressive capabilities are enhanced by:
1. Its large training corpus containing a wide range of natural emotions and speaking styles
2. The model's ability to capture and reproduce prosodic patterns
3. The emotion classifier guiding appropriate responses

In generated speech, CSM can:
- Insert pauses, emphases, and appropriate intonation patterns
- End questions with rising pitch
- Adapt tone based on emotional context (speaking more softly for sad topics)
- Include human-like elements like laughs, sighs, or pace changes

All this is achieved without explicit rule-based controls, but through the model's internalization of patterns from training data.

The technical components dealing with expressivity include both the model architecture's ability to process prosodic signals (through audio tokens) and auxiliary components like the emotion classifier that help the model understand and appropriately produce emotional cues. This combination allows CSM to produce speech that isn't just intelligible, but **expressively appropriate** to the conversation context.

---

## Navigation

* [Back to Index](index.md)
* Previous: [Model Architecture](architecture.md)
* Next: [Training Pipeline](training.md)