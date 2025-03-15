# Training Pipeline and Data Processing

## Dataset Collection and Preprocessing

Training CSM required an enormous amount of conversational speech data. Sesame assembled a custom dataset of approximately **1 million hours of audio** (predominantly English) drawn from public sources. The source data likely included:

- Audiobook collections
- Podcast archives
- YouTube videos with dialogues
- Call center recordings made public for research
- Research datasets of conversational speech

The key was to get diverse, **multi-speaker conversational** audio. Each audio source was processed with:

1. **Automated speech recognition (ASR)** to get transcripts
2. **Diarization** to identify speaker turns
3. **Segmentation** into manageable chunks (to fit the 2048-token window)
4. **Quality filtering** to remove segments with bad ASR transcripts or extremely noisy audio

Diarization was particularly important because the model needs to know _who_ is speaking when, so that it can learn to assign consistent voices to each speaker ID and follow turn-taking structure.

The resulting training examples each contain a sequence of text and audio tokens. A single example might look like:
- _[Speaker0: "Hello, how are you?" + audio of that]_
- _[Speaker1: "I'm doing well, thanks." + audio]_
- _[Speaker0: "Great to hear." + audio]_

By structuring the data this way, the model learns to predict each speaker's next audio given the ongoing conversation. The massive scale – orders of magnitude larger than typical TTS datasets – empowers the large backbone model to generalize and produce human-like dialogue.

## Training Framework and Hyperparameters

The team trained CSM end-to-end using the paired data, treating it as a **language modeling task** over an extended token vocabulary (which includes both text subword tokens and audio code tokens).

At training time the model is fed a long token sequence (2048 tokens) and it tries to predict the next token at each position. These tokens could be text or audio, so the model is learning to handle both:

- Whenever a text token is next in sequence (e.g., the next words of a transcript), the model's job is effectively like that of a language model predicting text.
- Whenever an audio token is next (e.g., the next code0 or code1 of a response), the model's job is to output the correct code index.

The loss function is the sum of the cross-entropy losses for predicting all these next tokens. Because audio tokens far outnumber text tokens in the sequence (continuous speech has many frames, whereas text might be shorter), a large portion of the training optimizes predicting the audio tokens correctly.

Training was done for **5 epochs** over the 1M-hour dataset. One epoch means the model saw each hour of audio once (though likely shuffled in short segments). Five epochs means effectively 5 million hours worth of audio-equivalent training.

To handle this scale, the training was almost certainly distributed across many GPUs (or TPUs):

- The backbone being 8B parameters and the context 2048 tokens means memory per example is huge
- Distributed data parallelism with gradient checkpointing and possibly model parallelism would be needed
- Mixed precision (FP16 or bfloat16) would be used to speed up computation and fit in memory
- AdamW optimizer with a learning rate schedule was likely used
- The batch size (in tokens) would be chosen to maximize throughput without overflow

Importantly, the open-source release is the 1B parameter variant, suggesting the full 8B model was more of an internal or demo model.

## Compute Amortization for Efficiency

One of the most novel training techniques used in CSM is what Sesame calls a _compute amortization scheme_, which addresses the heavy memory and compute load introduced by the multi-codebook audio decoder.

The problem is that if the model had to calculate loss for every single audio token (including all N codebooks for every frame in a 2-minute sequence), the amount of computation would be enormous – the decoder would backpropagate through maybe tens of thousands of tokens per sample.

To make this feasible, Sesame **sparsified the training of the audio decoder**:

1. In each training batch, for each training example, they randomly select only a small fraction of audio frames for the decoder to learn on
2. Specifically, they used _1/16 of the frames_ (about 6.25%) and computed the decoder's loss on those frames, ignoring the others
3. The backbone, however, still learns on every frame (predicting code0 for each frame in the sequence)

This approach dramatically reduces gradient computations for the decoder. Essentially, they "amortize" the decoder training over multiple batches: any given batch updates the decoder on a subset of frames, but over many batches the decoder sees all frames.

Importantly, they reported that this did **not hurt the decoder's performance** – there was no noticeable difference in decoder loss or output quality when using this 1/16th frame sampling. Likely the redundancy in speech (lots of frames are similar) and the strong conditioning from the backbone makes it possible for the decoder to learn effectively even with sparse feedback.

This trick is a big reason they could scale to an 8B backbone with an RVQ decoder. It mitigates the memory bottleneck: since the decoder is smaller, and now it's also only active for a few frames per sequence on the backward pass, the peak GPU memory usage and computation time per iteration drop significantly.

## Fine-Tuning for Contextual Memory and Quality

After the main training, the base CSM model can generate speech given text and context. Sesame then applied fine-tuning for specific improvements and uses. They mention that a _fine-tuned variant_ of CSM was used to power their interactive voice demo.

Possible fine-tuning steps include:

- **Conversation fine-tuning:** Reinforcing dialogue coherence by fine-tuning on actual conversations (as opposed to random segments)
- **Style/Persona fine-tuning:** Optimizing for friendliness and expressivity, possibly using a subset of data with appropriate tone
- **Safety and appropriateness:** Training the model not to use offensive tones or mimic sensitive voices
- **Multi-speaker consistency:** Fine-tuning separate _voice profiles_ by conditioning on a single speaker's data
- **Emotion recognition integration:** Integrating the emotion classifier with the backbone to better utilize the classifier's output

The open-sourced model **does not** come with specific voices baked in – it's a _base model_ capable of many voices. This enables greater flexibility for different use cases.

## Comparison to Moshi's Training Approach

In comparing to **Moshi's training approach**, Moshi likely went through a similar two-step process: first training on raw audio/text, then multimodal instruction tuning to make it behave as a conversational agent.

The big difference is that Moshi used a pre-trained language model (the 7B "Helium") as a starting point for text understanding, whereas CSM did not leverage a pre-trained text model. Moshi's team reported going from scratch to a working model in 6 months with 8 researchers, suggesting they used transfer learning and heavy existing components, whereas Sesame spent more time curating data and training from ground up.

Additionally, Moshi had to coordinate ASR and TTS together (since it's full-duplex), possibly training in stages (first ASR, first TTS, then combined). CSM's training pipeline is simpler in that it is one unified sequence task (albeit huge).

In terms of scale, both likely used several hundred GPUs – the details aren't public, but training multi-billion parameter models on ~million-hour data is at the cutting edge of what's being done in industry research.

To summarize, training CSM from scratch involved:
1. Building a massive dual-modal dataset
2. Designing a special architecture and tokenizer
3. Using a clever training trick to handle the load
4. Fine-tuning for performance and style

Each of these steps required careful engineering but they are documented at a high level in Sesame's blog and code release, providing a roadmap for replication.

---

## Navigation

* [Back to Index](index.md)
* Previous: [Technical Components](components.md)
* Next: [Inference and Processing](inference.md)