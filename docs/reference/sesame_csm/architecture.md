# Machine Learning Model Architecture and Layout

## Overall Architecture

CSM uses a _dual-Transformer_ architecture that operates on interleaved text and speech token inputs. It consists of two autoregressive Transformer models working in tandem:

1. A large **Backbone Transformer**
2. A smaller **Audio Decoder Transformer**

The backbone is a multimodal model that processes both textual tokens (from the conversation transcript) and audio tokens (from previous speech, represented as discrete codes) in a single sequence. Its role is to model the high-level content and context – effectively understanding "what to say and how to say it" at a coarse level.

The output of the backbone at each time step is not directly speech, but the first level of an audio code (the **zeroth codebook token**) which captures the semantic and prosodic content of the next chunk of speech. The second model, the audio decoder, then takes this predicted code and generates the remaining audio codebook tokens needed to produce the final speech waveform for that time step.

In essence, the backbone produces a _context-aware linguistic+prosodic summary_ of the next bit of speech, and the decoder "fills in" the fine acoustic details to make it sound realistic. This division of labor allows CSM to remain end-to-end (the whole system is trained jointly) while being efficient: the heavy context modeling is done in one place, and the fine waveform reconstruction in another, smaller model.

## Model Visualization

CSM's dual-Transformer architecture at inference:

![CSM Model Architecture](https://i.imgur.com/placeholder.png)

- Text tokens (T) and audio tokens (A) from the conversation history are interleaved and fed into the large Backbone Transformer
- The Backbone Transformer predicts the next semantic audio token (codebook 0)
- The smaller Audio Decoder then generates the remaining acoustic tokens (codebooks 1 through N−1) needed for that frame
- The full set of audio tokens (A) for that frame is decoded to speech, and also fed back into the backbone for predicting subsequent tokens, until an end-of-utterance token is produced

## Dual-Transformer Framework

The backbone and decoder operate in a loop to produce conversational speech. During generation, the model maintains a sequence of tokens representing the dialogue so far: this includes text tokens for things that were said, plus audio tokens for how they were said.

At a given step, backbone takes in the recent text prompt (the new sentence it needs to speak) along with prior context tokens, and autoregressively emits the next audio code (semantic code) for the response. The decoder immediately takes that code and generates the detailed acoustic codes (e.g. timbre, fine intonation) for the corresponding audio frame.

Once the decoder outputs are obtained, they are assembled into a complete audio frame which can be converted to a short waveform segment. Crucially, that audio token is fed back into the backbone's input sequence for the next iteration. This means the backbone always has access to the _up-to-date audio history_ of the conversation, including the very speech it has generated so far.

The process repeats token by token until the model produces a special end-of-utterance token, indicating the response is complete. Because the decoder is much smaller and faster, this two-stage autoregressive generation is still low-latency – the backbone does the complex part only once per frame, and the decoder's work is quick to compute.

The result is an integrated but efficient pipeline: by splitting at the semantic code (codebook 0), the system avoids having a single massive model generate _all_ waveform details sequentially (which would be slow). Instead, the backbone focuses on long-range coherence and context, while the decoder focuses on high-fidelity speech synthesis.

## Encoder-Decoder Analogy

Although CSM's architecture is autoregressive rather than a traditional encoder-decoder, you can think of the backbone+decoder division as analogous to an encoder-decoder system. The backbone "encodes" the text and prior audio context into a semantic audio representation, and the decoder "decodes" that into actual speech tokens.

However, both are generative Transformers running in sequence each time step, rather than one encoding a fixed input and the other generating an output all at once. This design proved advantageous for modeling conversational speech: the backbone Transformer can attend over a long history of both modalities (text _and_ audio) to decide the manner of speaking, while the decoder Transformer, conditioned on that decision, ensures the speech comes out sounding natural and in the target voice.

All components are trained together, so the backbone learns to produce semantic codes that are easy for the decoder to reconstruct into audio.

## Comparison to Moshi (Kyutai Labs)

The Moshi system by Kyutai Labs also uses a dual-transformer concept, but with a different layout and objectives. Moshi is built as a _full-duplex spoken dialogue model_, meaning it can listen and speak simultaneously in real time. It employs a **global-local transformer architecture**, sometimes referred to as a Temporal vs. Depth transformer structure.

In Moshi, a large 7B-param model (codenamed "Helium") handles temporal sequence modeling (the global structure of the conversation over time), and a separate small "Depth" transformer handles the inter-codebook dependencies _within each audio frame_. This is conceptually similar to CSM's backbone vs. decoder split, but Moshi's design is even more specialized: the Depth transformer in Moshi is responsible for producing multiple acoustic codebooks in parallel for each time step (reducing the need to generate them one-by-one), while the Temporal transformer looks after the content of what's being said and when.

Additionally, Moshi integrates speech recognition and text generation into its architecture – it actually predicts text tokens (transcription) for the AI's own speech as it generates it, as a form of _internal language modeling_ to guide the speech generation. CSM, by contrast, is only a TTS model (multimodal in that it takes audio context, but it doesn't produce transcripts or take audio input on the fly).

In terms of latency and real-time performance, Moshi's architecture is highly optimized for streaming. By using the Mimi codec at 12.5 Hz and parallel codebook generation, it achieves as low as ~200 ms response latency on appropriate GPU hardware, even allowing interruptions mid-speech. CSM, while efficient for a non-streaming model, typically achieves ~380 ms latency on a high-end GPU for an utterance (it generates slightly slower and in a turn-based fashion).

Both architectures share the principle of splitting high-level content modeling from low-level acoustic modeling, and both leverage the same audio codec (Mimi) – but Moshi blurs the line between "speech model" and "language model," whereas CSM cleanly separates them (CSM strictly focuses on speech generation, delegating language understanding to other systems).

---

## Navigation

* [Back to Index](index.md)
* Previous: [Introduction](introduction.md)
* Next: [Technical Components](components.md)