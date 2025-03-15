# Comparison to Moshi by Kyutai Labs

CSM and Moshi are both cutting-edge conversational speech generation models, but they were developed by different teams with somewhat different goals, leading to differences in architecture and capabilities. Here we'll compare them in several aspects:

## Architectural Design

Both CSM and Moshi utilize dual transformer architectures, but the way they partition tasks is different:

| CSM | Moshi |
|-----|-------|
| Splits by _function_ (context modeling vs. acoustic rendering) | Splits by _time scale_ (temporal vs. depth) |
| Large backbone vs. small decoder | Temporal (global) transformer vs. Depth (local) transformer |

CSM's backbone is analogous to Moshi's Temporal transformer (both model long sequences of a "primary" token stream: code0 for CSM, combined audio-text for Moshi), and CSM's decoder is analogous to Moshi's Depth transformer (both model dependencies among codebooks per frame).

However, Moshi's architecture is more complex because it is a _speech-text_ model: it doesn't just predict audio tokens, it also predicts text tokens for its own output concurrently. Moshi essentially learned to _speak and listen in the same model_, integrating an ASR component and a TTS component.

Moshi uses a form of next-token prediction where the next token could be from either the user's audio stream or the AI's own audio stream or the AI's own transcript. CSM, by contrast, is only a TTS model (multimodal in that it takes audio context, but it doesn't produce transcripts or take audio input on the fly).

This means Moshi's training and architecture had to account for two simultaneous audio streams (full-duplex) and a text stream, making it a _three-stream model (user audio, AI audio, AI text)_, whereas CSM is a _two-stream model (past audio, upcoming audio, plus text as annotations)_.

## Full-duplex vs. Turn-based

A key practical difference is that Moshi is designed for **full-duplex dialogue** – it can handle speaking and listening at the same time without explicit turn-taking.

In Moshi's demo, the AI can interject "mm-hm" while the user is still speaking or start formulating an answer before the user fully stops. This is achieved by Moshi's architecture continuously processing input audio and generating output audio tokens with minimal delay (only 80 ms frame delay). It doesn't wait for an "end of utterance"; it treats the conversation like two audio streams flowing concurrently.

CSM, on the other hand, currently works in a **turn-based** manner. It assumes one speaker talks, then stops, then the other responds. It needs the text (and ideally the audio) of a user utterance as input context, then it produces the response. If the user were to interrupt while CSM is speaking, CSM has no mechanism to incorporate that mid-response – it would likely continue until completion (unless an external system stops it).

Therefore, Moshi is more suitable for _natural, overlapping conversation_, whereas CSM is more aligned with the typical voice assistant interaction pattern (speak → wait → respond).

## Integration of Language Modeling

| Moshi | CSM |
|-------|-----|
| **Speech-dialogue foundation model** | **Text-to-speech model** |
| Decides _what_ to say and _how_ to say it | Decides only _how_ to say provided text |
| Fine-tuned on dialogue data | Only trained on speech generation |
| Unified model for understanding and generation | Requires external LLM for understanding |

Moshi is, at its heart, a **speech-dialogue foundation model** – it not only generates speech, but also decides _what_ to say (like ChatGPT with a voice). In fact, the first component of Moshi ("Helium") is a fine-tuned LLM that was trained to generate conversational content, and it outputs both text and some representation that guides the speech output.

CSM deliberately does _not_ include a language generation component – it requires an external text input for the content of speech. Sesame's design philosophy was to separate concerns: use the best text-based LLM for language understanding and use CSM for rendering speech from that text.

This has pros and cons:
- With CSM+LLM, you can upgrade the text brain independently
- With Moshi, the text and speech are entwined; you have one model that does both

Moshi's integrated approach could yield more coherence between wording and prosody – since it knows exactly what it's saying as it says it, it might place emphasis more appropriately. For instance, Moshi's internal text prediction allows it to ensure that its spoken output matches the intended text exactly (because it literally generates the text as a token sequence too, reducing risk of say, mispronouncing a word or skipping a word).

## Model Size and Efficiency

| Aspect | Moshi | CSM |
|--------|-------|-----|
| Main model size | ~7B parameters | ~8B parameters |
| Secondary model | Small Depth transformer | ~300M decoder |
| ASR component | Built-in | External |
| End-to-end latency | ~200 ms | ~380 ms |

Moshi's main Transformer is about 7 billion parameters (plus whatever the Depth transformer adds, which is relatively small, maybe a few hundred million). CSM's largest is 8B + 0.3B decoder. So in raw size they're comparable.

However, Moshi's architecture might demand more compute at inference because it's essentially performing ASR and TTS in one model – although the ASR part is mostly just ingesting audio tokens, which, thanks to Mimi, is also a 12.5 Hz stream, so it's not that heavy.

Moshi's advantage is it does fewer autoregressive steps because of parallel codebook prediction and because it doesn't wait for an end-of-utterance. In terms of **latency**, Moshi achieves a remarkable ~200 ms end-to-end latency (including the 80 ms frame delay) in a streaming scenario. CSM, while very fast for a TTS system, is used in a call-response scenario where you likely have an additional ASR delay (maybe 200 ms) plus the CSM generation (~380 ms) – so the total user-perceived latency might be ~0.5-1.0 s if you include ASR.

In a fair measure just of TTS, CSM might be ~0.3-0.4 s vs Moshi's ~0.2 s, but Moshi doesn't have the ASR step because it's built-in. For practical purposes, both are low enough to feel interactive, but Moshi edges out if you need immediate back-and-forth.

## Voice Naturalness and Quality

Both models produce highly natural voices, nearly human-like in many cases. Subjective evaluations (like CMOS scores) for CSM showed that without context, people could hardly tell it apart from real speech. Moshi's public demos also astonished listeners with how human and spontaneous it sounded (to the point of some being unsettled).

CSM places emphasis on using context to produce _appropriate_ prosody – e.g., its evaluations showed that when given conversational context, human evaluators preferred real human speech slightly over CSM, meaning CSM is extremely close but perhaps not perfect yet in contextual expressiveness. Moshi, with its ability to "um" and interrupt, demonstrates a very high degree of conversational realism.

One difference is that CSM's output is deterministically tied to the input text. If the input text is poorly phrased, CSM will still read it in a fluent way, but it won't _rewrite_ or improvise. Moshi can actually choose wording on the fly, leading to very fluid dialogues. This can make Moshi seem more _alive_ as a personality, whereas CSM is more like a highly advanced _reader/actor_.

In terms of **speaker similarity** or cloning, both can impersonate voices given a sample. Moshi presumably can do zero-shot voice based on a short prompt (similar to how Vall-E does), and CSM can do it if provided a snippet in context. The underlying tokenization being the same (Mimi) means if both see the same voice token sequence, they should both be able to render that voice.

## Context Handling

CSM can handle 2048 tokens of context (roughly 2 minutes, could be many turns). Moshi, being streaming, effectively can handle indefinite context – it doesn't have a set context length in the same sense, because it processes input continuously. However, Moshi's internal architecture might still have some window for attention for its temporal model.

Moshi also uses the technique of predicting its own transcript, which provides a textual summary of the conversation as part of the context (the "inner monologue"). This likely helps it maintain context in long dialogues by relying on text memory (which is compact) rather than pure audio memory.

CSM relies on having both text and audio tokens fed in for context – text gives it the words, audio gives it tone. Both approaches have merit.

## Tokenization and Audio Compression

Both CSM and Moshi use the **Mimi tokenizer**. This means they share the same fundamental audio representation. There might be minor differences in implementations, but both output one semantic token and multiple acoustic tokens per frame.

Where they differ is how those tokens are generated:
- CSM's decoder generates tokens sequentially
- Moshi's depth transformer might generate all tokens at once for a frame

Moshi's approach might be slightly faster in generation within each frame, but both are so fast at that level it's negligible compared to the big transformer passes.

## Model Efficiency and Footprint

Moshi, by being end-to-end, requires one model to do everything. This means if one wanted to deploy Moshi on a device, that device must handle the 7B model. CSM could be split: ASR on device (maybe a small Whisper model), LLM in the cloud, CSM TTS in the cloud, etc.

For strictly TTS usage, CSM's 1B model is already out and much smaller to run. Moshi has not released weights yet (as far as known); only the demo is available. So from an open-source standpoint, CSM currently is more accessible.

Efficiency also includes training efficiency: CSM's training used the amortization trick to reduce memory, whereas Moshi's training complexity came from multitask learning. It's not obvious which was more efficient – both likely pushed GPUs to their limits.

## Differences in Naturalness and Context Handling

In terms of _natural interactive behaviors_, Moshi can do things like incremental speech, backchanneling ("uh-huh", "right…") and quick turn-taking. CSM can produce very natural _speaking style_ but doesn't inherently decide to inject backchannels unless the text explicitly says "uh-huh" (since it doesn't generate discourse behaviors by itself).

Another subtle point: Moshi's integrated text+speech modeling might help with _pronunciation_. If Moshi encounters a word it's not sure how to say, it might derive it from spelling via its language model knowledge. CSM relies on the training data and possibly the semantic token to get pronunciation right.

Both claim to handle tricky cases (CSM tested homograph disambiguation using context, which is typically a language understanding problem – CSM solved it by using textual context to choose correct pronunciation).

## Summary of Approach Differences

In short, **CSM = LLM architecture + audio tokens; Moshi = joint LLM + speech architecture**. CSM chooses modularity (separate text generation, focus on speech rendering), while Moshi chooses unity (one model to rule them all).

Both achieve impressive efficiency with Mimi codec, but use slightly different tricks to overcome the multi-codebook challenge (CSM with a sequential decoder but skipping frames in training, Moshi with a parallel codebook prediction using a second transformer).

Depending on the application (closed-loop assistant vs. open-ended chatbot), one or the other might be preferable. For replicating CSM, one can focus on the speech part and interface it with existing NLP systems, whereas replicating Moshi requires tackling ASR, NLP, and TTS all together.

---

## Navigation

* [Back to Index](index.md)
* Previous: [Code Implementation](implementation.md)
* Next: [Implementation Considerations](considerations.md)