# Moshi Architecture and Components

## Introduction

Moshi is a recently introduced speech-to-speech foundation model by Kyutai Labs that enables real-time, full-duplex spoken dialogue. Unlike traditional voice assistants that pipeline ASR (speech recognition), NLP, and TTS (speech synthesis), Moshi treats conversation as a single end-to-end process: it directly **generates speech from speech**, using text only as an internal intermediate representation.

This unified approach preserves non-verbal cues (like emotion and intonation) and allows truly interactive conversations with overlapping speech (no fixed turn-taking). Moshi's design is powered by three primary components:

1. A **text language model backbone (Helium)** for language understanding/generation
2. A **neural audio codec (Mimi)** for compressing/decompressing speech audio into discrete tokens
3. A **multi-stream Transformer architecture** (Temporal and Depth Transformers) to model the simultaneous audio streams of user and system in a hierarchical fashion

The following sections provide an in-depth technical breakdown of Moshi's architecture, training regime, real-time inference process, model topology, implementation details, and safety considerations – all the information an expert ML engineer would need to reimplement Moshi from scratch.

## Overview

High-level architecture of Moshi's full-duplex dialogue model:

- The Helium Temporal Transformer (7B) autoregressively models time steps (at 12.5 Hz)
- A smaller Depth Transformer generates the multiple audio codec tokens per time step (semantic & acoustic tokens)
- The user's incoming audio is encoded by Mimi into discrete tokens fed into the model as one stream
- Moshi's own outgoing audio tokens are generated simultaneously on another stream
- An "Inner Monologue" text token may be produced as a prefix to Moshi's audio tokens, improving linguistic quality
- Moshi's audio output tokens are decoded by Mimi back to speech audio

This architecture enables processing of two audio streams concurrently with minimal latency.

## Helium: Text Language Model Backbone (7B LLM)

### Architecture

Helium is a 7-billion-parameter autoregressive language model that forms the core "brain" of Moshi. It uses a Transformer decoder architecture similar to recent large language models (comparable to LLaMA-7B) with several modern tweaks for efficiency and performance.

Helium adopts:
- **RMSNorm** normalization layers (instead of LayerNorm) at the input of attention/FFN blocks and output
- **Rotary Positional Embeddings (RoPE)** for 4,096 token context
- **Gated Linear Unit (GLU)** feed-forward layers with SiLU activation

These choices follow best practices from contemporary LLM research – e.g., RMSNorm and RoPE help stable training at long context lengths, and GLU (a gated feed-forward such as SwiGLU) improves parameter efficiency and learning capacity.

The model dimension is 4096 with 32 Transformer layers and 32 attention heads, and feed-forward hidden size ~11k (11264), consistent with a 7B parameter scale. A SentencePiece unigram tokenizer with a vocabulary of 32,000 subword tokens (primarily English) is used for text; it includes byte fallback to avoid unknown tokens and splits numbers into individual digits to preserve information.

### Training Data & Objectives

Helium was pretrained on an extremely large text corpus of **2.1 trillion tokens** of English text. Kyutai curated high-quality sources like Wikipedia, StackExchange, and a large collection of scientific articles, then augmented with filtered CommonCrawl web data to reach the required scale.

The data pipeline involved:
- Aggressive deduplication (using hash-based filtering and bloom filters)
- Language ID filtering for English
- A **quality filtering** classifier that scored pages (favoring those related to high-quality domains like STEM or humanities)

This ensured Helium's text training data is rich and diverse while minimizing low-quality or toxic content. Training Helium to convergence required on the order of 500k optimization steps with very large batches (4.2M tokens per batch) – achieved via distributed training on GPU clusters. Optimization used the **AdamW** optimizer with a fixed learning rate and cosine decay schedule.

### Role in Moshi

In the Moshi architecture, Helium serves as the **Temporal Transformer** – it processes the dialogue history (in text and audio-token form) over time and produces latent context embeddings that will be used to generate new outputs.

Essentially, Helium is responsible for understanding the conversational context and deciding _what_ to say next (at the semantic/text level), leveraging its LLM capabilities. However, rather than directly outputting text, Moshi uses Helium's next-step embedding to condition a secondary model that will produce audio tokens (speech) for that step.

This allows the model to **speak in audio** while still thinking in terms of language. Helium's large capacity enables Moshi to carry on coherent, contentful dialogues – it provides the language backbone that ensures responses are contextually relevant and logically sound. Helium was even fine-tuned on dialogue-style data (e.g., OpenHermes synthetic dialogue and real transcripts) to better handle interactive conversations, which helps Moshi produce realistic conversational behavior.

## Mimi: Streaming Neural Audio Codec

### Purpose

Mimi is Moshi's **neural audio codec**, responsible for converting raw speech waveforms into discrete token sequences (and back). It compresses audio at 24 kHz into a low-bitrate sequence of codec tokens that the language model can handle. Without Mimi, Moshi would have to deal with raw audio signals directly, which is intractable for an LLM – Mimi provides a learned discrete representation of audio that is far more compact while preserving speech content and quality.

### Codec Structure

Mimi follows the general design of neural codecs like **SoundStream/EnCodec**:
- An **encoder** (which compresses audio to a latent space)
- A quantization bottleneck
- A **decoder** (which reconstructs audio from quantized latents)

The encoder is a convolutional neural network that progressively downsamples the input waveform; specifically, Mimi's encoder uses a stack of residual conv blocks with strides 4×5×6×8×2 (total downsampling factor 1920) to turn 24 kHz audio into a 512-dimensional latent vector at **12.5 Hz frame rate** (i.e., one 512-d vector every 80 ms).

All convolutions are causal (no future context) with dilations, ensuring the encoder can run in a streaming fashion (producing output as audio comes in). The decoder mirrors this (with transpose convolutions) to upsample 12.5 Hz latent back to 24 kHz audio.

At the bottleneck, Mimi uses **Residual Vector Quantization (RVQ)** to discretize the 512-d latent. Importantly, Mimi's quantizer is _split_ into two parts:
- One **semantic codebook** 
- Several **acoustic codebooks**

In total Mimi uses **Q = 8 quantizers** (codebooks) per frame in the paper's design. The first quantizer produces a "semantic token" capturing high-level linguistic content of that 80ms audio frame, while the remaining 7 quantizers produce "acoustic tokens" that capture the detailed voice timbre, prosody, and other low-level audio features.

Each codebook has 2048 possible entries (11 bits). Thus for each 80ms frame, Mimi yields 8 discrete tokens (1 semantic + 7 acoustic), and the overall bitrate is **12.5 Hz × 8 tokens × 11 bits ≈ 1.1 kbps** per audio stream. This is a **very low bitrate** compression – much smaller than traditional codecs – making it feasible for Moshi to model speech with manageable sequence lengths.

## Multi-Stream Modeling: Temporal & Depth Transformers for Dual Audio Streams

A core challenge in Moshi is modeling **two simultaneous streams** of audio tokens (user and system) along with any text tokens, without flattening everything into an exorbitantly long sequence. To handle this, Moshi introduces a **hierarchical Transformer architecture** called the **RQ-Transformer**.

It consists of:
1. The large **Temporal Transformer** (Helium) which operates along the time axis
2. A smaller **Depth Transformer** which operates along the "codebook depth" axis at each time step

This effectively factorizes the generation: Helium predicts the progression of conversation over time (in steps of 80ms increments), and at each time step, the Depth Transformer produces all the required codec tokens for that time frame.

By doing so, Moshi's architecture avoids the need to handle every audio token in a single sequence. Instead of one giant autoregressive sequence of length _T × K_ (time steps × codebook tokens) for a conversation, it can handle _T_ steps with the big model and up to _K_ steps with the smaller model, dramatically reducing computational cost.

---

## Navigation

* [Back to Index](index.md)
* Next: [Training Procedures](training.md)