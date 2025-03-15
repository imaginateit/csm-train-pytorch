# Inference and Real-Time Processing

One of Moshi's headline achievements is being the **first real-time, full-duplex large language model for speech**. In deployment, Moshi can listen and speak concurrently, with low latency, enabling natural conversation flow much closer to human-human interaction than prior systems. This section explains how the inference process works, how streaming is implemented, and what performance/latency characteristics the system has.

## Full-Duplex Streaming Mechanism

During inference, Moshi operates in a **continuous loop** processing audio in frames (e.g., 40–80ms chunks). It treats the incoming user audio and its own generated audio as parallel streams of tokens that grow over time, and it generates new tokens autoregressively as time advances.

### Typical Inference Flow

A typical real-time session proceeds as follows:

#### Audio Input Encoding
- As the user speaks into the microphone, audio is streamed through the Mimi encoder to produce tokens in real time
- Every 80ms of user speech yields one new semantic token and 7 acoustic tokens (once fully quantized)
- Mimi can output the first token slightly earlier due to causal streaming
- These user tokens are fed incrementally into Moshi's Temporal Transformer as they arrive (on the user stream channel)
- In practice, the system likely buffers a short initial window (e.g., 160ms) to get a couple of frames encoded before Moshi starts responding

#### Autoregressive Generation
- The Moshi model (Temporal+Depth) generates its output tokens step by step
- At each time step, it takes into account all tokens so far (user and Moshi's) and decides whether Moshi should speak (and what)
- If the model "chooses" to speak, it will produce a semantic token (and an inner text token) followed by acoustic tokens for that time
- If the model decides to remain silent (perhaps because the user is talking and it should listen), it might produce a special silence token or simply no Moshi tokens
- This behavior is learned from training where overlapping speech was present
- The inference uses the Depth Transformer to generate all required audio tokens for each time step on the fly
- Importantly, this is all done autoregressively **with streaming context**: Moshi doesn't wait for the user to finish an utterance

#### Output Audio Decoding
- As soon as Moshi generates audio tokens for its own stream, those tokens are sent to the Mimi decoder
- The Mimi decoder synthesizes actual waveform audio that is played back through the speaker
- Because generation happens frame by frame, Moshi can begin speaking _while_ the user is still talking
- The Mimi decoder works in streaming mode, decoding 80ms at a time with only 80ms latency

This pipeline creates a **tight loop**:
```
user audio → Mimi encode → Moshi model → Mimi decode → system audio
```

The architecture's design ensures all components are low-latency:
- The **Temporal Transformer** runs one forward pass per 80ms frame
- On modern hardware, a 7B transformer can compute one step in far less than 80ms
- The **Depth Transformer** is much smaller with negligible computation per step
- Mimi's encode/decode each add ~80ms initial latency but then operate continuously streaming

Once the pipeline is filled, Moshi can respond with as little as a single-frame delay relative to the user.

### Latency

The theoretical minimal latency of the full system is about **160 ms** (as reported by the authors). In practice, they achieved about **200 ms** end-to-end latency on an NVIDIA L4 GPU. This latency budget includes:
- Input audio buffering (~80ms frame)
- Model computation (~80ms or less)
- Another frame of lookahead or safety margin

For comparison, conventional voice assistants often have 2–3 seconds latency (due to waiting for end-of-speech, performing ASR+NLP, then TTS). Moshi's approach is ~10× faster in response.

### Latency Optimization Techniques

Several techniques ensure low latency:

1. **Continuous Processing**
   - Moshi does not wait for a full utterance or a "stop speaking" signal
   - It uses the multi-stream model to handle partial overlap
   - This eliminates the need for a separate VAD (voice activity detector), which typically adds hundreds of ms

2. **Token Delay Tuning**
   - The **token delay** mechanism can be tuned to trade off latency vs. understanding
   - Delaying Moshi's audio tokens by a couple seconds effectively turns it into a streaming TTS
   - Delaying the text tokens yields a streaming ASR
   - The live system likely uses a small fixed text delay (τ=1 corresponds to ~80ms)

3. **Efficient Computation**
   - The computational graph is lightweight per step
   - Mixed precision and model compilation techniques are employed
   - Moshi can **run in real-time on a single NVIDIA L4 GPU or an Apple M3 MacBook Pro**

### Streaming Implementation

In a practical streaming implementation:
- A **scheduler thread** reads microphone audio continuously, chunking it into 80ms frames
- Each frame is encoded via Mimi and appended to the user token sequence
- Moshi's generate function is invoked for one step to produce new Moshi tokens
- New Moshi audio tokens are decoded to sound and immediately played
- This loop runs fast enough to keep up with real time

For managing long conversations, a **sliding context window** approach is often used:
- After a certain number of steps (say, 3000 steps ≈ 4 minutes), the context may be trimmed from the beginning
- This keeps the prompt length manageable within Moshi's 4096 token context limit
- For extended sessions, earlier dialogue could be summarized into a brief prompt if needed

## Real-Time Dialogue Behavior

Moshi's full-duplex capability enables **natural conversational behaviors** that earlier systems could not achieve:

### Backchanneling
- Moshi can produce **backchannel acknowledgments** ("mm-hmm", "I see") while the user is speaking
- This shows it's actively listening without interrupting the user's flow
- The model can decide to inject a short response on its stream without waiting for the user to finish

### Interruption Handling
- If the user interjects while Moshi is talking, Moshi might stop its speech generation mid-way
- The model can adjust on the fly based on user interruptions
- This behavior was learned from training on overlapping speech in Fisher and synthetic data
- Moshi can learn to yield the floor when the user speaks over it

### Dynamic Interaction
- Conversations don't strictly alternate with long silences
- There can be overlaps and quick interjections, which Moshi can navigate
- This creates a more **dynamic, human-like interaction** pattern
- The model's ability to both listen and speak simultaneously enables more natural conversation flow

### Response Timing
- Moshi can begin responding before the user has finished their complete thought
- It can predict likely completions of user questions based on context
- The system can provide immediate feedback rather than awkward silences
- This maintains conversation momentum similar to human-human dialogue

## Deployment, Scalability, and Hardware

The Moshi model is designed to be deployed in various settings, from cloud services to local devices. Key considerations include:

### Hardware Requirements
- **GPU Requirements**: Moshi can run on mid-range GPUs like the NVIDIA L4 (22 TFLOPS)
- **Apple Silicon**: Optimized to run on Apple M3 chips using the MLX framework
- **Memory Usage**: The 7B parameter model requires approximately 14-28GB of memory depending on precision

### Optimization Techniques
- **Quantization**: Models can be quantized to int8 or int4 precision to reduce memory footprint
- **Attention Caching**: KV-cache mechanisms optimize repeated computation during streaming
- **Batching**: In server environments, multiple conversations can be batched for efficiency
- **Model Compression**: Smaller variants might use techniques like pruning or distillation

### Scaling Considerations
- **Server Deployment**: One GPU can likely handle multiple concurrent conversations
- **Edge Deployment**: On-device operation requires optimization for mobile/edge hardware
- **Hybrid Approaches**: Some implementations might use cloud for heavy computation and edge for audio I/O

### Integration Points
- **Voice Assistants**: Moshi can be integrated into voice assistant platforms
- **Communication Tools**: Video conferencing and messaging apps can incorporate the technology
- **Direct Interfaces**: Standalone applications for direct human-AI conversation

Moshi's implementation shows that complex conversational AI can be deployed with reasonable hardware requirements while maintaining near-human response times. This enables practical applications in everyday devices and services.

---

## Navigation

* [Back to Index](index.md)
* Previous: [Training Procedures](training.md)
* Next: [Model Architecture Details](model_architecture.md)