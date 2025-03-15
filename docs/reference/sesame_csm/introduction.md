# Introduction to Sesame AI Lab and CSM

Sesame AI Lab is a research team focused on advancing AI-driven voice technology to achieve what they call "voice presence" – the feeling that a machine's voice is as genuine and engaging as a human's. Traditional voice assistants and text-to-speech systems often sound neutral and lack emotional depth, which makes interactions feel flat over time. To address this, Sesame AI Lab set out to create AI companions that can carry _interactive, context-aware conversations_ with natural prosody and emotion. Their Conversational Speech Model (CSM) is the centerpiece of this effort, designed to generate speech that not only has high audio quality but also responds appropriately to the conversational context.

CSM's development was driven by key innovations in conversational speech generation:

## End-to-End Multimodal Approach

Unlike conventional two-stage pipelines (where a model predicts a transcript or prosody tokens and a separate vocoder produces audio), CSM directly generates audio tokens from text and audio context in a single integrated process. This _single-stage_ approach improves efficiency and expressivity by allowing the model to jointly consider linguistic content and acoustic context when producing speech.

## Conversation History Utilization

CSM is explicitly designed to leverage **conversation history** – it looks at previous dialogue (both what was said and how it was said) to inform the speech it generates. This helps solve the "one-to-many" mapping problem in speech generation: a given sentence can be spoken in many ways, and by providing context (tone, tempo, recent dialogue), CSM can choose a rendition that fits naturally.

In practice, CSM can adapt its speaking style to the situation – for example, sounding sympathetic if the prior user turn was sad, or more energetic if the conversation is upbeat.

## Novel Evaluation Metrics

Sesame's focus on new evaluation metrics and realistic benchmarks for conversational speech sets CSM apart. They found that common metrics like word error rate or speaker similarity are saturated (modern TTS models already score near-human), so they introduced context-specific tests (like disambiguating homographs through context, and maintaining pronunciation consistency in dialogue) to measure CSM's understanding of context.

These efforts underscore that CSM isn't just about generating **audio** – it's about generating the **right** audio for a given conversational moment.

---

## Navigation

* [Back to Index](index.md)
* Next: [Model Architecture](architecture.md)