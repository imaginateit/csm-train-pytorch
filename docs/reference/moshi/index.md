# Moshi Documentation

Moshi is a full-duplex speech-text model architecture that enables real-time, conversational speech interactions. This documentation provides a comprehensive overview of Moshi's architecture, implementation details, and usage.

This documentation has been split into multiple pages for better readability and organization. Each page focuses on a specific aspect of the Moshi model.

## Table of Contents

1. **[Architecture and Components](architecture.md)**
   - [Helium: Text Language Model Backbone](architecture.md#helium-text-language-model-backbone-7b-llm)
   - [Mimi: Streaming Neural Audio Codec](architecture.md#mimi-streaming-neural-audio-codec)
   - [Multi-Stream Modeling: Temporal & Depth Transformers](architecture.md#multi-stream-modeling-temporal--depth-transformers-for-dual-audio-streams)

2. **[Training Procedures](training.md)**
   - [Datasets and Preprocessing](training.md#datasets-and-preprocessing)
   - [Loss Functions and Optimization](training.md#loss-functions-and-optimization)

3. **[Inference and Processing](inference.md)**
   - [Full-Duplex Streaming Mechanism](inference.md#full-duplex-streaming-mechanism)
   - [Real-Time Dialogue Behavior](inference.md#real-time-dialogue-behavior)
   - [Deployment, Scalability, and Hardware](inference.md#deployment-scalability-and-hardware)

4. **[Model Architecture Details](model_architecture.md)**
   - [Topology and Component Interaction](model_architecture.md#topology-and-component-interaction-block-diagram-explanation)
   - [Helium and Depth Transformer Details](model_architecture.md#helium-and-depth-transformer-details)
   - [Mimi Audio Codec Architecture](model_architecture.md#mimi-audio-codec-architecture)
   - [Input/Output Representations and Tokenization](model_architecture.md#inputoutput-representations-and-tokenization)

5. **[Implementation](implementation.md)**
   - [Code Structure and Frameworks](implementation.md#code-structure-and-frameworks)
   - [Validation and Testing](implementation.md#validation-and-testing)
   - [Dependencies and Engineering Challenges](implementation.md#dependencies-and-engineering-challenges)
   - [Performance and Debugging Tools](implementation.md#performance-and-debugging-tools)

6. **[Ethical Considerations](ethics.md)**
   - [Safety Measures](ethics.md#safety-measures)
   - [Responsible AI Practices](ethics.md#responsible-ai-practices)
   - [Limitations and Mitigations](ethics.md#limitations-and-mitigations)