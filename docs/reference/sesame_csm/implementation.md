# Code Implementations and Publicly Available Resources

Sesame AI Labs has made significant portions of CSM available to the public, which is a boon for anyone looking to understand or replicate the model. The core resources are:

## GitHub Repository

The **CSM GitHub repository** (`SesameAILabs/csm`) contains code and documentation for the model. As of March 13, 2025, they have released the **1B-parameter variant** of CSM (1B backbone + 100M decoder) along with a pre-trained checkpoint for that model on Hugging Face Hub. The larger models (3B, 8B) were not immediately open-sourced, presumably due to their size and the sensitivity of extremely high-quality voices, but the 1B model still demonstrates the system's capabilities and provides a reference for the architecture.

The repository includes:
- Model definition
- Training and inference code
- Usage examples

The code is implemented in Python (likely using PyTorch given the context). Organization-wise, the code includes modules to load the backbone and decoder models and run generation. For instance, there's a `generator.py` that provides a high-level `generate` function to produce audio from text.

### Example Usage

Using the repo is straightforward: after installation, you can do something like:

```python
from generator import load_csm_1b
generator = load_csm_1b("path/to/ckpt.pt", device="cuda")
audio = generator.generate(text="Hello from Sesame.", speaker=0, context=[])
```

This would output a PyTorch tensor with the waveform for the spoken sentence. The `context` parameter can be used to pass in a list of prior segments (each segment containing text, speaker ID, and possibly raw audio or audio tokens) to ground the generation in conversation.

The repository likely defines a `Segment` class or similar to structure that data. The example in the README shows how to load audio files for previous utterances and prepare segments with `Segment(text=..., speaker=..., audio=...)`.

### License and Guidelines

The GitHub also contains:
- An **Apache 2.0 license**, meaning you can use and modify the code freely in your own projects
- Important information about model capabilities and limitations
- Ethical guidelines for using the model responsibly

For replication, the GitHub repository includes useful components beyond just the model code:
- A "complete architecture whitepaper" and possibly design docs
- REST API examples
- Audio preprocessing toolkit
- Model quantization guide

## Hugging Face Model and Demos

Sesame has uploaded the 1B model checkpoint to Hugging Face (`sesame/csm_1b`). While the weights are gated (requiring acceptance of terms), it makes it convenient to download and use via the `huggingface_hub` library as shown in their example code:

```python
from huggingface_hub import hf_hub_download
from generator import load_csm_1b
import torchaudio

# Download model from Hugging Face
model_path = hf_hub_download(repo_id="sesame/csm_1b", filename="csm_1b.pt")

# Load model
generator = load_csm_1b(model_path, device="cuda")

# Generate speech
audio = generator.generate(text="Hello world", speaker=0, context=[])
```

The model card on Hugging Face reiterates the architecture: a LLaMA-based backbone with a smaller decoder producing Mimi audio codes. It also links back to Sesame's site for the blog post and to the GitHub repo for code.

Hugging Face also hosts a **CSM demo Space** (likely a Gradio or Streamlit app) where users can input text and get audio out, or even simulate a conversation by inputting multi-turn dialogues. This is a quick way to test the model's outputs without setting up code locally.

## The Mimi Codec

Additionally, the **Mimi codec** itself is available as an open model: Kyutai Labs has a repository on Hugging Face for Mimi (`kyutai/mimi`) with a model card describing its function and even instructions to use it with Transformers or with Moshi.

The Mimi model card provides technical details like the frame rate (12Hz) and bitrate (1.1 kbps) and notes that Mimi's first codebook is aligned with WavLM features. It also links to a **GitHub repo for Mimi** (which might be part of the Moshi repository) and a paper reference.

If one wanted to replicate CSM's tokenizer, they could use the Mimi code and pre-trained model directly instead of training a new codec from scratch – this saves a lot of effort and ensures compatibility (Sesame chose Mimi presumably because it was excellent and open).

## Other Resources and Discussions

The research community has been actively discussing CSM and similar models. There are references to CSM in articles and forums, such as:

- Ars Technica articles about the demo's realism and the concept of "voice presence"
- YouTube demos 
- The Product Hunt listing

These provide qualitative insight and third-party impressions which can be valuable in understanding how humans perceive the model.

Kyutai Labs' Moshi project provides a point of comparison; their GitHub (`kyutai-labs/moshi`) is open-source and includes not only code but also a preprint on arXiv detailing Moshi's architecture. By reading Moshi's code (which is in the same domain of speech-text modeling), an engineer can glean ideas about implementing streaming ASR, integrating a text model, etc., which could be complementary to what CSM does.

## Replication Opportunities

These resources mean one does not have to start from zero when implementing or studying CSM. You can:

- Use the open-source **CSM 1B model** to experiment and familiarize yourself with the model's behavior.
- Leverage the **code** to see how the model class is built.
- Use the **preprocessing scripts** to prepare any custom data.
- Follow the **evaluation scripts or metrics** if provided.

One thing to note: at the time of writing, the **full training code** for CSM had not been released, though the wiki suggests it may come later in 2025. This means that while we have the model architecture and an inference code path, setting up the exact same training (with distributed data loading, etc.) might require some work and guesswork.

In addition to Sesame's and Kyutai's resources, it's wise to look at related research like **AudioLM, VALL-E, and SpeechLM**. Some of those projects have released code or models that, while not identical in purpose, share components (e.g., AudioLM's use of SoundStream tokens, VALL-E's approach to zero-shot voice cloning with codebooks).

## Summary of Available Resources

The publicly available resources for CSM include:
- Open-source **code repository**
- Ready-to-use **1B model checkpoint**
- The underlying **Mimi codec code/model**
- A detailed **technical blog post**
- Various **evaluation results and discussions**

Together these give a comprehensive picture that a machine learning engineer can use to fully understand CSM's design. By leveraging the code and models, one can experiment hands-on, which is invaluable for learning or attempting a reimplementation of the model from scratch.

---

## Navigation

* [Back to Index](index.md)
* Previous: [Inference and Processing](inference.md)
* Next: [Comparison with Moshi](comparison.md)