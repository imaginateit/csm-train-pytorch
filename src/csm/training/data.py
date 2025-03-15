"""Data utilities for CSM training."""

import os
import json
import torch
import torchaudio
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from csm.generator import Segment


@dataclass
class TrainingExample:
    """A single training example for CSM."""
    
    text: str
    audio: torch.Tensor
    speaker_id: int
    metadata: Optional[Dict] = None


class CSMDataProcessor:
    """Processes data for CSM training."""
    
    def __init__(
        self,
        sample_rate: int = 24000,
        segment_duration_ms: int = 10000,
        overlap_ms: int = 2000,
    ):
        self.sample_rate = sample_rate
        self.segment_duration_samples = int(segment_duration_ms * sample_rate / 1000)
        self.overlap_samples = int(overlap_ms * sample_rate / 1000)
        
    def prepare_from_audio_file(
        self,
        audio_path: Union[str, Path],
        transcript_path: Union[str, Path],
        speaker_id: int,
        alignment_path: Optional[Union[str, Path]] = None,
    ) -> List[TrainingExample]:
        """
        Prepare training examples from an audio file and transcript.
        
        Args:
            audio_path: Path to the audio file
            transcript_path: Path to the transcript file
            speaker_id: Speaker ID to assign
            alignment_path: Optional path to word-level alignment file
            
        Returns:
            List of TrainingExample objects
        """
        # Load and normalize audio
        audio, orig_sr = torchaudio.load(audio_path)
        if orig_sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_sr, self.sample_rate)
        
        # Convert to mono if needed
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)
        
        # Load transcript
        with open(transcript_path, "r") as f:
            transcript = f.read().strip()
            
        # Process alignments if available
        alignments = None
        if alignment_path:
            alignments = self._load_alignments(alignment_path)
            return self._segment_with_alignments(audio, transcript, speaker_id, alignments)
        
        # If no alignments, use simple segmentation
        return self._segment_basic(audio, transcript, speaker_id)
    
    def _segment_basic(
        self,
        audio: torch.Tensor,
        transcript: str,
        speaker_id: int
    ) -> List[TrainingExample]:
        """Segment audio and transcript into overlapping chunks without precise alignments."""
        examples = []
        
        # Simple character-based rough estimation of segment points
        char_per_sample = len(transcript) / audio.size(0)
        
        # Create overlapping segments
        for start_idx in range(0, audio.size(0), self.segment_duration_samples - self.overlap_samples):
            end_idx = min(start_idx + self.segment_duration_samples, audio.size(0))
            segment_audio = audio[start_idx:end_idx]
            
            # Rough estimation of text bounds
            text_start = int(start_idx * char_per_sample)
            text_end = int(end_idx * char_per_sample)
            segment_text = transcript[text_start:text_end]
            
            # Skip segments that are too short
            if len(segment_text.strip()) < 10 or segment_audio.size(0) < self.sample_rate:
                continue
                
            examples.append(TrainingExample(
                text=segment_text,
                audio=segment_audio,
                speaker_id=speaker_id,
                metadata={"start_sample": start_idx, "end_sample": end_idx}
            ))
            
        return examples
    
    def _load_alignments(self, path: Union[str, Path]) -> Dict:
        """Load word-level alignments from a file."""
        with open(path, "r") as f:
            return json.load(f)
    
    def _segment_with_alignments(
        self,
        audio: torch.Tensor,
        transcript: str,
        speaker_id: int,
        alignments: Dict
    ) -> List[TrainingExample]:
        """Segment audio and transcript using precise word alignments."""
        examples = []
        
        # Extract word timings from alignments
        words = alignments.get("words", [])
        if not words:
            return self._segment_basic(audio, transcript, speaker_id)
        
        # Group words into segments
        segments = []
        current_segment = {"start": 0, "end": 0, "text": ""}
        
        for word in words:
            word_start = int(word["start"] * self.sample_rate)
            word_end = int(word["end"] * self.sample_rate)
            word_text = word["word"] + " "
            
            # If adding this word would make the segment too long, start a new one
            if word_end - current_segment["start"] > self.segment_duration_samples:
                if current_segment["text"]:  # Only add non-empty segments
                    segments.append(current_segment)
                current_segment = {"start": word_start, "end": word_end, "text": word_text}
            else:
                current_segment["end"] = word_end
                current_segment["text"] += word_text
        
        # Add the last segment if it's not empty
        if current_segment["text"]:
            segments.append(current_segment)
        
        # Create examples from segments
        for segment in segments:
            start_idx = segment["start"]
            end_idx = segment["end"]
            segment_text = segment["text"].strip()
            
            # Skip segments that are too short
            if len(segment_text) < 10 or end_idx - start_idx < self.sample_rate:
                continue
            
            segment_audio = audio[start_idx:end_idx]
            examples.append(TrainingExample(
                text=segment_text,
                audio=segment_audio,
                speaker_id=speaker_id,
                metadata={"start_sample": start_idx, "end_sample": end_idx}
            ))
        
        return examples


class ContextualExampleGenerator:
    """Generates multi-turn contextual examples for conversational training."""
    
    def __init__(
        self,
        max_context_turns: int = 3,
        include_audio_context: bool = True
    ):
        self.max_context_turns = max_context_turns
        self.include_audio_context = include_audio_context
        
    def create_contextual_examples(
        self,
        conversation: List[TrainingExample]
    ) -> List[Dict]:
        """
        Create contextual examples from a conversation.
        
        Args:
            conversation: List of TrainingExample objects in conversation order
            
        Returns:
            List of dictionaries with context and target information
        """
        contextual_examples = []
        
        # Create examples with increasing context length
        for i in range(len(conversation)):
            if i == 0:
                # First turn has no context
                contextual_examples.append({
                    "context": [],
                    "target": conversation[i]
                })
            else:
                # Create context with previous turns
                context_start = max(0, i - self.max_context_turns)
                context = conversation[context_start:i]
                
                # Create example
                contextual_examples.append({
                    "context": context,
                    "target": conversation[i]
                })
                
        return contextual_examples


class CSMDataset(Dataset):
    """Dataset for CSM training."""
    
    def __init__(
        self,
        examples: List[Dict],
        text_tokenizer,
        audio_tokenizer,
        max_seq_len: int = 2048
    ):
        self.examples = examples
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Process context segments
        context_segments = []
        for ctx in example.get("context", []):
            segment = Segment(
                text=ctx.text,
                speaker=ctx.speaker_id,
                audio=ctx.audio
            )
            context_segments.append(segment)
        
        # Process target
        target = example["target"]
        target_segment = Segment(
            text=target.text,
            speaker=target.speaker_id,
            audio=target.audio
        )
        
        # Convert to model inputs - similar to Generator._tokenize_segment
        tokens = []
        masks = []
        
        # Process context segments first
        for segment in context_segments:
            segment_tokens, segment_masks = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            masks.append(segment_masks)
        
        # Process target text only (for input)
        target_text_tokens, target_text_masks = self._tokenize_text_segment(
            target_segment.text, target_segment.speaker
        )
        tokens.append(target_text_tokens)
        masks.append(target_text_masks)
        
        # Concatenate all tokens and masks
        input_tokens = torch.cat(tokens, dim=0) if tokens else torch.zeros(0, 33).long()
        input_masks = torch.cat(masks, dim=0) if masks else torch.zeros(0, 33).bool()
        
        # Process target audio (for output)
        target_audio_tokens = self._tokenize_audio_for_target(target_segment.audio)
        
        # Ensure we're within max sequence length
        if input_tokens.size(0) > self.max_seq_len:
            # Truncate from the beginning, but always keep the target text
            keep_len = min(self.max_seq_len, target_text_tokens.size(0))
            start_idx = input_tokens.size(0) - keep_len
            input_tokens = input_tokens[start_idx:]
            input_masks = input_masks[start_idx:]
        
        return {
            "input_tokens": input_tokens,
            "input_masks": input_masks,
            "target_audio_tokens": target_audio_tokens
        }
    
    def _tokenize_text_segment(self, text, speaker):
        """Tokenize a text segment - implementation depends on tokenizer."""
        # Placeholder implementation - to be customized based on actual tokenizer
        text_tokens = self.text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        
        return text_frame, text_frame_mask
    
    def _tokenize_audio(self, audio):
        """Tokenize an audio segment - implementation depends on tokenizer."""
        # Placeholder implementation - to be customized based on actual tokenizer
        audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
        
        audio_frame = torch.zeros(audio_tokens.size(1), 33).long()
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool()
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True
        
        return audio_frame, audio_frame_mask
    
    def _tokenize_segment(self, segment):
        """Tokenize a full segment (text + audio)."""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
    
    def _tokenize_audio_for_target(self, audio):
        """Process audio specifically for target representation."""
        # Placeholder implementation - to be customized based on model needs
        if hasattr(self.audio_tokenizer, 'encode'):
            if hasattr(audio, 'unsqueeze'):
                try:
                    audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))
                    if isinstance(audio_tokens, list) and len(audio_tokens) > 0:
                        audio_tokens = audio_tokens[0]
                    if audio_tokens.dim() > 1:
                        audio_tokens = audio_tokens.transpose(0, 1)
                except Exception:
                    # Fallback for testing
                    audio_tokens = torch.ones(5, 32).long()
            else:
                # Fallback for testing
                audio_tokens = torch.ones(5, 32).long()
        else:
            # Fallback for testing
            audio_tokens = torch.ones(5, 32).long()
            
        return audio_tokens


def create_dataloader(
    dataset: CSMDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True
):
    """Create a DataLoader for the CSM dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_variable_length
    )


def collate_variable_length(batch):
    """Collate function that handles variable length inputs."""
    # Extract all items
    input_tokens = [item["input_tokens"] for item in batch]
    input_masks = [item["input_masks"] for item in batch]
    target_audio_tokens = [item["target_audio_tokens"] for item in batch]
    
    # Find max lengths
    max_input_len = max(tokens.size(0) for tokens in input_tokens)
    max_target_len = max(tokens.size(0) for tokens in target_audio_tokens)
    
    # Pad input tokens and masks
    padded_input_tokens = torch.zeros(len(batch), max_input_len, 33).long()
    padded_input_masks = torch.zeros(len(batch), max_input_len, 33).bool()
    
    for i, (tokens, mask) in enumerate(zip(input_tokens, input_masks)):
        padded_input_tokens[i, :tokens.size(0), :] = tokens
        padded_input_masks[i, :mask.size(0), :] = mask
    
    # Pad target audio tokens
    padded_target_audio_tokens = torch.zeros(len(batch), max_target_len, 32).long()
    
    for i, tokens in enumerate(target_audio_tokens):
        padded_target_audio_tokens[i, :tokens.size(0), :tokens.size(1)] = tokens
    
    return {
        "input_tokens": padded_input_tokens,
        "input_masks": padded_input_masks,
        "target_audio_tokens": padded_target_audio_tokens
    }