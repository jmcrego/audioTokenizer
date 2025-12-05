#!/usr/bin/env python3
"""
Class to extract embeddings from audio using mHuBERT, wav2vec2, or Whisper encoder.
Supports WAV/MP3 files or raw numpy audio arrays.
"""

import torch
import logging
import numpy as np
import soundfile as sf
import librosa
import soxr

# from scripts.Utils import arguments, descr
from Utils import arguments, descr

logger = logging.getLogger("audio_embedder")

def process_audio(audio_input, sample_rate=16000, channel=0, top_db=0):
    """Load WAV, convert to mono, resample, remove silence, ..."""

    if isinstance(audio_input, str):
        wav, sr = sf.read(audio_input)
    elif isinstance(audio_input, np.ndarray):
        wav = audio_input
    else:
        raise ValueError("audio_input must be a path or np.ndarray")
    logger.debug(f"wav size={wav.shape} sr={sr} time={wav.shape[0]/sr:.2f} sec")

    #  CHANNEL handling (mono)
    if len(wav.shape) > 1: 
        if wav.shape[1] > 1:
            if channel == 0: #first channel
                wav = wav[:, 0]
            elif channel == 1: #second channel
                wav = wav[:, 1]
            elif channel == -1: #average channels
                wav = np.mean(wav, axis=1)       
            else:         
                raise ValueError(f"Invalid channel {channel} for audio with {wav.shape[1]} channels")
        logger.debug(f"handled channels, wav size={wav.shape} time={wav.shape[0]/sr:.2f} sec")

    # RESAMPLE
    if sr != sample_rate:
        wav = soxr.resample(wav, sr, sample_rate)
        logger.debug(f"resampled, wav size={wav.shape} sr={sample_rate} time={wav.shape[0]/sample_rate:.2f} sec")

    # Ensure float32 dtype
    wav = wav.astype(np.float32)

    # --- REMOVE SILENCE ---
    if top_db:
        wav_trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
        logger.debug(f"removed silence, wav size={wav_trimmed.shape} time={wav_trimmed.shape[0]/sample_rate:.2f} sec")
        wav = wav_trimmed

    return wav

class AudioEmbedder:
    """
    Audio embeddings extractor.
    Models supported: 'mhubert-147', 'wav2vec2-xlsr-53', 'whisper'
    """

    def __init__(self, model: str = "utter-project/mhubert-147", top_db=0, l2_norm: bool=True, device: str = "cpu"):
        self.meta = arguments(locals())
        logger.info(f"Initializing {self.meta}")

        self.device = torch.device(device)
        self.top_db = top_db
        self.l2_norm = l2_norm
        self.model = model.lower()

        if "mhubert" in model.lower():
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model)
            self.embedder = HubertModel.from_pretrained(model)
            self.D = self.embedder.config.hidden_size

        elif "wav2vec2" in model.lower():
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model)
            self.embedder = Wav2Vec2Model.from_pretrained(model)
            self.D = self.embedder.config.hidden_size

        elif "whisper" in self.model.lower():
            from transformers import WhisperFeatureExtractor, WhisperModel
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model)
            self.embedder = WhisperModel.from_pretrained(model).encoder
            self.D = self.embedder.config.d_model

        else:
            raise ValueError(f"Unknown model: {model}")

        self.sample_rate = self.feature_extractor.sampling_rate
        self.embedder.to(self.device)
        self.embedder.eval()

    def __call__(self, audio_input) -> torch.Tensor:
        """
        Extract embeddings from a WAV numpy array.
        Args:
            audio_input: str path to WAV file or np.ndarray (float32)
        Returns:
            embeddings: torch.Tensor [T, emb_dim]
        """
        #wav = self.audio_processor(audio_input)

        # read input if necessary
        audio_input = process_audio(audio_input, sample_rate=self.sample_rate, channel=0, top_db=self.top_db)

        # extract features
        if "mhubert" in self.model.lower():
            input_features = self.feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt").input_values

        elif "wav2vec2" in self.model.lower():
            input_features = self.feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt").input_values

        elif "whisper" in self.model.lower():
            input_features = self.feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt").input_features

        else:
            raise ValueError("Unsupported model")

        logger.debug(f"input_features size={input_features.shape}")

        # compute embeddings
        input_features = input_features.to(self.device)
        with torch.no_grad():
            embeddings = self.embedder(input_features).last_hidden_state.squeeze(0)  # [T, emb_dim]

        #L2-normalize embeddings for better clustering
        if self.l2_norm:
            # Compute the L2 norm along the last dimension
            norm = torch.norm(embeddings, dim=-1, keepdim=True)
            # Avoid division by zero
            norm = torch.clamp(norm, min=1e-8)
            # Normalize the embeddings
            embeddings = embeddings / norm

        logger.debug(f"embeddings {descr(embeddings)}")
        return embeddings

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract audio embeddings from file or array.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="utter-project/mHuBERT-147", help="Path or HuggingFace model name (i.e. openai/whisper-small, utter-project/mhubert-147, facebook/wav2vec2-xlsr-53 models)")
    parser.add_argument("--wav", type=str, help="Path to WAV/MP3 file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda').")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_embedder = AudioEmbedder(model=args.model, device=args.device)
    embeddings = audio_embedder(args.wav)
