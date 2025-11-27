#!/usr/bin/env python3
"""
Class to extract embeddings from audio using mHuBERT, wav2vec2, or Whisper encoder.
Supports WAV/MP3 files or raw numpy audio arrays.
"""

import soxr
import torch
import logging
import numpy as np
import soundfile as sf
import librosa

from Utils import load_wav

logger = logging.getLogger("audio_embedder")

def remove_silence(wav, top_db=30):
    # wav: np.ndarray, shape [N,]
    # returns a trimmed wav
    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    return trimmed

class AudioEmbedder:
    """
    Audio embeddings extractor.
    Models supported: 'mhubert-147', 'wav2vec2-xlsr-53', 'whisper'
    """

    def __init__(self, model_name: str = "utter-project/mhubert-147", device: str = "cpu", remove_silence: bool = False):
        self.device = torch.device(device)
        self.model_name = model_name.lower()
        self.sample_rate = 16000  # default for all considered models
        self.remove_silence = remove_silence
        self.stride = None #needed to pad the audio chunk unless whisper
        self.receptive_field = None #needed to pad the audio chunk unless whisper
        #always 1500 features

        if "mhubert" in model_name.lower():
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = HubertModel.from_pretrained(model_name)
            self.stride = 320
            self.receptive_field = 400
            #7 layers with total stride: 5 × 2 × 2 × 2 × 2 × 2 × 2 = 320
            #every 320 audio samples → 1 embedding frame
            
        elif "wav2vec2" in model_name.lower():
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            self.stride = 320
            self.receptive_field = 400
            #7 layers with total stride: 5 × 2 × 2 × 2 × 2 × 2 × 2 = 320
            #every 320 audio samples → 1 embedding frame

        elif "whisper" in self.model_name.lower():
            from transformers import WhisperFeatureExtractor, WhisperModel
            self.processor = WhisperFeatureExtractor.from_pretrained(model_name)
            self.model = WhisperModel.from_pretrained(model_name).encoder

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, audio_input, channel: int = 0) -> torch.Tensor:
        """
        Extract embeddings from a WAV file path or numpy array.
        Args:
            audio_input: str path to WAV file or np.ndarray (float32)
        Returns:
            embeddings: torch.Tensor [T, emb_dim]
        """
        if isinstance(audio_input, str):
            wav = load_wav(audio_input, channel=channel, sample_rate=self.sample_rate)
        elif isinstance(audio_input, np.ndarray):
            wav = audio_input
        else:
            raise ValueError("audio_input must be a path or np.ndarray")

        logger.info(f"wav size is {wav.shape}")

        # --- REMOVE SILENCE ---
        if self.remove_silence:
            wav = remove_silence(wav, top_db=30)


        # --- PAD THE AUDIO TO MATCH THE STRIDE ---
        if self.stride is not None: #mHuBERT / wav2vec2
            remainder = (len(wav) - self.receptive_field) % self.stride
            if remainder != 0:
                pad_len = self.stride - remainder
                wav = np.pad(wav, (0, pad_len), mode='constant') 
                logger.info(f"Padded wav by {pad_len} samples, new size {wav.shape}")

        # extract features
        if "mhubert" in self.model_name.lower():
            inputs = self.processor(wav, sampling_rate=self.sample_rate, return_tensors="pt").input_values

        elif "wav2vec2" in self.model_name.lower():
            inputs = self.processor(wav, sampling_rate=self.sample_rate, return_tensors="pt").input_values

        elif "whisper" in self.model_name.lower():
            inputs = self.processor(wav, sampling_rate=self.sample_rate, return_tensors="pt").input_features

        else:
            raise ValueError("Unsupported model")

        logger.info(f"inputs size is {inputs.shape}")

        # compute embeddings
        inputs = inputs.to(self.device)
        with torch.no_grad():
            embeddings = self.model(inputs).last_hidden_state.squeeze(0)  # [T, emb_dim]

        #L2-normalize embeddings for better clustering
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8) 

        logger.info(f"embeddings size is {embeddings.shape}")
        return embeddings

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract audio embeddings from file or array.")
    parser.add_argument("--model", type=str, default="utter-project/mHuBERT-147", help="Path or HuggingFace model name (i.e. openai/whisper-small, utter-project/mhubert-147, facebook/wav2vec2-xlsr-53 models)")
    parser.add_argument("--wav", type=str, help="Path to WAV/MP3 file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_embedder = AudioEmbedder(model_name=args.model)
    embeddings = audio_embedder(args.wav)
    print("Embeddings shape:", embeddings.shape)
