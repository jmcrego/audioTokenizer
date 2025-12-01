#!/usr/bin/env python3
"""
Class to extract embeddings from audio using mHuBERT, wav2vec2, or Whisper encoder.
Supports WAV/MP3 files or raw numpy audio arrays.
"""

import torch
import logging
import numpy as np

from Utils import arguments, descr

logger = logging.getLogger("audio_embedder")

class AudioEmbedder:
    """
    Audio embeddings extractor.
    Models supported: 'mhubert-147', 'wav2vec2-xlsr-53', 'whisper'
    """

    def __init__(self, audio_processor, model: str = "utter-project/mhubert-147", l2_norm: bool=True, device: str = "cpu"):
        logger.info(f"Initialing {arguments(locals())}")
        self.processor = audio_processor
        self.device = torch.device(device)
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

        self.embedder.to(self.device)
        self.embedder.eval()

    def meta(self) -> dict:
        meta = {'model': self.model, 'D': self.D, 'l2_norm': self.l2_norm, 'processor': self.processor.meta()}
        return meta

    def __call__(self, audio_input) -> torch.Tensor:
        """
        Extract embeddings from a WAV numpy array.
        Args:
            audio_input: str path to WAV file or np.ndarray (float32)
        Returns:
            embeddings: torch.Tensor [T, emb_dim]
        """
        wav = self.processor(audio_input)

        # extract features
        if "mhubert" in self.model.lower():
            input_features = self.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_values

        elif "wav2vec2" in self.model.lower():
            input_features = self.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_values

        elif "whisper" in self.model.lower():
            input_features = self.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features

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
    from AudioProcessor import AudioProcessor

    parser = argparse.ArgumentParser(description="Extract audio embeddings from file or array.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="utter-project/mHuBERT-147", help="Path or HuggingFace model name (i.e. openai/whisper-small, utter-project/mhubert-147, facebook/wav2vec2-xlsr-53 models)")
    parser.add_argument("--wav", type=str, help="Path to WAV/MP3 file")
    parser.add_argument("--top_db", type=int, default=30, help="Threshold (db) to remove silence (set 0 to avoid removing silence OR when whisper)")
    parser.add_argument("--stride", type=int, default=320, help="CNN stride used, necessary to pad audio (set 0 to avoid padding OR when whisper)")
    parser.add_argument("--rf", type=int, default=400, help="CNN receptive field used, necessary to pad audio")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_processor = AudioProcessor(top_db=args.top_db, stride=args.stride, receptive_field=args.rf)
    audio_embedder = AudioEmbedder(audio_processor, model=args.model)

    #wav = audio_processor(args.wav)
    embeddings = audio_embedder(args.wav)
