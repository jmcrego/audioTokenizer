#!/usr/bin/env python3

import soxr
import torch
import logging
import numpy as np
import torch.nn as nn
import soundfile as sf

logger = logging.getLogger("Embedder")

# next are to speed up the embedding
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = True

def preprocess_audio(audio_input, sample_rate=16000, channel=0):
    """
    Load audio from:
      - file path (str)
      - numpy array (float waveform)

    Returns:
      wav: np.ndarray, shape [T_samples], dtype float32
    """

    # -----------------------------
    # Load audio
    # -----------------------------
    if isinstance(audio_input, str):
        wav, sr = sf.read(audio_input)              # wav: [T] or [T, C]
    elif isinstance(audio_input, np.ndarray):
        wav = audio_input
        sr = sample_rate
    else:
        raise ValueError("audio_input must be a path or np.ndarray")

    logger.debug(f"preprocess: wav shape={wav.shape}, sr={sr}")

    # -----------------------------
    # Convert to mono
    # -----------------------------
    if wav.ndim > 1:
        if channel == -1:
            wav = wav.mean(axis=1)                  # [T]
        else:
            wav = wav[:, channel]                   # [T]

    # -----------------------------
    # Resample if needed
    # -----------------------------
    if sr != sample_rate:
        wav = soxr.resample(wav, sr, sample_rate)
        sr = sample_rate

    # -----------------------------
    # Convert to float32
    # -----------------------------
    wav = wav.astype(np.float32)

    return wav


class Embedder(nn.Module):
    """
    Audio → frame embeddings extractor.

    Output:
      frames: [B, T_frames, D]   float32
      mask:   [B, T_frames]      bool (True = valid frame)
    """

    def __init__(self, config):
        super().__init__()

        self.path = config["path"]

        # ----------------------------------------------------
        # Load backbone
        # ----------------------------------------------------
        if "mhubert" in self.path.lower():
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.path)
            self.embedder = HubertModel.from_pretrained(self.path)
            self.embedding_dim = self.embedder.config.hidden_size

            # Disable SpecAugment
            self.embedder.config.mask_time_prob = 0.0
            self.embedder.config.mask_feature_prob = 0.0
            self.embedder.config.apply_spec_augment = False

        elif "wav2vec2" in self.path.lower():
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.path)
            self.embedder = Wav2Vec2Model.from_pretrained(self.path)
            self.embedding_dim = self.embedder.config.hidden_size

        elif "whisper" in self.path.lower():
            from transformers import WhisperFeatureExtractor, WhisperModel
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.path)
            self.embedder = WhisperModel.from_pretrained(self.path).encoder
            self.embedding_dim = self.embedder.config.d_model

        else:
            raise ValueError(f"Unknown audio model: {self.path}")

        self.sample_rate = self.feature_extractor.sampling_rate
        self.downsample_ratio = self._downsample_ratio()
        
        # wrapper may also do this
        self.embedder.eval()  
        for p in self.embedder.parameters():
            p.requires_grad = False

        logger.info(f"Loaded {self.path} | "
                    f"embedding_dim={self.embedding_dim} | "
                    f"sample_rate={self.sample_rate} | "
                    f"downsample_ratio={self.downsample_ratio}")


    def forward(self, audio_inputs):
        """
        Args:
            audio_inputs: list of str paths or np.ndarray audio
        Returns:
            embeddings: [B, T_max, D] float32
            masks: [B, T_max] bool
        """
        device = next(self.parameters()).device

        # ====================================================
        # 1. Preprocess each audio independently
        # ====================================================
        preprocessed = [
            preprocess_audio(a, sample_rate=self.sample_rate) 
            for a in audio_inputs
        ] # waveforms: list of np.ndarray [T_i]

        # ====================================================
        # 2. Feature extractor (handles padding + mask)
        # ====================================================

        if "whisper" in self.path.lower():
            feat = self.feature_extractor(
                preprocessed,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding="max_length",
                max_length=3000,
                truncation=True,
            )#.input_features.to(device)
        else:
            feat = self.feature_extractor(
                preprocessed,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
        # For HuBERT / wav2vec2:
        #   input_values: [B, T_samples]
        #   attention_mask: [B, T_samples]
        #
        # For Whisper:
        #   input_features: [B, n_mels, T_frames]

        if "whisper" in self.path.lower():
            inputs = feat.input_features.to(device, dtype=torch.float32)
            sample_mask = None
        else:
            inputs = feat.input_values.to(device, dtype=torch.float32)
            sample_mask = feat.attention_mask.to(device)  # [B, T_samples]

        logger.debug(
            f"Audio inputs to encoder: {inputs.shape}, dtype={inputs.dtype}"
        )

        # ====================================================
        # 3. Encoder forward
        # ====================================================
        with torch.no_grad():
            if sample_mask is not None:
                frames = self.embedder(inputs, attention_mask=sample_mask).last_hidden_state
            else:
                frames = self.embedder(inputs).last_hidden_state
            # frames: [B, T_frames, D], float32
        logger.debug(f"Frame embeddings: {frames.shape}")

        # ====================================================
        # 4. Frame-level mask 
        # ====================================================
        if "whisper" in self.path.lower():
            # Whisper encoder outputs are always dense (no mask needed; all valild)
            frames_mask = torch.ones(frames.shape[:2], dtype=torch.bool, device=device)
        else:
            frames_mask = self.embedder._get_feature_vector_attention_mask(frames.shape[1], sample_mask).bool()
        logger.debug(f"Frame mask: {frames_mask.shape}")

        return frames, frames_mask


    def _downsample_ratio(self):
        """
        Compute the ratio between number of audio samples and features (or embeddings)
        This is, how many samples are used for one feature
        """
        if "whisper" in self.path.lower():
            return self.feature_extractor.hop_length #usually 160
        stride = 1
        for layer in self.embedder.feature_extractor.conv_layers:
            stride *= layer.conv.stride[0]
        return stride #usually 320




if __name__ == "__main__":
    import argparse
    import time
    import json
    parser = argparse.ArgumentParser(description="Extract audio embeddings from file or array.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--audio_files", type=str, help="Comma separated list of audio files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    audio_embedder = Embedder(config=config['audio'])
    t = time.time()
    embeddings, masks = audio_embedder(args.audio_files.split(','))
    print(f"Output embeddings {embeddings.shape}, masks {masks.shape}, took {time.time()-t:.2f} sec")
