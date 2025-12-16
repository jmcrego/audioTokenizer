#!/usr/bin/env python3

import soxr
import time
import json
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
    """Load WAV from file or an audio chunk (float32 numpy array), convert to mono (channel), resample (sample_rate), normalize, ..."""

    if isinstance(audio_input, str):
        wav, sr = sf.read(audio_input)
    elif isinstance(audio_input, np.ndarray):
        wav = audio_input
        sr = sample_rate 
    else:
        raise ValueError("audio_input must be a path or np.ndarray")
    logger.debug(f"preprocess: wav size={wav.shape} sr={sr} time={wav.shape[0]/sr:.2f} sec")

    # -----------------------------
    # --- mono CHANNEL ------------
    # -----------------------------
    if len(wav.shape) > 1:
        if channel == -1:
            wav = np.mean(wav, axis=1)
        else:
            wav = wav[:, channel]
        logger.debug(f"preprocess: handled channels, wav size={wav.shape} time={wav.shape[0]/sr:.2f} sec")

    # -----------------------------
    # --- RESAMPLE ----------------
    # -----------------------------
    if sr != sample_rate:
        wav = soxr.resample(wav, sr, sample_rate)
        logger.debug(f"preprocess: resampled, wav size={wav.shape} sr={sample_rate} time={wav.shape[0]/sample_rate:.2f} sec")

    # -----------------------------
    # --- Normalize audio amplitude
    # -----------------------------
    wav = wav / max(1e-8, np.abs(wav).max())

    # -----------------------------
    # --- ENSURE float32 dtype ----
    # -----------------------------
    wav = wav.astype(np.float32)

    return wav


class Embedder(nn.Module):
    """
    Audio embedding extractor for: 'mhubert', 'wav2vec2', 'whisper'
    No manual chunking; handles padding and masks.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.path = config["path"]

        if "mhubert" in self.path.lower():
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.path)
            self.embedder = HubertModel.from_pretrained(self.path)
            self.embedding_dim = self.embedder.config.hidden_size
            # Disable augmentation
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
            raise ValueError(f"Unknown model: {self.path}")

        self.sample_rate = self.feature_extractor.sampling_rate
        self.l2_norm = config.get("l2_norm", False)
        self.downsample_ratio = self._downsample_ratio()

        logger.info(f"Loaded {self.path}, embedding_dim={self.embedding_dim}, sample_rate={self.sample_rate} downsample_ratio={self.downsample_ratio}")

    def forward(self, audio_inputs):
        """
        Args:
            audio_inputs: list of str paths or np.ndarray audio
        Returns:
            embeddings: [B, T_max, D] float32
            masks: [B, T_max] bool
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # --- Preprocess all audios ---
        preprocessed = [preprocess_audio(a, sample_rate=self.sample_rate) for a in audio_inputs]
        lengths = [len(a) for a in preprocessed]  # number of samples per audio

        # --- Pad sequences to max length ---
        max_len = max(lengths)
        batch = np.stack([np.pad(a, (0, max_len - len(a))) for a in preprocessed])  # float32, [B, T_samples]
        masks = np.stack([np.pad(np.ones(len(a), dtype=bool), (0, max_len - len(a))) for a in preprocessed])  # bool, [B, T_samples]

        input_dict = self.feature_extractor(batch, sampling_rate=self.sample_rate, return_tensors="pt", padding=False)
        inputs = input_dict.input_values if "whisper" not in self.path.lower() else input_dict.input_features
        inputs = inputs.to(device=device, dtype=dtype)  # [B, T_frames, F], float32

        # Compute frames (embeddings)
        frames = self.embedder(inputs).last_hidden_state  # [B, T_frames, D], float32

        # --- Optional L2 normalization ---
        if self.l2_norm:
            frames = torch.nn.functional.normalize(frames, dim=-1)  # [B, T_frames, D], float32

        # Downsample mask: sample-level → frame-level (each downsample_ratio samples is one frame)
        # sample idx:   0 1 2 3 | 4 5 6 7 | 8 9 10 11
        # mask value:   1 1 1 1 | 1 1 0 0 | 0 0 0 0
        # using: frame_masks = masks[:, ::4]
        # kept idx:     0       4       8
        # frame_masks:  1       1       0
        # this is, a mask is valid (not padded) if its first audio sample is valid (not padded)
        frames_masks = masks[:, ::self.downsample_ratio]
        frames_masks = frames_masks[:, :frames.shape[1]]  # same length than frames
        frames_masks = torch.from_numpy(frames_masks).to(device)

        return frames, frames_masks  # out: [B, T', D] float32, mask_tensor: [B, T] bool


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
    print(f"Output embeddings {embeddings.shape}, maks {masks.shape}, took {time.time()-t:.2f} sec")
