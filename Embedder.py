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
        embedding_dim = config["embedding_dim"]

        if "mhubert" in self.path.lower():
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.path)
            self.embedder = HubertModel.from_pretrained(self.path)
            assert embedding_dim == self.embedder.config.hidden_size
            # Disable augmentation
            self.embedder.config.mask_time_prob = 0.0
            self.embedder.config.mask_feature_prob = 0.0
            self.embedder.config.apply_spec_augment = False

        elif "wav2vec2" in self.path.lower():
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.path)
            self.embedder = Wav2Vec2Model.from_pretrained(self.path)
            assert embedding_dim == self.embedder.config.hidden_size

        elif "whisper" in self.path.lower():
            from transformers import WhisperFeatureExtractor, WhisperModel
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.path)
            self.embedder = WhisperModel.from_pretrained(self.path).encoder
            assert embedding_dim == self.embedder.config.d_model
        else:
            raise ValueError(f"Unknown model: {self.path}")

        self.sample_rate = self.feature_extractor.sampling_rate
        self.l2_norm = config.get("l2_norm", False)
        self.ratio = self._downsample_ratio()

        logger.info(f"Loaded {self.path}, embedding_dim={embedding_dim}, sample_rate={self.sample_rate}")

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

        # Downsample mask: sample-level → frame-level (each ratio samples is one frame)
        # sample idx:   0 1 2 3 | 4 5 6 7 | 8 9 10 11
        # mask value:   1 1 1 1 | 1 1 0 0 | 0 0 0 0
        # using: frame_masks = masks[:, ::4]
        # kept idx:     0       4       8
        # frame_masks:  1       1       0
        # this is, a mask is valid (not padded) if its first audio sample is valid (not padded)
        frames_masks = masks[:, ::self.ratio]
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



    # def forward(self, audio_inputs) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Extract embeddings from a batch of audio files or numpy arrays with chunk/stride.
    #     Args:
    #         audio_inputs: List of str paths or np.ndarray audio chunks.
    #     Returns:
    #         embeddings: torch.Tensor [B, T, D] padded to the longest sequence
    #         mask: torch.BoolTensor [B, T] indicating valid frames
    #     """
    #     param = next(self.parameters())
    #     device = param.device
    #     dtype  = param.dtype
    #     embedding_dim = self.config['embedding_dim']
    #     l2_norm = self.config['l2_norm']

    #     all_chunks = []
    #     lengths = []

    #     # ----------------------------------------------
    #     # --- preprocess audios and build chunks -------
    #     # ----------------------------------------------
    #     t = time.time()
    #     for i, audio in enumerate(audio_inputs):
    #         wav = preprocess_audio(audio, sample_rate=self.sample_rate)
    #         n_samples = len(wav)
    #         # # 1. Compute chunk start positions
    #         # starts = np.arange(0, n_samples, stride)
    #         # # 2. Pad wav ONCE so all end slices exist
    #         # padded_len = starts[-1] + chunk_size
    #         # if padded_len > n_samples:
    #         #     wav = np.pad(wav, (0, padded_len - n_samples))
    #         # # 3. Extract chunks: fast vectorized slicing
    #         # chunks = np.stack([wav[s:s + chunk_size] for s in starts])
    #         # # results
    #         # all_chunks.append(chunks) # [n_chunks, chunk_size]
    #         # lengths.append(len(chunks)) # number of chunks per audio input
    #         # logger.debug(f"audio {i}, n_samples={n_samples} n_chunks={len(chunks)} time={n_samples/self.sample_rate:.2f} sec")
    #     t_preprocess = time.time()-t

    #     # ----------------------------------------------
    #     # --- concat chunks, extract feats/embeds ------
    #     # ----------------------------------------------
    #     t = time.time()
    #     # Concatenate all chunks for batch processing
    #     batch_chunks = np.concatenate(all_chunks, axis=0)  # [C, cs] # C ~ Total chunks; cs ~ chunk size (number of samples in a chunk)
    #     logger.debug(f"Concatenated n_chunks={batch_chunks.shape[0]} chunk_size={chunk_size} samples")

    #     # Prepare waveforms for the embedding (not feature extraction)
    #     input_dict = self.feature_extractor(batch_chunks, sampling_rate=self.sample_rate, return_tensors="pt", padding=False)
    #     inputs = input_dict.input_values if "whisper" not in path.lower() else input_dict.input_features

    #     if device.type == "cuda":
    #         inputs = inputs.to(device, dtype=dtype, non_blocking=True)
    #     else:
    #         inputs = inputs.to(device, dtype=dtype)

    #     #C ~ batch size (total number of chunks)
    #     #F ~ time dimension (number of frames per audio chunk)
    #     #f ~ feature dimension (for spectrograms)

    #     # Forward pass
    #     with torch.no_grad():
    #         out = self.embedder(inputs).last_hidden_state  # [C, E, D] # E ~ number of embeddings in chunk (frames) # D ~ embedding dimension

    #     t_embeddings = time.time()-t
    #     logger.debug(f"Extracted embeddings {out.shape} dtype={out.dtype}")

    #     # Optional L2 normalization (only for computing clusters)
    #     if l2_norm:
    #         out = torch.nn.functional.normalize(out, dim=-1)

    #     # ----------------------------------------------
    #     # --- back to original format ------------------
    #     # ----------------------------------------------
    #     t = time.time()
    #     # Split outputs back into original audios (B), each audio input is an entry in batch
    #     embeddings = []
    #     masks = []
    #     idx = 0
    #     for i, n_chunks in enumerate(lengths): #n_chunks (nC) is the number of chunks on each audio file
    #         emb_audio = out[idx: idx + n_chunks]  # [nC_i, E, D] #nC_i ~ number of chunks in this audio file
    #         idx += n_chunks
    #         # Flatten chunks along time dimension
    #         emb_audio = emb_audio.reshape(-1, embedding_dim)  # [nC_i*E, D] # nC_i*E is the number of embeddings in current audio file
    #         embeddings.append(emb_audio)
    #         # mask: valid embeddings are all ones as we padded only at audio level
    #         mask = torch.ones(emb_audio.shape[0], dtype=torch.bool, device=device) #[nC_i*E]
    #         masks.append(mask)
    #         logger.debug(f"Audio {i} embeddings = {emb_audio.shape} mask = {mask.shape}")

    #     #embeddings ~ [B, nC_i*E, D] (nC_i*E is different on each list element)
    #     #masks = [B, nC_i*E]

    #     # ----------------------------------------------
    #     # --- add padding and return tensors -----------
    #     # ----------------------------------------------
    #     # Pad all sequences to the max length of embeddings (T)
    #     max_len = max(e.shape[0] for e in embeddings)
    #     padded_embeddings = torch.stack([torch.nn.functional.pad(e, (0,0,0,max_len - e.shape[0])) for e in embeddings]) #[B, T, D] 
    #     logger.debug(f"Padded embeddings: {padded_embeddings.shape}")
    #     padded_masks = torch.stack([torch.nn.functional.pad(m, (0,max_len - m.shape[0])) for m in masks]) #[B, T]
    #     logger.debug(f"Padded masks: {padded_masks.shape} invalid frames={(~padded_masks).sum().item()}")
    #     t_formatting = time.time()-t

    #     logger.debug(f"Embedder times (msec): preprocess={1000*t_preprocess:.1f}, embedding={1000*t_embeddings:.1f}, formatting={1000*t_formatting:.1f}")
    #     return padded_embeddings, padded_masks




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
