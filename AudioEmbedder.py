#!/usr/bin/env python3
"""
Class to extract embeddings from audio using mHuBERT, wav2vec2, or Whisper encoder.
Supports WAV/MP3 files or raw numpy audio arrays.
"""

import torch
import logging
import numpy as np
import soundfile as sf
import soxr

logger = logging.getLogger("audio_embedder")

def arguments(args):
    args.pop('self', None)  # None prevents KeyError if 'self' doesn't exist
    return args


def preprocess_audio(audio_input, sample_rate=16000, channel=0):
    """Load WAV from file or an audio chunk (float32 numpy array), convert to mono (channel), resample (sample_rate), ..."""

    if isinstance(audio_input, str):
        wav, sr = sf.read(audio_input)
    elif isinstance(audio_input, np.ndarray):
        wav = audio_input
        sr = sample_rate 
    else:
        raise ValueError("audio_input must be a path or np.ndarray")
    logger.debug(f"wav size={wav.shape} sr={sr} time={wav.shape[0]/sr:.2f} sec")

    # -----------------------------
    # --- mono CHANNEL ------------
    # -----------------------------
    if len(wav.shape) > 1: 
        if channel == -1:
            wav = np.mean(wav, axis=1)
        else:
            wav = wav[:, channel]
        logger.debug(f"handled channels, wav size={wav.shape} time={wav.shape[0]/sr:.2f} sec")

    # -----------------------------
    # --- RESAMPLE ----------------
    # -----------------------------
    if sr != sample_rate:
        wav = soxr.resample(wav, sr, sample_rate)
        logger.debug(f"resampled, wav size={wav.shape} sr={sample_rate} time={wav.shape[0]/sample_rate:.2f} sec")

    # -----------------------------
    # --- Normalize audio amplitude
    # -----------------------------
    wav = wav / max(1e-8, np.abs(wav).max())

    # -----------------------------
    # --- ENSURE float32 dtype ----
    # -----------------------------
    wav = wav.astype(np.float32)

    return wav


class AudioEmbedder:
    """
    Audio embeddings extractor.
    Models supported: 'mhubert-147', 'wav2vec2-xlsr-53', 'whisper'
    """

    def __init__(self, model: str = "utter-project/mhubert-147", l2_norm: bool=False, half_precision: bool=False, device: str = "cpu"):
        self.meta = arguments(locals())
        logger.info(f"Initializing {self.meta}")

        self.device = torch.device(device)
        self.l2_norm = l2_norm
        self.model = model.lower()
        self.half_precision = half_precision
        self.window_size = 16000  # e.g., 1 second windows at 16 kHz
        self.stride = 320          # e.g., 20ms stride = 320 samples at 16 kHz


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
        if self.half_precision:
            self.embedder = self.embedder.half() #when using A100/H100
        self.embedder = torch.compile(self.embedder)
        self.embedder.eval()


    def __call__(self, audio_inputs) -> tuple[torch.Tensor, list[int]]:
        """
        Extract embeddings from a batch of audio files or numpy arrays.
        
        Args:
            audio_inputs: List of str paths or np.ndarray audio chunks.
            
        Returns:
            embeddings: torch.Tensor [B, T, D] padded to the longest sequence
            lengths: List[int] original lengths of each audio input (in frames)
        """

        all_embeddings = []
        all_masks = []

        # Preprocess audio
        waves = [preprocess_audio(f, sample_rate=self.sample_rate) for f in audio_inputs]
        for w in waves:
            frames = []
            frame_masks = []
            start = 0
            while start < len(w):
                end = min(start + self.window_size, len(w))
                chunk = w[start:end]

                # pad to window_size
                pad_len = self.window_size - len(chunk)
                if pad_len > 0:
                    chunk = np.pad(chunk, (0, pad_len))

                # feature extraction
                input_dict = self.feature_extractor(chunk, sampling_rate=self.sample_rate, return_tensors="pt")
                inputs = input_dict.input_values if "whisper" not in self.model.lower() else input_dict.input_features
                inputs = inputs.to(self.device)
                if self.half_precision:
                    inputs = inputs.half()

                with torch.inference_mode():
                    out = self.embedder(inputs).last_hidden_state.squeeze(0)  # [T, D]

                if self.l2_norm:
                    out = torch.nn.functional.normalize(out, dim=-1)

                frames.append(out)
                frame_masks.append(torch.ones(out.shape[0], dtype=torch.bool))

                start += self.stride  # move window

            # concatenate all windows for this audio
            audio_emb = torch.cat(frames, dim=0)  # [T_total, D]
            audio_mask = torch.cat(frame_masks, dim=0)  # [T_total]
            all_embeddings.append(audio_emb)
            all_masks.append(audio_mask)

        # batch pad to max length
        max_len = max(e.shape[0] for e in all_embeddings)
        batch_emb = torch.stack([torch.nn.functional.pad(e, (0, 0, 0, max_len - e.shape[0])) for e in all_embeddings])
        batch_mask = torch.stack([torch.nn.functional.pad(m, (0, max_len - m.shape[0])) for m in all_masks])

        return batch_emb, batch_mask



        # # -----------------------------
        # # Preprocess audio
        # # -----------------------------
        # waves = [preprocess_audio(f, sample_rate=self.sample_rate) for f in audio_inputs]
        # lengths = [len(w) for w in waves]
        # max_len = max(lengths)

        # # Pad sequences to the same length
        # padded = [np.pad(w, (0, max_len - len(w))) for w in waves]
        # batch = np.stack(padded)  # [B, max_len]
        # mask = torch.zeros(len(waves), max_len, dtype=torch.bool, device=self.device)
        # for i, l in enumerate(lengths):
        #     mask[i, :l] = 1

        # # -----------------------------
        # # Feature extraction
        # # -----------------------------
        # input_dict = self.feature_extractor(batch, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        # inputs = input_dict.input_values if "whisper" not in self.model.lower() else input_dict.input_features
        # inputs = inputs.pin_memory().to(self.device, non_blocking=True)

        # # Optionally cast to half precision for faster GPU inference
        # if self.half_precision:
        #     inputs = inputs.half()

        # # -----------------------------
        # # Forward pass
        # # -----------------------------
        # with torch.inference_mode():
        #     out = self.embedder(inputs).last_hidden_state  # [B, T, D]

        # # -----------------------------
        # # Optional L2 normalization
        # # -----------------------------
        # if self.l2_norm:
        #     out = torch.nn.functional.normalize(out, dim=-1)

        # # Optionally cast back to float32 if needed
        # if self.half_precision:
        #     out = out.float()

        # return out, mask




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract audio embeddings from file or array.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="utter-project/mHuBERT-147", help="Path or HuggingFace model name (i.e. openai/whisper-small, utter-project/mhubert-147, facebook/wav2vec2-xlsr-53 models)")
    parser.add_argument("--wav", type=str, help="Path to WAV/MP3 file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_embedder = AudioEmbedder(model=args.model, device=args.device)
    embeddings, masks = audio_embedder([args.wav])
    print(f"embeddings {embeddings.shape}")
