#!/usr/bin/env python3
"""
Class to Preprocess audio files.
Supports WAV/MP3 files or raw numpy audio arrays.
"""

import soxr
import torch
import logging
import numpy as np
import soundfile as sf
import librosa

from Utils import arguments

logger = logging.getLogger("audio_processor")

class AudioProcessor:

    def __init__(self, top_db: int = 30, stride: int = 320, receptive_field: int = 400, channel: int = 0):
        logger.info(f"Initializing {arguments(locals())}")
        self.sample_rate = 16000  # default for all considered models
        self.top_db = top_db # to remove silence
        self.stride = stride #to pad the audio chunk unless whisper
        self.receptive_field = receptive_field #to pad the audio chunk unless whisper
        self.channel = channel
        self.total_audio = 0
        self.total_noise = 0
        self.total_pad = 0

    def __call__(self, audio_input, sr=16000) -> torch.Tensor:
        """
        Preprocess from a WAV file path or numpy array.
        Args:
            audio_input: str path to WAV file or np.ndarray (float32)
            sr: original sr of audio chunk (not used with audio files as the value is loaded)
        Returns:
            embeddings: torch.Tensor [T, emb_dim]
        """

        if isinstance(audio_input, str):
            wav, sr = sf.read(audio_input)
        elif isinstance(audio_input, np.ndarray):
            wav = audio_input
        else:
            raise ValueError("audio_input must be a path or np.ndarray")
        logger.debug(f"wav size={wav.shape} sr={sr} time={wav.shape[0]/sr:.2f} sec")

        #  CHANNEL handling (mono)
        if len(wav.shape) > 1: 
            if self.channel < -1 or self.channel >= wav.shape[1]:
                raise ValueError(f"Invalid channel {self.channel} for audio with {wav.shape[1]} channels")
            elif wav.shape[1] > 1:
                # select channel or average channels
                if self.channel == 0:
                    wav = wav[:, 0]
                elif self.channel == 1:
                    wav = wav[:, 1]
                else:
                    wav = np.mean(wav, axis=1)
            logger.debug(f"handled channels, wav size={wav.shape} time={wav.shape[0]/sr:.2f} sec")

        # RESAMPLE
        if sr != self.sample_rate:
            wav = soxr.resample(wav, sr, self.sample_rate)
            logger.debug(f"resampled, wav size={wav.shape} sr={self.sample_rate} time={wav.shape[0]/self.sample_rate:.2f} sec")
        # Ensure float32 dtype
        wav = wav.astype(np.float32)
        self.total_audio += len(wav)

        # --- REMOVE SILENCE ---
        if self.top_db:
            wav_trimmed, _ = librosa.effects.trim(wav, top_db=self.top_db)
            logger.debug(f"removed silence, wav size={wav_trimmed.shape} time={wav.shape[0]/self.sample_rate:.2f} sec")
            self.total_noise += len(wav)-len(wav_trimmed)
            wav = wav_trimmed

        # --- PAD THE AUDIO TO MATCH THE STRIDE ---
        if self.stride: #mHuBERT / wav2vec2
            remainder = (len(wav) - self.receptive_field) % self.stride
            if remainder != 0:
                pad_len = self.stride - remainder
                wav = np.pad(wav, (0, pad_len), mode='constant') 
                logger.debug(f"padded wav by {pad_len} samples, wav size={wav.shape} time={wav.shape[0]/self.sample_rate:.2f} sec")
                self.total_pad += pad_len

        return wav 

    def stats(self):
        return {
            'total audio': self.total_audio,
            'total noise': self.total_noise,
            'total pad': self.total_pad
        }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess audio from file or array.")
    parser.add_argument("--wav", type=str, help="Path to WAV/MP3 file")
    parser.add_argument("--top_db", type=int, default=10, help="Threshold (db) to remove silence (set 0 to avoid removing silence OR when whisper)")
    parser.add_argument("--stride", type=int, default=320, help="CNN stride used, necessary to pad audio (set 0 to avoid padding OR when whisper)")
    parser.add_argument("--rf", type=int, default=400, help="CNN receptive field used, necessary to pad audio")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_processor = AudioProcessor(top_db=args.top_db, stride=args.stride, receptive_field=args.rf)
    wav = audio_processor(args.wav)
