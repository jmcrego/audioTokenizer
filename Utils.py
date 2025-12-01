import os
import soxr
import logging
import numpy as np
import soundfile as sf

SUPPORTED_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

def load_wav(file_path: str, channel: int = 0, sample_rate: int = 16000) -> np.ndarray:
    wav, sr = sf.read(file_path)
    logging.info(f"file wav size is {wav.shape} sr={sr}")
    # stereo (multi-channel)
    if len(wav.shape) > 1: 
        if channel < -1 or channel >= wav.shape[1]:
            raise ValueError(f"Invalid channel {channel} for audio with {wav.shape[1]} channels")
        elif wav.shape[1] > 1:
            # select channel or average channels
            if channel == 0:
                wav = wav[:, 0]
            elif channel == 1:
                wav = wav[:, 1]
            else:
                wav = np.mean(wav, axis=1)
        logging.info(f"handled channels, wav size is {wav.shape}")
    # resample if needed 
    if sr != sample_rate:
        wav = soxr.resample(wav, sr, sample_rate)
        logging.info(f"resampled wav wav size is {wav.shape} sr={sr}")
    # Ensure float32 dtype
    wav = wav.astype(np.float32)

    return wav


def secs2human(t):
    sec = int(t)
    ms = int((t - sec) * 1000)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}.{ms:03d}"


def list_audio_files(path: str):
    """Return list of audio files from a file or directory."""
    files = []
    if os.path.isfile(path):
        #return [path]
        with open(path, 'r') as fd:
            for l in fd:
                parts = l.strip().split('\t')
                if len(parts) == 3:
                    files.append(parts[2])
    else:
        for root, _, fs in os.walk(path):
            for f in fs:
                if f.lower().endswith(SUPPORTED_EXT):
                    files.append(os.path.join(root, f))
    return sorted(files)

def arguments(args):
    args.pop('self', None)  # None prevents KeyError if 'self' doesn't exist
    args.pop('audio_processor', None)
    args.pop('audio_embedder', None)
    args.pop('audio_tokenizer', None)
    return args


def descr(var):
    myshape = var.shape if hasattr(var, 'shape') else "-"
    mydtype = var.dtype if hasattr(var, 'dtype') else "-"
    mytype  = var.__class__.__name__
    return f"shape={myshape} type={mytype} dtype={mydtype}"

