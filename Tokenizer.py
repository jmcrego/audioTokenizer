#!/usr/bin/env python3
"""
Convert audio → embeddings → discrete tokens using pretrained centroids.

The tokenizer supports:
- Loading centroids from .npy, .pt, or sklearn KMeans pickle.
- Tokenizing audio files or numpy waveforms.
- Accepting precomputed embeddings directly.

Usage:
    tokenizer = AudioTokenizer("centroids.npy", embedder)
    tokens = tokenizer("speech.wav")
"""

import os
import torch
import logging
import numpy as np
from typing import Union

from Embedder import AudioEmbedder
from Utils import load_wav

logger = logging.getLogger("record_mic_stream")

class AudioTokenizer:
    """
    Audio Tokenizer: converts embedding frames into discrete token IDs.
    """
    def __init__(self, model_name: str, centroid_file: str, device: str = "cpu"):
        """
        Load pretrained centroids and initializes the audio embedder.
        Args:
            - model name (HuggingFace) or path to embedding model (e.g., "utter-project/mhubert-147")
            - path to centroids file (.npy)
            - computation device ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.embedder = AudioEmbedder(model_name=model_name, device=device)
        if not os.path.exists(centroid_file):
            raise FileNotFoundError(f"Centroid file not found: {centroid_file}")
        self.centroids = np.load(centroid_file)
        self.centroids = torch.tensor(self.centroids, dtype=torch.float32).to(self.device)
        self.num_centroids = self.centroids.shape[0]


    def __call__(self, audio_input: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Args:
            - audio filepath (.wav, .mp3) or waveform (numPy array)
            - numpy array audio chunk
        Returns:
            token_ids: numpy array [T]
        """
        if isinstance(audio_input, np.ndarray) or isinstance(audio_input, str):
            embeddings = self.embedder(audio_input)  # [T, D]
        else:
            raise ValueError("audio_input must be a filepath or numpy array")

        # nearest centroid → token IDs
        tokens = torch.argmin(torch.cdist(embeddings, self.centroids), dim=1)
        #For speed on large corpora or client inference, use Faiss (index with flat L2 or HNSW) to get very fast nearest-centroid lookup.

        return tokens.numpy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize audio using pretrained centroids")
    parser.add_argument("--model", type=str, default="utter-project/mhubert-147")
    parser.add_argument("--centroids", type=str, default="centroids.mhubert-147.100.npy")
    parser.add_argument("--wav", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    tokenizer = AudioTokenizer(args.model, args.centroids)

    tokens = tokenizer(args.wav)
    print("Token sequence length:", len(tokens))
    print("Tokens:", tokens)
