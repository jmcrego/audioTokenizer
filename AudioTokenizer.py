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

from Utils import arguments

logger = logging.getLogger("audio_tokenizer")

class AudioTokenizer:
    """
    Audio Tokenizer: converts embedding frames into discrete token IDs.
    """
    def __init__(self, audio_embedder, centroid_file: str, l2_norm: bool=True, device: str = "cpu"):
        """
        Load pretrained centroids and initializes the audio embedder.
        Args:
            - audio_embedder
            - path to centroids file (.npy)
            - computation device ("cpu" or "cuda")
        """
        logger.info(f"Initializing {arguments(locals())}")
        self.device = torch.device(device)
        self.embedder = audio_embedder
        self.l2_norm = l2_norm
        if not os.path.exists(centroid_file):
            raise FileNotFoundError(f"Centroid file not found: {centroid_file}")
        self.centroids = np.load(centroid_file)
        self.centroids = torch.tensor(self.centroids, dtype=torch.float32).to(self.device)


    def __call__(self, audio_input: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Args:
            - audio filepath (.wav, .mp3) or waveform (numPy array)
            - numpy array audio chunk
        Returns:
            token_ids: numpy array [T]
        """
        embeddings = self.embedder(audio_input)  # [T, D]

        #L2-normalize embeddings for better clustering
        if self.l2_norm:
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8) 

        # nearest centroid → token IDs
        tokens = torch.argmin(torch.cdist(embeddings, self.centroids), dim=1)
        #For speed on large corpora or client inference, use Faiss (index with flat L2 or HNSW) to get very fast nearest-centroid lookup.

        return tokens.numpy()


if __name__ == "__main__":
    import argparse
    from AudioEmbedder import AudioEmbedder
    from AudioProcessor import AudioProcessor

    parser = argparse.ArgumentParser(description="Tokenize audio using pretrained centroids")
    parser.add_argument("--model", type=str, default="utter-project/mhubert-147")
    parser.add_argument("--centroids", type=str, default="centroids/centroids.mhubert-147.100.npy")
    parser.add_argument("--wav", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_processor = AudioProcessor(top_db=30, stride=320, receptive_field=400)
    audio_embedder = AudioEmbedder(audio_processor, model=args.model)
    audio_tokenizer = AudioTokenizer(audio_embedder, args.centroids)

    tokens = audio_tokenizer(args.wav)
    print("Token sequence length:", len(tokens))
    print("Tokens:", tokens)
