#!/usr/bin/env python3
"""
Convert audio → embeddings → discrete tokens using pretrained centroids.

The tokenizer supports:
- Loading centroids from .npy, .index (faiss).
- Tokenizing audio files or numpy waveforms.
- Accepting precomputed embeddings directly.
Usage:
    tokenizer = AudioTokenizer("centroids.npy", embedder)
    tokens = tokenizer("speech.wav")
"""

import os
import faiss
import torch
import logging
import numpy as np
from typing import Union

from Utils import arguments, descr

logger = logging.getLogger("audio_tokenizer")

class AudioTokenizer:
    """
    Audio Tokenizer: converts embedding frames into discrete token IDs.
    """
    def __init__(self, audio_embedder, centroid_file: str, device: str = "cpu"):
        """
        Load pretrained centroids and initializes the audio embedder.
        Args:
            - audio_embedder
            - path to centroids file (.npy or .index)
            - computation device ("cpu" or "cuda")
        """
        self.meta = arguments(locals())
        # self.meta['audio_embedder'] = audio_embedder.meta
        logger.info(f"Initializing {self.meta}")

        # logger.info(f"Initializing {arguments(locals())}")
        self.device = torch.device(device)
        self.audio_embedder = audio_embedder
        if not os.path.exists(centroid_file):
            raise FileNotFoundError(f"Centroid file not found: {centroid_file}")
        
        self.use_faiss = centroid_file.endswith('.index')
        if self.use_faiss:
            self.faiss_index = faiss.read_index(centroid_file)
        else:
            self.centroids = np.load(centroid_file)
            self.centroids = torch.tensor(self.centroids, dtype=torch.float32).to(self.device)


    def __call__(self, audio_input: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Args:
            - audio filepath (.wav, .mp3) or waveform (numpy array)
            - numpy array audio chunk
        Returns:
            token_ids: numpy array [T]
        """
        embeddings = self.audio_embedder(audio_input)  # [T, D]
        logger.debug(f"embeddings {descr(embeddings)}")

        # ---- FIX: move to CPU + convert to NumPy ----
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        # nearest centroid → token IDs
        if self.use_faiss:
            _, tokens = self.faiss_index.search(embeddings,1)   # tokens = [T, 1] (nearest centroid for new embedding)
            tokens = tokens.squeeze() # numpy array [T]
        else:
            tokens = torch.argmin(torch.cdist(embeddings, self.centroids), dim=1)
            tokens = tokens.numpy() # numpy array [T]

        logger.debug(f"tokens {descr(tokens)}")
        return tokens


if __name__ == "__main__":
    import argparse
    from AudioEmbedder import AudioEmbedder
    from AudioProcessor import AudioProcessor

    parser = argparse.ArgumentParser(description="Tokenize audio using pretrained centroids.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="utter-project/mhubert-147")
    parser.add_argument("--centroids", type=str, default="centroids/centroids.mhubert-147.100.npy")
    parser.add_argument("--wav", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_processor = AudioProcessor(top_db=30, stride=320, receptive_field=400)
    audio_embedder = AudioEmbedder(audio_processor, model=args.model)
    audio_tokenizer = AudioTokenizer(audio_embedder, args.centroids)

    tokens = audio_tokenizer(args.wav)
    print(tokens)
