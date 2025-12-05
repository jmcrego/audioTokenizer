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

logger = logging.getLogger("audio_tokenizer")

def arguments(args):
    args.pop('self', None)  # None prevents KeyError if 'self' doesn't exist
    if 'audio_embedder' in args:
        args['audio_embedder'] = args['audio_embedder'].meta
    return args


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
            index = faiss.read_index(centroid_file)
            if self.device.type == "cuda":
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            self.faiss_index = index
        else:
            self.centroids = np.load(centroid_file)
            self.centroids = torch.tensor(self.centroids, dtype=torch.float32).to(self.device)


    def __call__(self, audio_input: Union[str, np.ndarray]):
        """
        Args:
            - audio filepath (.wav, .mp3) or waveform (numpy array)
            - numpy array audio chunk
        Returns:
            token_ids: numpy array [T]
        """
        embeddings = self.audio_embedder(audio_input)  # [T, D]
        logger.debug(f"embeddings {embeddings.shape} type={embeddings.__class__.__name__} dtype={embeddings.dtype}")

        # nearest centroid → token IDs
        if self.use_faiss:
            # --- FIX: move to CPU and convert to numpy ---
            if torch.is_tensor(embeddings):
                embeddings = embeddings.detach().cpu().numpy()
            # Ensure dtype + contiguity for FAISS
            embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
            _, tokens = self.faiss_index.search(embeddings,1)   # tokens = [T, 1] (nearest centroid for new embedding)
            tokens = tokens.squeeze() # numpy array [T]
        else:
            tokens = torch.argmin(torch.cdist(embeddings, self.centroids), dim=1)
            tokens = tokens.cpu().numpy() # numpy array [T]

        logger.debug(f"tokens {tokens.shape} type={tokens.__class__.__name__} dtype={tokens.dtype}")
        return tokens


if __name__ == "__main__":
    import argparse
    from AudioEmbedder import AudioEmbedder
    from AudioProcessor import AudioProcessor

    parser = argparse.ArgumentParser(description="Tokenize audio using pretrained centroids.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="utter-project/mhubert-147")
    parser.add_argument("--centroids", type=str, default="centroids/centroids.mhubert-147.100.npy")
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_embedder = AudioEmbedder(model=args.model, top_db=0, device=args.device)
    audio_tokenizer = AudioTokenizer(audio_embedder, args.centroids)

    tokens = audio_tokenizer(args.wav)
    print(tokens)
