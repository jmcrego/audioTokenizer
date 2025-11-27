#!/usr/bin/env python3
"""
train_kmeans.py
Train clustering centroids (e.g., K-means) on audio embeddings extracted
with AudioEmbeddings (mHuBERT / wav2vec2 XLSR / Whisper encoder).

Usage:
    python train_kmeans.py --model utter-project/mhubert-147 --data path/to/audio_dir --k 200
"""

import os
import torch
import random
import logging
import argparse
import numpy as np
import faiss

from AudioEmbedder import AudioEmbedder
from AudioProcessor import AudioProcessor
from Utils import list_audio_files, secs2human


def audio2embeddings(embedder, data_path: str, max_audio_files: int = None, max_frames_file: int = None,  max_frames_total: int = None, device: str = 'cpu'):
    # ---------- Find audio files ----------
    audio_files = list_audio_files(data_path)
    if not audio_files:
        raise RuntimeError("No audio files found!")
    random.shuffle(audio_files)
    audio_files = audio_files[: max_audio_files] if max_audio_files is not None else audio_files
    logging.info(f"Found {len(audio_files)} audio files.")

    # ---------- Extract embeddings ----------
    all_embeddings = []
    for i, path in enumerate(audio_files):
        try:
            emb = embedder(path) # Tensor [T, D]
            logging.info(f"  {i+1}/{len(audio_files)} {os.path.basename(path)}: {emb.shape[0]} frames, dim {emb.shape[1]}")
            emb = emb.cpu().numpy() #[T, D]

            #shuffle frames if needed
            if max_frames_file is not None and emb.shape[0] > max_frames_file:
                idx = np.random.choice(emb.shape[0], max_frames_file, replace=False)
                emb = emb[idx]
            all_embeddings.append(emb) #[N_i, D]

        except Exception as e:
            logging.error(f"ERROR with {path}: {e}")

    # ---------- Stack ----------
    logging.info("Stacking embeddings...")
    X = np.concatenate(all_embeddings, axis=0) # [N_i, D]
    sample_rate = 16000
    stride = 320
    logging.info(f"Total frames: {len(X)}, dim: {X.shape[1]}, time: {secs2human(len(X) * stride / sample_rate)}")

    # ---------- keep args.max_frames from all_frames ----------
    if max_frames_total is not None and len(X) > max_frames_total:
        idx = np.random.choice(len(X), max_frames_total, replace=False)
        X = X[idx] #[<=max_frames, D]
        logging.info(f"Subsampled to {len(X)} frames.")

    #shuffle globally
    np.random.shuffle(X) 

    return X  #[N, D]


def train_kmeans(embeddings: np.ndarray, k: int, n_iter=1000, batch_size=16384, device='cpu'):
    """Train FAISS K-means on large embedding matrix."""
    logging.info(f"Training faiss kmeans: k={k}, dim={embeddings.shape[1]}")

    d = embeddings.shape[1]  # embedding dimension
    use_gpu = (device == 'cuda')

    # ---------- GPU or CPU resources ----------
    if use_gpu:
        res = faiss.StandardGpuResources()
    else:
        res = None

    # ---------- Create the KMeans object ------
    kmeans = faiss.Kmeans(
        d=d,
        k=k,
        niter=n_iter,
        verbose=True,
        gpu=use_gpu,
    )

    # ---------- Train ----------
    kmeans.train(embeddings.astype(np.float32))
    logging.info(f"KMeans training finished. centroids = {kmeans.centroids.shape}")  # numpy array [k, d]

    return kmeans


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or HuggingFace model name (utter-project/mhubert-147, openai/whisper-base, wav2vec2-XLSR, etc.)")
    parser.add_argument("--data", type=str, required=True, help="Audio file or directory of audio files.")
    parser.add_argument("--k", type=int, default=500, help="Number of centroids.")
    parser.add_argument("--n-iter", type=int, default=1000, help="Kmeans number of iterations.")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size.")
    parser.add_argument("--max-audio-files", type=int, default=None, help="Max number of audio files to process (random subsampling).")
    parser.add_argument("--max-frames-file", type=int, default=None, help="Max number of frames to use per audio file (random subsampling).")
    parser.add_argument("--max-frames-total", type=int, default=None, help="Max number of frames to use (random subsampling).")
    parser.add_argument("--output", type=str, default="centroids", help="Output file for centroids (OUTPUT.MODEL.K.{kmeans_faiss.index,centroids.npy} is created).")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()
    args.device="cuda" if args.device == 'cuda' and torch.cuda.is_available() else "cpu"

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_processor = AudioProcessor(top_db=30, stride=320, receptive_field=400)
    audio_embedder = AudioEmbedder(audio_processor, model=args.model, device=args.device)
    embeddings = audio2embeddings(audio_embedder, args.data, args.max_audio_files, args.max_frames_file, args.max_frames_total, device=args.device) #[N, D]    
    kmeans = train_kmeans(embeddings, k=args.k, n_iter=args.n_iter, batch_size=args.batch_size, device=args.device) # [k, D]

    args.output = f"{args.output}.{os.path.basename(args.model)}.k{args.k}.n_iter{args.n_iter}.bs{args.batch_size}"
    faiss.write_index(kmeans.index, f"{args.output}.kmeans_faiss.index")
    # save separately centroids lets you load them without FAISS
    np.save(f"{args.output}.centroids.npy", kmeans.centroids)

    #when reading, use:

    # index = faiss.read_index("kmeans.index")
    # D, I = index.search(x, 1)   # nearest centroid for new embedding

    # or only centroids:
    # centroids = np.load("centroids.npy")
