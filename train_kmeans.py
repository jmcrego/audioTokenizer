#!/usr/bin/env python3
"""
train_kmeans.py
Train clustering centroids (e.g., K-means) on audio embeddings extracted
with AudioEmbeddings (mHuBERT / wav2vec2 XLSR / Whisper encoder).

Usage:
    python train_kmeans.py --model utter-project/mhubert-147 --data path/to/audio_dir --k 200
"""

import os
import json
import faiss
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import tempfile

from AudioEmbedder import AudioEmbedder
from AudioProcessor import AudioProcessor
from Utils import list_audio_files, secs2human, descr

def estimate_niter(N, D, K):
    """
    Estimate a good niter for FAISS KMeans based on dataset size, dimension, and number of clusters.
    """

    # Base on number of clusters
    base = 20 + 10 * np.log10(K)

    # High-dimensional smoothing
    if D >= 512:
        base += 10   # small bump

    # Large dataset correction
    if N >= 2_000_000:
        base += 10
    elif N >= 10_000_000:
        base += 20

    # Boundaries
    niter = int(np.clip(base, 30, 150))
    return niter

def train_kmeans_memmap(memmap_path: str,
                        n_vectors: int,
                        d: int,
                        k: int,
                        device='cpu',
                        sample_size=None,
                        output=None):
    """
    Train FAISS K-means from a memmap file containing float32 embeddings [N, d].

    Args:
        memmap_path: path to memmap file created by audio2memmap()
        n_vectors: number of valid vectors in the memmap (ptr returned by audio2memmap)
        d: embedding dimension
        k: number of centroids
        device: 'cpu' or 'cuda'
        sample_size: how many vectors to sample for training (if None use n_vectors)
    """
    logging.info(f"Training FAISS KMeans from memmap: k={k}, n_vectors={n_vectors}, d={d}")

    if n_vectors == 0:
        raise RuntimeError("No embeddings found in memmap!")

    # -------------------------------
    # Suggested default sample size
    # -------------------------------
    if sample_size is None:
        sample_size = n_vectors 

    logging.info(f"KMeans training size = {sample_size}")

    # -------------------------------
    # Estimate niter 
    # -------------------------------
    niter = estimate_niter(sample_size, d, k)
    logging.info(f"KMeans niter = {niter} (estimated)")

    # -------------------------------
    # Open memmap for reading
    # -------------------------------
    Xmm = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=(n_vectors, d))

    # -------------------------------
    # Sample indices
    # -------------------------------
    if sample_size < n_vectors:
        idx = np.random.choice(n_vectors, sample_size, replace=False)
        X = np.asarray(Xmm[idx], dtype=np.float32, copy=True)
    else:
        X = np.asarray(Xmm, dtype=np.float32, copy=True)

    # shuffle for good measure (same as your function)
    np.random.shuffle(X)

    # -------------------------------
    # Clustering parameters
    # -------------------------------
    cp = faiss.ClusteringParameters()
    cp.niter = niter
    cp.nredo = 3
    cp.min_points_per_centroid = 5
    cp.train_size = sample_size

    # -------------------------------
    # Construct FAISS index (CPU or GPU)
    # -------------------------------
    if device == 'cuda':
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
    else:
        index = faiss.IndexFlatL2(d)

    # -------------------------------
    # KMeans training
    # -------------------------------
    kmeans = faiss.Clustering(d, k, cp)
    logging.info("Running FAISS kmeans.train()...")
    kmeans.train(X, index)

    # -------------------------------
    # Extract centroids
    # -------------------------------
    centroids = faiss.vector_to_array(kmeans.centroids).reshape(k, d).astype(np.float32)
    logging.info(f"KMeans finished. Centroids shape = {centroids.shape}")

    ofile1 = f"{output}.centroids.npy"
    ofile2 = f"{output}.kmeans_faiss.index"

    # save centroids
    np.save(ofile1, centroids)
    # Create FAISS search index for inference
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    faiss.write_index(index, ofile2)


    return centroids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings from audio files and compute centroids using FAISS KMeans.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("memdata", type=str, required=True, help="File with embeddings to cluster (memdata format).")
    parser.add_argument("--k", type=int, default=500, help="Number of centroids.")
    parser.add_argument("--sample_size", type=int, default=None, help="Train with this number of embeddings (random sampling).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda').")

    args = parser.parse_args()

    args.device="cuda" if args.device == 'cuda' and torch.cuda.is_available() else "cpu"
    
    args.output = f"{args.memdata}.k{args.k}"
    lfile  = f"{args.output}.log"
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(),logging.FileHandler(lfile)])

    with open(f"{args.mempath}.json") as f:
        meta = json.load(f)

    centroids = train_kmeans_memmap(
        args.memmap,
        n_vectors=meta['n_vectors'],
        d=meta['D'],
        k=args.k,
        device=args.device,
        sample_size=args.sample_size
        output=args.output)


