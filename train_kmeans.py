#!/usr/bin/env python3
"""
train_kmeans.py
Train clustering centroids (e.g., K-means) on audio embeddings extracted
with AudioEmbeddings (mHuBERT / wav2vec2 XLSR / Whisper encoder).

Usage:
    python train_kmeans.py --model utter-project/mhubert-147 --data path/to/audio_dir --k 200
"""

import os
import faiss
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

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

def audio2embeddings(embedder, 
                     data_path: str,
                     max_audio_files: int = None,
                     max_frames_file: int = None,
                     max_frames_total: int = None,
                     chunk_size: int = 256_000):
    """
    Convert a directory of audio files into embeddings.
    Returns a single numpy array [N_total, D].
    """
    # ---------- Find audio files ----------
    audio_files = list_audio_files(data_path)
    logging.info(f"Found {len(audio_files)} audio files.")
    if not audio_files:
        raise RuntimeError("No audio files found!")

    random.shuffle(audio_files)
    audio_files = audio_files[:max_audio_files] if max_audio_files is not None else audio_files
    logging.info(f"Processing {len(audio_files)} audio files (max frames total = {max_frames_total}, max frames file = {max_frames_file})")

    D = embedder.D

    # ---------- Setup progress bars ----------
    f_bar = tqdm(total=len(audio_files), desc="Files", unit=" file", position=0, leave=True)
    e_bar = tqdm(total=max_frames_total, desc="Embed", unit=" embs", position=1, leave=True)

    # ---------- Dynamic chunk allocation ----------
    X = np.empty((chunk_size, D), dtype=np.float32)
    ptr = 0  # pointer to next empty row

    # ---------- Process files ----------
    for i, path in enumerate(audio_files):
        try:
            emb = embedder(path)  # Tensor [T, D]
            emb = emb.cpu().numpy()

            # Per-file subsampling
            if max_frames_file is not None and emb.shape[0] > max_frames_file:
                idx = np.random.choice(emb.shape[0], max_frames_file, replace=False)
                emb = emb[idx]

            n = emb.shape[0]

            # Determine how many frames can be added
            if max_frames_total is not None:
                frames_to_add = min(n, max_frames_total - ptr)
                if frames_to_add <= 0:
                    break
                emb = emb[:frames_to_add]
            else:
                frames_to_add = n

            # Resize X if needed
            while ptr + frames_to_add > X.shape[0]:
                new_size = X.shape[0] + chunk_size
                X_new = np.empty((new_size, D), dtype=np.float32)
                X_new[:ptr, :] = X[:ptr, :]
                X = X_new

            # Copy embeddings
            X[ptr:ptr + frames_to_add, :] = emb
            ptr += frames_to_add

            # Update progress bars
            f_bar.update(1)
            e_bar.update(frames_to_add)

        except Exception as e:
            logging.error(f"ERROR with {path}: {e}")

    # ---------- Trim X to actual size ----------
    X = X[:ptr, :]

    # ---------- Optional global subsampling ----------
    if max_frames_total is not None and len(X) > max_frames_total:
        idx = np.random.choice(len(X), max_frames_total, replace=False)
        X = X[idx]
        logging.info(f"Subsampled to {len(X)} frames (global limit).")

    # ---------- Log total time / frames ----------
    sample_rate = 16000
    stride = 320
    total_seconds = len(X) * stride / sample_rate
    logging.info(f"\n\nTotal frames: {X.shape}, approximate time: {secs2human(total_seconds)}")

    return X  # [N_total, D]


def train_kmeans(embeddings: np.ndarray, k: int, device='cpu'):
    """Train FAISS K-means on large embedding matrix."""
    logging.info(f"Training faiss kmeans: k={k}, embeddings.shape={embeddings.shape}")
    if embeddings.shape[0] == 0:
        raise RuntimeError("No embeddings were extracted!")

    embeddings = embeddings.astype(np.float32)
    np.random.shuffle(embeddings)
    n, d = embeddings.shape # num of embeddings, embedding dimension
    use_gpu = (device == 'cuda')
    niter = estimate_niter(n, d, k)
    logging.info(f"KMeans niter={niter} (estimated)")

    # --------- Clustering parameters ---------
    cp = faiss.ClusteringParameters()
    cp.niter = niter
    cp.nredo = 3
    cp.min_points_per_centroid = 5
    cp.train_size = 0  # 0:uses all, >0:uses this many, not set:automatic sampling (recommended)

    # Build the distance index
    index = faiss.IndexFlatL2(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    # Create clustering object
    kmeans = faiss.Clustering(d, k, cp)
    # Train KMeans
    kmeans.train(embeddings, index)

    # Convert flat FAISS vector to NumPy array
    centroids = faiss.vector_to_array(kmeans.centroids).reshape(k, d)
    logging.info(f"KMeans training finished. centroids = {descr(centroids)}")  # numpy array [k, d]

    return centroids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or HuggingFace model name (utter-project/mhubert-147, openai/whisper-base, wav2vec2-XLSR, etc.)")
    parser.add_argument("--data", type=str, required=True, help="Audio file or directory of audio files.")
    parser.add_argument("--k", type=int, default=500, help="Number of centroids to compute.")
    parser.add_argument("--top_db", type=int, default=30, help="Threshold (db) to remove silence (set 0 to avoid removing silence OR when whisper)")
    parser.add_argument("--stride", type=int, default=320, help="CNN stride used, necessary to pad audio (set 0 to avoid padding OR when whisper)")
    parser.add_argument("--rf", type=int, default=400, help="CNN receptive field used, necessary to pad audio")
    parser.add_argument("--max-audio-files", type=int, default=None, help="Max number of audio files to process (random subsampling).")
    parser.add_argument("--max-frames-file", type=int, default=50, help="Max number of frames to use per audio file (random subsampling).")
    parser.add_argument("--max-frames-total", type=int, default=None, help="Max number of frames to use OR max(256*K, 1M) (random subsampling).")
    parser.add_argument("--output", type=str, default="centroids", help="Output file for centroids (OUTPUT.MODEL.K.{kmeans_faiss.index,centroids.npy} is created).")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()
    args.device="cuda" if args.device == 'cuda' and torch.cuda.is_available() else "cpu"
    if args.max_frames_total is None:
        args.max_frames_total = max(256 * args.k, 1000000)
    
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_processor = AudioProcessor(top_db=args.top_db, stride=args.stride, receptive_field=args.rf)
    audio_embedder = AudioEmbedder(audio_processor, model=args.model, device=args.device)

    embeddings = audio2embeddings(audio_embedder, 
                                  args.data, 
                                  max_audio_files=args.max_audio_files, 
                                  max_frames_file=args.max_frames_file, 
                                  max_frames_total=args.max_frames_total) #[N, D]    

    centroids = train_kmeans(embeddings, 
                             k=args.k, 
                             device=args.device) # [k, D]

    args.output = f"{args.output}.{os.path.basename(args.model)}.k{args.k}"
    # save centroids
    np.save(f"{args.output}.centroids.npy", centroids)
    # Create FAISS search index for inference
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    faiss.write_index(index, f"{args.output}.kmeans_faiss.index")
