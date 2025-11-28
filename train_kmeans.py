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

def audio2embeddings(embedder, data_path: str, max_audio_files: int = None, max_frames_file: int = None,  max_frames_total: int = None):
    # ---------- Find audio files ----------
    audio_files = list_audio_files(data_path)
    logging.info(f"Found {len(audio_files)} audio files.")
    if not audio_files:
        raise RuntimeError("No audio files found!")
    random.shuffle(audio_files)

    logging.info(f"Shuffle {len(audio_files)} audio files.")
    audio_files = audio_files[: max_audio_files] if max_audio_files is not None else audio_files
    logging.info(f"Max audio files is {len(audio_files)}")
    logging.info(f"Max embeddings is {max_frames_total}")

    D = embedder.D
    # ---------- Extract embeddings ----------

    file_bar = tqdm(total=len(audio_files), desc="Files", position=0)
    emb_bar = tqdm(total=max_frames_total, desc="Embeddings", position=1)

    chunk_size = 256_000
    X = np.empty((chunk_size, D), dtype=np.float32) # Pre-allocate one chunk in the array
    ptr = 0  # pointer to next empty row

    for i, path in enumerate(audio_files):
        try:
            emb = embedder(path)  # Tensor [T, D]
            logging.debug(
                f"  {i+1}/{len(audio_files)} {os.path.basename(path)}:"
                f" {emb.shape[0]} frames, dim {emb.shape[1]}"
            )

            emb = emb.cpu().numpy()  # [T, D]

            #subsample
            if max_frames_file is not None and emb.shape[0] > max_frames_file:
                idx = np.random.choice(emb.shape[0], max_frames_file, replace=False)
                emb = emb[idx]

            n = emb.shape[0]

            # Resize X with another chunk if needed
            while ptr + n > X.shape[0]:
                X_new = np.empty((X.shape[0]+chunk_size, D), dtype=np.float32)
                X_new[:ptr, :] = X[:ptr, :]
                X = X_new

            # Copy embeddings into X
            X[ptr:ptr+n, :] = emb
            ptr += n

            # Update progress bars
            file_bar.update(1)
            emb_bar.update(emb.shape[0])

            ### enough samples
            if X.shape[0] >= max_frames_total:
                break

        except Exception as e:
            logging.error(f"ERROR with {path}: {e}")

    X = X[:ptr, :]

    # ---------- Stack ----------
    # logging.info("Stacking embeddings...")
    # X = np.concatenate(all_embeddings, axis=0) # [N_i, D]
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


def train_kmeans(embeddings: np.ndarray, k: int, device='cpu'):
    """Train FAISS K-means on large embedding matrix."""
    logging.info(f"Training faiss kmeans: k={k}, embeddings.shape={embeddings.shape}")

    embeddings = embeddings.astype(np.float32)
    n, d = embeddings.shape # num of embeddings, embedding dimension
    use_gpu = (device == 'cuda')
    niter = estimate_niter(n, d, k)
    logging.info(f"KMeans niter={niter} (estimated)")

    # --------- Clustering parameters ---------
    cp = faiss.ClusteringParameters()
    cp.niter = niter
    cp.nredo = 3
    cp.min_points_per_centroid = 5
    #cp.train_size = 0  # 0:uses all, >0:uses this many, not set:automatic sampling (recommended)

    # # ---------- Create the KMeans object ------
    # kmeans = faiss.Kmeans(d=d, k=k, cp=cp, verbose=True, seed=1234, gpu=use_gpu, index_factory_string="Flat")
    # # ---------- Train ----------
    # kmeans.train(embeddings)

    # Build the distance index
    index = faiss.IndexFlatL2(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    # Create clustering object
    kmeans = faiss.Clustering(d, k, cp)
    # Train KMeans
    kmeans.train(embeddings, index)
    logging.info(f"KMeans training finished. centroids = {descr(kmeans.centroids)}")  # numpy array [k, d]

    return faiss.vector_to_array(kmeans.centroids).reshape(k, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or HuggingFace model name (utter-project/mhubert-147, openai/whisper-base, wav2vec2-XLSR, etc.)")
    parser.add_argument("--data", type=str, required=True, help="Audio file or directory of audio files.")
    parser.add_argument("--k", type=int, default=500, help="Number of centroids to compute.")
    parser.add_argument("--top_db", type=int, default=30, help="Threshold (db) to remove silence (set 0 to avoid removing silence OR when whisper)")
    parser.add_argument("--stride", type=int, default=320, help="CNN stride used, necessary to pad audio (set 0 to avoid padding OR when whisper)")
    parser.add_argument("--rf", type=int, default=400, help="CNN receptive field used, necessary to pad audio")
    parser.add_argument("--max-audio-files", type=int, default=None, help="Max number of audio files to process (random subsampling).")
    parser.add_argument("--max-frames-file", type=int, default=None, help="Max number of frames to use per audio file (random subsampling).")
    parser.add_argument("--max-frames-total", type=int, default=None, help="Max number of frames to use OR max(256*K, 1M) (random subsampling).")
    parser.add_argument("--output", type=str, default="centroids", help="Output file for centroids (OUTPUT.MODEL.K.{kmeans_faiss.index,centroids.npy} is created).")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()
    args.device="cuda" if args.device == 'cuda' and torch.cuda.is_available() else "cpu"
    if args.max_frames_total is None:
        args.max_frames_total = max_train_samples=max(256 * args.k, 100000)
    
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_processor = AudioProcessor(top_db=args.top_db, stride=args.stride, receptive_field=args.rf)
    audio_embedder = AudioEmbedder(audio_processor, model=args.model, device=args.device)
    embeddings = audio2embeddings(audio_embedder, args.data, args.max_audio_files, args.max_frames_file, args.max_frames_total) #[N, D]    
    centroids = train_kmeans(embeddings, k=args.k, device=args.device) # [k, D]

    args.output = f"{args.output}.{os.path.basename(args.model)}.k{args.k}.n_iter{args.n_iter}"

    # save centroids
    np.save(f"{args.output}.centroids.npy", centroids)
    # Create FAISS search index for inference
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    faiss.write_index(index, f"{args.output}.kmeans_faiss.index")
