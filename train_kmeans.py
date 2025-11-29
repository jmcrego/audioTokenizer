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
    e_bar = tqdm(total=max_frames_total, desc="Embed", unit=" embs", position=0, leave=True)
    f_bar = tqdm(total=len(audio_files), desc="Files", unit=" file", position=1, leave=True)

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

def audio2embeddings_memmap(embedder,
                 data_path: str,
                 memmap_path: str,
                 max_audio_files: int = None,
                 max_frames_file: int = None,
                 max_frames_total: int = None,
                 chunk_size: int = 256_000):
    """
    Convert audio files to embeddings stored in a numpy memmap on disk.

    Returns:
        memmap_path (str)  : path to memmap file (same as arg)
        n_written (int)    : number of embeddings actually written (<= max_frames_total)
        D (int)            : embedding dimension
    NOTE: max_frames_total must be provided (memmap needs a fixed shape).
    """
    if max_frames_total is None:
        raise ValueError("max_frames_total must be set when using memmap.")

    audio_files = list_audio_files(data_path)
    logging.info(f"Found {len(audio_files)} audio files.")
    if not audio_files:
        raise RuntimeError("No audio files found!")

    random.shuffle(audio_files)
    audio_files = audio_files[:max_audio_files] if max_audio_files is not None else audio_files
    logging.info(f"Processing {len(audio_files)} files (memmap={memmap_path}, max_frames_total={max_frames_total})")

    D = embedder.D
    # create memmap file (mode 'w+' creates or overwrites)
    X = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(max_frames_total, D))
    ptr = 0

    # progress bars
    f_bar = tqdm(total=len(audio_files), desc="Files", unit="file", position=0, leave=True)
    e_bar = tqdm(total=max_frames_total, desc="Embeds", unit="emb", position=1, leave=True)

    for i, path in enumerate(audio_files):
        try:
            emb = embedder(path)  # Tensor [T, D]
            emb = emb.cpu().numpy()

            # per-file subsample
            if max_frames_file is not None and emb.shape[0] > max_frames_file:
                idx = np.random.choice(emb.shape[0], max_frames_file, replace=False)
                emb = emb[idx]

            n = emb.shape[0]
            if n == 0:
                f_bar.update(1)
                continue

            # how many can we store
            can_add = max_frames_total - ptr
            if can_add <= 0:
                # reached capacity
                break

            add = n if n <= can_add else can_add
            if add < n:
                emb = emb[:add]

            # write into memmap
            X[ptr:ptr+add, :] = emb
            ptr += add

            # update bars
            f_bar.update(1)
            e_bar.update(add)

            # stop if full
            if ptr >= max_frames_total:
                break

        except Exception as e:
            logging.error(f"ERROR with {path}: {e}")

    # flush to disk
    X.flush()
    f_bar.n = f_bar.total; f_bar.refresh(); f_bar.close()
    e_bar.n = e_bar.total if ptr >= max_frames_total else ptr; e_bar.refresh(); e_bar.close()

    meta = {"n_vectors": ptr, "dim": D}
    with open(memmap_path + ".json", "w") as f:
        json.dump(meta, f)

    logging.info(f"Finished writing memmap: {memmap_path}, written_frames={ptr}, dim={D}")
    return memmap_path, ptr, D

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
    cp.train_size = embeddings.shape[0]  # 0:uses all, >0:uses this many, not set:automatic sampling (recommended)

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

def train_kmeans_memmap(memmap_path: str,
                        n_vectors: int,
                        d: int,
                        k: int,
                        device='cpu',
                        sample_size=None):
    """
    Train FAISS K-means from a memmap file containing float32 embeddings [N, d].

    Args:
        memmap_path: path to memmap file created by audio2memmap()
        n_vectors: number of valid vectors in the memmap (ptr returned by audio2memmap)
        d: embedding dimension
        k: number of centroids
        device: 'cpu' or 'cuda'
        sample_size: how many vectors to sample for training
                     If None: uses FAISS best practice = min(N, max(256*k, 100k))
    Returns:
        centroids: numpy array [k, d] float32
    """
    logging.info(f"Training FAISS KMeans from memmap: k={k}, n_vectors={n_vectors}, d={d}")

    if n_vectors == 0:
        raise RuntimeError("No embeddings found in memmap!")

    # -------------------------------
    # Suggested default sample size
    # -------------------------------
    if sample_size is None:
        sample_size = n_vectors #min(n_vectors, max(256 * k, 100_000))

    logging.info(f"KMeans sample_size = {sample_size}")

    # -------------------------------
    # Estimate niter (same as your version)
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
    cp.train_size = sample_size   # use the full sampled subset, and let FAISS handle iteration-level sampling

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

    return centroids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build embeddings from audi files and compute kmeans centroids", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- embedding options ---
    embedding_group = parser.add_argument_group("embedding options")
    embedding_group.add_argument("--model", type=str, required=True, help="Path or HuggingFace model name.")
    embedding_group.add_argument("--data", type=str, required=True, help="File containing audio files to consider.")
    embedding_group.add_argument("--top_db", type=int, default=30, help="Threshold (db) to remove silence.")
    embedding_group.add_argument("--stride", type=int, default=320, help="Processor CNN stride.")
    embedding_group.add_argument("--rf", type=int, default=400, help="Processor CNN receptive field.")
    embedding_group.add_argument("--max-audio-files", type=int, default=None, help="Max number of audio files.")
    embedding_group.add_argument("--max-frames-file", type=int, default=None, help="Max number of frames per file.")
    embedding_group.add_argument("--max-frames-total", type=int, required=True, help="Total max frames.")
    embedding_group.add_argument("--memmap", action="store_true", help="Use memmap to reduce RAM usage.")

    # --- centroid options ---
    centroid_group = parser.add_argument_group("centroid options")
    centroid_group.add_argument("--k", type=int, default=500, help="Number of centroids.")

    # --- common options ---
    common_group = parser.add_argument_group("common options")
    common_group.add_argument("--output", type=str, default="centroids", help="Output file prefix.")
    common_group.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda').")

    args = parser.parse_args()

    args.device="cuda" if args.device == 'cuda' and torch.cuda.is_available() else "cpu"
    if args.max_frames_total is None:
        args.max_frames_total = max(256 * args.k, 1000000)
    
    args.output = f"{args.output}.{os.path.basename(args.model)}.k{args.k}"
    ofile1 = f"{args.output}.centroids.npy"
    ofile2 = f"{args.output}.kmeans_faiss.index"
    lfile  = f"{args.output}.log"
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(),logging.FileHandler(lfile)])

    audio_processor = AudioProcessor(top_db=args.top_db, stride=args.stride, receptive_field=args.rf)
    audio_embedder = AudioEmbedder(audio_processor, model=args.model, device=args.device)

    ### Build embeddings
    ##############################################
    if args.memmap:
        memmap_path = args.data + ".memmap"
        if not os.path.exists(memmap_path):
            memmap_path, n_written, D = audio2embeddings_memmap(
                embedder=audio_embedder,
                data_path=args.data,
                memmap_path=memmap_path,
                max_audio_files=args.max_audio_files,
                max_frames_file=args.max_frames_file,
                max_frames_total=args.max_frames_total)
        else:
            with open(memmap_path + ".json") as f:
                meta = json.load(f)
            n_written = meta['n_vectors']
            D = meta['D']
            logging.info(f"Skipping memmap creation, existing file {memmap_path} with n_written={n_written} D={D}")

        centroids = train_kmeans_memmap(
            memmap_path,
            n_vectors=n_written,
            d=D,
            k=args.k,
            device=args.device,
            sample_size=n_written)

    else:
        embeddings = audio2embeddings(
            audio_embedder, 
            args.data, 
            max_audio_files=args.max_audio_files, 
            max_frames_file=args.max_frames_file, 
            max_frames_total=args.max_frames_total) #[N, D]    

        centroids = train_kmeans(
            embeddings, 
            k=args.k, 
            device=args.device) # [k, D]

    # save centroids
    np.save(ofile1, centroids)
    # Create FAISS search index for inference
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    faiss.write_index(index, ofile2)
