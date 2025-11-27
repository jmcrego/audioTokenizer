#!/usr/bin/env python3
"""
train_kmeans.py
Train clustering centroids (e.g., K-means) on audio embeddings extracted
with AudioEmbeddings (mHuBERT / wav2vec2 XLSR / Whisper encoder).

Usage:
    python train_kmeans.py --model utter-project/mhubert-147 --data path/to/audio_dir --k 200
"""

import os
import random
import logging
import argparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from AudioEmbedder import AudioEmbedder
from AudioProcessor import AudioProcessor
from Utils import list_audio_files, secs2human


def audio2embeddings(embedder, data_path: str, max_audio_files: int = None, max_frames_file: int = None,  max_frames_total: int = None):
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


def train_kmeans(embeddings: np.ndarray, k: int, max_iter=1000, batch_size=16384):
    """Train MiniBatchKMeans on large embedding matrix."""
    logging.info(f"Training MiniBatchKMeans: k={k}, dim={embeddings.shape[1]}")
    kmeans = MiniBatchKMeans(   
        n_clusters=k,
        batch_size=batch_size,
        random_state=0,
        verbose=1,
        max_iter=max_iter,
    )
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or HuggingFace model name (mHuBERT, wav2vec2-XLSR, Whisper supported)")
    parser.add_argument("--data", type=str, required=True, help="Audio file or directory of audio files.")
    parser.add_argument("--k", type=int, default=500, help="Number of centroids.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max number of iterations (set same as 2xK).")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size.")
    parser.add_argument("--max-audio-files", type=int, default=None, help="Max number of audio files to process (random subsampling).")
    parser.add_argument("--max-frames-file", type=int, default=None, help="Max number of frames to use per audio file (random subsampling).")
    parser.add_argument("--max-frames-total", type=int, default=None, help="Max number of frames to use (random subsampling).")
    parser.add_argument("--ouput", type=str, default="centroids", help="Output file name for centroids (OUTPUT.MODEL.K.npy is created).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_processor = AudioProcessor(top_db=30, stride=320, receptive_field=400)
    audio_embedder = AudioEmbedder(audio_processor, model=args.model)

    embeddings = audio2embeddings(audio_embedder, args.data, args.max_audio_files, args.max_frames_file, args.max_frames_total) #[N, D]    
    centroids = train_kmeans(embeddings, args.k, args.max_iter, args.batch_size) # [k, D]
    logging.info("Centroids shape:", centroids.shape)

    output = args.output + f".{os.path.basename(args.model)}" + f".{args.k}.npy"
    np.save(output, centroids)
    logging.info(f"Saved centroids → {output}")


