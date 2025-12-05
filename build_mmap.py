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
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import tempfile

from scripts.AudioEmbedder import AudioEmbedder
from scripts.Utils import list_audio_files

def arguments(args):
    args.pop('self', None)  # None prevents KeyError if 'self' doesn't exist
    if 'audio_embedder' in args:
        args['audio_embedder'] = args['audio_embedder'].meta
    return args

SUPPORTED_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

def list_audio_files(path: str, field=0):
    """Return list of audio files from a file or directory."""
    files = []
    if os.path.isfile(path):
        with open(path, 'r') as fd:
            for l in fd:
                parts = l.strip().split('\t')
                if len(parts) > field:
                    files.append(parts[field])
    else:
        for root, _, fs in os.walk(path):
            for f in fs:
                if f.lower().endswith(SUPPORTED_EXT):
                    files.append(os.path.join(root, f))
    return sorted(files)

def build_mmap_from_audio(
        audio_embedder,
        data_path: str,
        memmap_path: str,
        max_f: int = None,
        max_e: int = None,
        max_epf: int = None):
    """
    Convert audio files to embeddings and store them in a numpy memmap on disk.
    NOTE: max_e must be provided (memmap needs a fixed shape).
    """
    meta = arguments(locals())
    if max_e is None:
        raise ValueError("max_e must be set when using memmap.")

    audio_files = list_audio_files(data_path)
    logging.info(f"Found {len(audio_files)} audio files.")
    if not audio_files:
        raise RuntimeError("No audio files found!")

    if max_f is not None and max_f < len(audio_files):
        random.shuffle(audio_files)
        audio_files = audio_files[:max_f] if max_f is not None else audio_files
        logging.info(f"Kept {len(audio_files)} audio files.")

    # progress bars for files and embeddings
    f_bar = tqdm(total=len(audio_files), desc="Files", unit="file", position=0, leave=True)
    e_bar = tqdm(total=max_e, desc="Embeds", unit="emb", position=1, leave=True)

    # create memmap file (mode 'w+' creates or overwrites) with max_e embeddings of dimension D
    X = np.memmap(memmap_path+".memmap", dtype=np.float32, mode='w+', shape=(max_e, audio_embedder.D))
    ptr = 0

    for i, path in enumerate(audio_files):
        try:            
            if ptr >= max_e: #reached max capacity
                break

            emb = audio_embedder(path)  # Tensor [T, D]
            emb = emb.cpu().numpy()
            f_bar.update(1)

            # per-file subsample
            if max_epf is not None and emb.shape[0] > max_epf:
                idx = np.random.choice(emb.shape[0], max_epf, replace=False)
                emb = emb[idx]

            if emb.shape[0] == 0:
                continue

            add = min(emb.shape[0], max_e - ptr)
            if add < emb.shape[0]:
                emb = emb[:add]

            # write into memmap
            X[ptr:ptr+add, :] = emb
            ptr += add
            e_bar.update(add)

        except Exception as e:
            logging.error(f"ERROR with {path}: {e}")

    # flush to disk
    X.flush()
    f_bar.n = f_bar.total; f_bar.refresh(); f_bar.close()
    e_bar.n = e_bar.total if ptr >= max_e else ptr; e_bar.refresh(); e_bar.close()

    meta['n_vectors'] = ptr
    meta['D'] = audio_embedder.D
    with open(memmap_path + ".json", "w") as f:
        json.dump(meta, f, indent=4)

    logging.info(f"Finished writing memmap: {memmap_path}.memmap")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio files to embeddings and store them in a numpy memmap on disk.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model", type=str, help="Path or HuggingFace model name.")
    parser.add_argument("data", type=str, help="File containing audio files to consider.")
    parser.add_argument("memmap", type=str, help="Output memmap prefix (MEMMAP.\{memmap,json,log\} will be written).")
    parser.add_argument("--top_db", type=int, default=0, help="Threshold (db) to remove silence (0 for no filtering).")
    parser.add_argument("--max-f", type=int, default=None, help="Max total number of audio files.")
    parser.add_argument("--max-e", type=int, default=None, help="Max total number of embeddings.")
    parser.add_argument("--max-epf", type=int, default=None, help="Max number of embeddings-per-file (all files in DATA if not set).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    if args.max_e is None:
        args.max_e = len(list_audio_files(args.data))

    if args.memmap.endswith(".memmap"):
        args.memmap = args.memmap[:-7] #remove extension

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(),logging.FileHandler(f"{args.memmap}.log")])
 
    audio_embedder = AudioEmbedder(model=args.model, top_db=args.top_db, device=args.device)

    if not os.path.exists(args.memmap):
        build_mmap_from_audio(
            audio_embedder=audio_embedder,
            data_path=args.data,
            memmap_path=args.memmap,
            max_f=args.max_f,
            max_epf=args.max_epf,
            max_e=args.max_e)
    else:
        raise(f"ERROR: memmap file {args.memmap} already exists!")
