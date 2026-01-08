# build_audio_cache.py

import argparse
from tqdm import tqdm
import logging
import torch
import json
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dataset import read_samples_from_tsv
from Embedder import Embedder

logger = logging.getLogger("build_audio_cache")


def process_bucket(audio_embedder, samples, bucket_indices, cache_dir, device, dtype, bucket_id):
    """
    Process a list of sample indices (bucket) and save all embeddings in a single .pt file.
    """
    audio_paths = [samples[idx]["audio_path"] for idx in bucket_indices]

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=(device == "cuda")):
            audio_embs, audio_mask = audio_embedder(audio_paths)

    # move to CPU once
    audio_embs_cpu = audio_embs.cpu().contiguous()
    audio_mask_cpu = audio_mask.cpu().contiguous()

    # save bucket
    pt_path = os.path.join(cache_dir, f"bucket_{bucket_id:06d}.pt")
    tmp_path = pt_path + ".tmp"
    torch.save({
        "audio_embs": audio_embs_cpu,   # shape [bucket_size, ...]
        "audio_mask": audio_mask_cpu,   # shape [bucket_size, ...]
        # "indices": bucket_indices       # original sample indices
    }, tmp_path, _use_new_zipfile_serialization=False)
    os.replace(tmp_path, pt_path)

    # update sample metadata
    for i, idx in enumerate(bucket_indices):
        samples[idx]["pt_path"] = f"bucket_{bucket_id:06d}.pt"
        samples[idx]["offset"] = i
        samples[idx]["n_audio_embs"] = int(audio_mask_cpu[i].sum().item())


def build_audio_cache(
        tsv_path,
        cache_dir,
        embedder_path,
        device,
        dtype,
        bucket_size):

    os.makedirs(cache_dir, exist_ok=True)
    torch_dtype = getattr(torch, dtype)

    # Initialize embedder
    audio_embedder = Embedder(config={'path': embedder_path})
    audio_embedder.to(device, dtype=torch_dtype)
    audio_embedder.eval()

    # Read TSV samples
    samples = read_samples_from_tsv(tsv_path)
    logger.info(f"Read {len(samples)} samples from {tsv_path}")

    bucket_indices = []
    bucket_id = 0

    for idx in tqdm(range(len(samples)), total=len(samples), desc="Process audio samples", unit="sample"):

        # Skip if already cached
        pt_path_tmp = os.path.join(cache_dir, f"bucket_*")  # placeholder
        existing_pt = samples[idx].get("pt_path")
        if existing_pt and os.path.exists(os.path.join(cache_dir, existing_pt)):
            continue

        bucket_indices.append(idx)

        if len(bucket_indices) == bucket_size:
            process_bucket(audio_embedder, samples, bucket_indices, cache_dir, device, torch_dtype, bucket_id)
            bucket_indices = []
            bucket_id += 1

    # Process remaining samples in last bucket
    if bucket_indices:
        process_bucket(audio_embedder, samples, bucket_indices, cache_dir, device, torch_dtype, bucket_id)

    logger.info(f"Saved embeddings in {bucket_id + 1} buckets in {cache_dir}")

    # Save meta.json
    meta_path = os.path.join(cache_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "embedder": {
                "path": embedder_path,
                "dtype": dtype,
                "device": str(device),
            },
            "samples": [{
                "audio_path": s["audio_path"],
                "pt_path": s["pt_path"],
                "offset": s["offset"],
                "duration": s["duration"],
                "n_audio_embs": s["n_audio_embs"],
                "src_lang": s["src_lang"],
                "asr": s["asr"],
                "tgt_lang": s.get("tgt_lang"),
                "stt": s.get("stt"),
            } for s in samples],
        }, f, indent=2)

    logger.info(f"Saved {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache audio embeddings as .pt files from TSV (bucketed)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tsv_path", type=str, required=True, help="TSV file with audio metadata")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory to store bucket .pt files and meta.json")
    parser.add_argument("--embedder_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium", help="Path of audio embedder")
    parser.add_argument("--device", type=str, default="cuda", help="Device for embeddings")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch dtype for embeddings")
    parser.add_argument("--bucket_size", type=int, default=64, help="Number of samples per saved bucket")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler()])

    build_audio_cache(
        args.tsv_path,
        args.cache_dir,
        args.embedder_path,
        args.device,
        args.dtype,
        args.bucket_size)
