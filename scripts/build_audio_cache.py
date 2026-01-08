#build_audio_cache.py

import argparse
from tqdm import tqdm
import itertools
import logging
import string
import torch
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dataset import read_samples_from_tsv
from Embedder import Embedder

logger = logging.getLogger("build_audio_cache")

def process_batch(audio_embedder, samples, batch, cache_dir, device, dtype):
    # batch is a list with idx's in samples to embed
    audio_paths = [samples[idx]["audio_path"] for idx in batch]

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=(device == "cuda")):
            audio_embs, audio_mask = audio_embedder(audio_paths)

    for i, idx in enumerate(batch):
        pt_path = os.path.join(cache_dir, f"{idx:09d}.pt")
        #To avoid partially written .pt files if killed mid-save:
        tmp_path = pt_path + ".tmp"
        torch.save({"audio_embs": audio_embs[i].cpu(), "audio_mask": audio_mask[i].cpu()}, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, pt_path)
        samples[idx]["pt_path"] = f"{idx:09d}.pt"
        samples[idx]["n_audio_embs"] = int(audio_mask[i].sum().item())


def build_audio_cache(
        tsv_path,
        cache_dir,
        embedder_path,
        device,
        dtype,
        batch_size):

    os.makedirs(cache_dir, exist_ok=True)
    torch_dtype = getattr(torch, dtype)

    # Initialize embedder
    audio_embedder = Embedder(config={'path': embedder_path})
    audio_embedder.to(device, dtype=torch_dtype)
    audio_embedder.eval()

    # Read TSV samples
    samples = read_samples_from_tsv(tsv_path)

    # Prepare batches
    batch = []

    for idx in tqdm(range(len(samples)), total=len(samples), desc="Process audio samples", unit="sample", unit_scale=True):

        pt_path = os.path.join(cache_dir, f"{idx:09d}.pt")

        if os.path.exists(pt_path):
            try:
                data = torch.load(pt_path, map_location="cpu")
                samples[idx]["pt_path"] = f"{idx:09d}.pt"
                samples[idx]["n_audio_embs"] = int(data["audio_mask"].sum().item())
                continue
            except Exception:
                logger.warning(f"Corrupted cache: {pt_path}, recomputing")
    
        batch.append(idx)
        # Process batch
        if len(batch) == batch_size:
            process_batch(audio_embedder, samples, batch, cache_dir, device, torch_dtype)
            batch = []

    # Process any remaining files
    if batch:
        process_batch(audio_embedder, samples, batch, cache_dir, device, torch_dtype)


    logger.info(f"Saved {len(samples)} .pt files in {cache_dir}")

    # Save mapping with metadata
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
    parser = argparse.ArgumentParser(description="Cache audio embeddings as .pt files from TSV (batched)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tsv_path", type=str, required=True, help="TSV file with audio metadata")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory to store .pt files and mapping.json")
    parser.add_argument("--embedder_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium", help="Path of audio embedder")
    parser.add_argument("--device", type=str, default="cuda", help="Device for embeddings")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch dtype for embeddings")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for audio embedding")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    build_audio_cache(
        args.tsv_path,
        args.cache_dir,
        args.embedder_path,
        args.device,
        args.dtype,
        args.batch_size)
