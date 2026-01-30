import argparse
from tqdm import tqdm
import logging
import torch
import json
import os
import sys
import time

from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dataset import read_samples_from_tsv, build_template #build_prompt, build_target
from Embedder import Embedder

logger = logging.getLogger("build_audio_cache")


def process_batch(audio_embedder, samples, batch_indices, device, dtype):
    """
    Embed audio for a batch of indices.
    Returns embeddings on CPU.
    """
    audio_paths = [samples[idx]["audio_path"] for idx in batch_indices]

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=(device == "cuda")):
            audio_res = audio_embedder(audio_paths)  # Whisper: ignore mask

        audio_embs = audio_res[0] if isinstance(audio_res, tuple) else audio_res

    return audio_embs.cpu().contiguous()


def split_batch(batch_indices, audio_embs):
    """
    Convert batch embeddings into a list of (index, embedding) tuples.
    """
    return [(idx, audio_embs[i]) for i, idx in enumerate(batch_indices)]


def save_bucket(samples, bucket, cache_dir, bucket_id):
    """
    Save a bucket of embeddings to disk and update sample metadata.
    """
    pt_path = os.path.join(cache_dir, f"bucket_{bucket_id:06d}.pt")
    tmp_path = pt_path + ".tmp"

    indices = [idx for idx, _ in bucket]
    embs = torch.stack([emb for _, emb in bucket])  # (B, T, D)

    torch.save({"audio_embs": embs}, tmp_path, _use_new_zipfile_serialization=False)
    os.replace(tmp_path, pt_path)

    # update sample metadata
    for i, idx in enumerate(indices):
        samples[idx]["pt_path"] = os.path.basename(pt_path)
        samples[idx]["offset"] = i
        #not used: samples[idx]["n_audio_embs"] = embs.shape[1]  # T

def save_sorted_samples(samples, audio_embedder, batch_size, bucket_size, cache_dir, device, torch_dtype):
    # embed (batch_size) samples and save embeddings in files containing bucket_size samples
    batch_indices = []
    bucket = []
    bucket_id = 0
    t_embedding = 0.0
    t_saving = 0.0

    for idx in tqdm(range(len(samples)), total=len(samples), desc="Embedding audio", unit=" sample"):

        batch_indices.append(idx)

        # process batch
        if len(batch_indices) == batch_size:
            tic = time.time()
            audio_embs_cpu = process_batch(audio_embedder, samples, batch_indices, device, torch_dtype)
            t_embedding += time.time() - tic
            bucket.extend(split_batch(batch_indices, audio_embs_cpu))
            batch_indices = []

        # process bucket
        while len(bucket) >= bucket_size:
            tic = time.time()
            save_bucket(samples, bucket[:bucket_size], cache_dir, bucket_id)
            t_saving += time.time() - tic
            bucket = bucket[bucket_size:]
            bucket_id += 1

    # process remaining batch
    if batch_indices:
        tic = time.time()
        audio_embs_cpu = process_batch(audio_embedder, samples, batch_indices, device, torch_dtype)
        t_embedding += time.time() - tic
        bucket.extend(split_batch(batch_indices, audio_embs_cpu))

    # process remaining bucket
    while bucket:
        tic = time.time()
        save_bucket(samples, bucket[:bucket_size], cache_dir, bucket_id)
        t_saving += time.time() - tic
        bucket = bucket[bucket_size:]
        bucket_id += 1

    logger.info(f"Saved {len(samples)} embeddings in {bucket_id} buckets dir={cache_dir}")
    logger.info(f"Embedding time = {t_embedding:.2f}s, Saving time = {t_saving:.2f}s")


def build_audio_cache(args):
    #     tsv_path, 
    #     cache_dir, 
    #     embedder_path, 
    #     tokenizer_path, 
    #     audio_token,
    #     template,
    #     task,
    #     device, 
    #     dtype, 
    #     batch_size, 
    #     bucket_size,
    # ):
    os.makedirs(args.cache_dir, exist_ok=True)
    torch_dtype = getattr(torch, args.dtype)

    # Initialize embedder
    audio_embedder = Embedder(config={'path': args.embedder_path})
    audio_embedder.to(args.device, dtype=torch_dtype)
    audio_embedder.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    # Read TSV samples
    samples = read_samples_from_tsv(args.tsv_path)

    # Compute tokenized lengths
    for s in tqdm(samples, total=len(samples), desc="Tokenizing text", unit=" sample"):
        prompt, target = build_template(type=args.template, task=args.task, 
            audio_token=args.audio_token, bos_token=tokenizer.bos_token, eos_token=tokenizer.eos_token, 
            src_lang=s.get("src_lang"), tgt_lang=s.get("tgt_lang"), 
            asr_text=s.get("asr"), stt_text=s.get("stt"),
        )
        s["seq_len"] = len(tokenizer(prompt, target, padding=False, truncation=False, add_special_tokens=False)["input_ids"])
    
    # Sort samples by tokenized length (shortest → longest)
    samples.sort(key=lambda x: x["seq_len"])

    save_sorted_samples(samples, audio_embedder, args.batch_size, args.bucket_size, args.cache_dir, args.device, torch_dtype)

    # Save meta.json
    meta_path = os.path.join(args.cache_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "info": {
                "tsv_path": args.tsv_path, 
                "cache_dir": args.cache_dir, 
                "embedder_path": args.embedder_path,
                "tokenizer_path": args.tokenizer_path,
                "dtype": args.dtype,
                "bucket_size": args.bucket_size,
            },
            "samples": [{
                "audio_path": s["audio_path"],
                "pt_path": s["pt_path"],
                "offset": s["offset"],
                "duration": s.get("duration"),
                "src_lang": s.get("src_lang"),
                "asr": s.get("asr"),
                "tgt_lang": s.get("tgt_lang"),
                "stt": s.get("stt"),
            } for s in samples],
        }, f, indent=2)
    logger.info(f"Saved {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache audio embeddings as .pt files from TSV (bucketed)")
    parser.add_argument("--tsv_path", type=str, required=True, help="TSV file with audio metadata")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory to store bucket .pt files and meta.json")
    parser.add_argument("--embedder_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium")
    parser.add_argument("--tokenizer_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B")
    parser.add_argument("--audio_token", type=str, default="<extra_id_0>")
    parser.add_argument("--template", type=str, default="oneline", help="declarative OR instruct OR oneline")
    parser.add_argument("--task", type=str, default="asr", help="asr OR ast OR stt OR ttt")
    parser.add_argument("--device", type=str, default="cuda", help="Device for embeddings")
    parser.add_argument("--dtype", type=str, default="float16", help="Torch dtype for embeddings")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples to fed to embedder")
    parser.add_argument("--bucket_size", type=int, default=128, help="Number of samples per saved bucket")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    build_audio_cache(args)
