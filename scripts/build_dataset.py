# build_dataset.py

import os
import json
import argparse
import logging
import torch
from transformers import AutoTokenizer

from Dataset import Dataset   

logger = logging.getLogger("build_dataset")

def save(args):
    meta = {k: v for k, v in locals().items()}
    logger.info(f"Arguments: {meta}")        

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    logger.info("Building dataset (prompt/target/audio lengths)")
    dataset = Dataset(
        file_path=args.data,
        tokenizer=tokenizer,
        audio_token=args.audio_token,
        sample_rate=args.sample_rate,
        downsample_ratio=args.downsample_ratio,
        conv_stride=args.conv_stride,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info("Sorting samples by total_length")
    data = sorted(dataset.data, key=lambda x: x["total_length"])

    lengths = [x["total_length"] for x in data]
    audio_times = [x["audio_time"] for x in data]

    meta = {
        "num_samples": len(lengths),
        "max_seq_len": args.max_seq_len,
        "tokenizer": args.tokenizer,
        "sample_rate": args.sample_rate,
        "downsample_ratio": args.downsample_ratio,
        "conv_stride": args.conv_stride,
        "audio_time_stats":{
            "sum": sum(audio_times),
            "max": max(audio_times),
            "min": min(audio_times),
            "avg": sum(audio_times)/len(audio_times),
        },
        "length_stats": {
            "min": min(lengths),
            "med": lengths[len(lengths)//2],
            "avg": sum(lengths)/len(lengths),
            "max": max(lengths),
        }
    }
    logger.info(f"meta: {meta}")

    logger.info(f"Saving dataset to {args.output}")
    torch.save({"data": data, "meta": meta}, args.output)

    with open(args.output.replace(".pt","") + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done.")

def load(data_file):


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and dump length-sorted Dataset (no batching)")
    # input / output
    parser.add_argument("--data", required=True, help="TSV dataset file")
    parser.add_argument("--output", required=True, help="Output .pt .json files")
    # tokenizer
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path or name")
    parser.add_argument("--audio-token", default="<extra_id_0>")
    # dataset params
    parser.add_argument("--sample-rate", type=int, default=16000, help="sample_rate of audio signal (usually 16kHz)")
    parser.add_argument("--downsample-ratio", type=int, default=160, help="downsample ratio (160 for whisper, 320 for mhubert/wav2vec)")
    parser.add_argument("--conv-stride", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    save(args)
