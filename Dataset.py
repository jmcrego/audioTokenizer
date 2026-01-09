# Dataset.py

import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
import soundfile as sf
from collections import defaultdict
from typing import Iterator, List, Dict, Optional

from torch.utils.data import Dataset, BatchSampler
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger("Dataset")

code2lang={
    "fr": "French",
    "ca": "Catalan",
    "de": "German",
    "es": "Spanish",
    "en": "English",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese"
}

def audio_length_in_embeddings(duration, conv_stride=30, sample_rate=16000, downsample_ratio=160):
    """
    Estimate number of tokens produced from an audio file
    after audio embedding + projection.
    """
    #whisper (always 1500 frames)
    if downsample_ratio == 160: 
        return (1500 + conv_stride - 1) // conv_stride
    
    # wav2vec OR mhubert
    n_samples = int(duration * sample_rate)

    # number of frame-level embeddings
    # (audio encoder internal downsampling)
    n_frames = (n_samples + downsample_ratio - 1) // downsample_ratio

    # number of tokens after stacking frames
    n_tokens = (n_frames + conv_stride - 1) // conv_stride

    return n_tokens


def build_prompt(audio_token="<extra_id_0>", src_lang=None, tgt_lang=None, asr=None):
    """
    Build a chat-style prompt for speech processing tasks.

    Supports:
    - ASR: transcribing speech into text.
    - STT: transcribing speech and translating it into a target language.

    Args:
        audio_token (str): Placeholder token representing the audio input.
        src_lang (str): Source language of the speech (required).
        tgt_lang (str, optional): Target language for translation.
        asr (str, optional): Transcription text, required if tgt_lang is provided.

    Returns:
        str: Formatted prompt compatible with chat-based LLMs.

    Raises:
        ValueError: If src_lang is not provided, or if tgt_lang is provided without asr.
    """    
    if src_lang is None:
        raise ValueError("Source language must be provided")

    # ASR: first step
    prompt = (
        f"<|im_start|>system\n"
        f"You are a professional {src_lang} interpreter.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Transcribe the following speech:\n"
        f"{audio_token}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if tgt_lang is not None:
        # STT: second step
        if asr is None:
            raise ValueError("asr must be provided")
        prompt += (
            f"{asr}\n"
            f"Translate into {tgt_lang}:\n"
        )

    return prompt


def build_target(asr=None, stt=None):
    """
    Build the training target corresponding to an ASR or STT prompt.

    If a translation (stt) is provided, it is used as the target.
    Otherwise, the transcription (asr) is used.

    Args:
        asr (str, optional): Transcription of the audio.
        stt (str, optional): Translation of the transcription.

    Returns:
        str: Target text terminated with an end-of-turn token.

    Raises:
        ValueError: If neither asr nor stt is provided.
    """
    if stt is not None:
        return f"{stt.strip()}\n<|im_end|>"
    if asr is not None:
        return f"{asr.strip()}\n<|im_end|>"
    raise ValueError("Either asr or stt must be provided")


def read_samples_from_tsv(path: str, max_duration: float = 30.0, sep: str = "\t"):
    """
    Read ASR and STT samples from a TSV file and build training examples.

    Each line in the file must contain either:
    - 3 fields: audio_path, source_language, transcription (ASR)
    - 5 fields: audio_path, source_language, transcription, target_language, translation (STT)

    The function validates audio files, constructs prompts and targets,
    and returns a list of samples suitable for training chat-based LLMs.

    Args:
        path (str): Path to the TSV file.
        max_duration (float): Maximum duration for an audio file.
        sep (str): Field separator used in the TSV file.

    Returns:
        list[dict]: A list of samples with audio metadata, prompt, and target text.
    """    
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)


    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(tqdm(f, total=total, desc=f"Reading TSV", unit="line"), start=1):
        
    # with open(path, "r", encoding="utf-8") as f:
    #     for line_no, line in enumerate(f, start=1):
            fields = line.rstrip("\n").split(sep)

            if len(fields) not in (3, 5):
                logger.warning(f"{path}:{line_no} expected 3 or 5 fields, got {len(fields)}")
                continue

            audio_path = fields[0]

            try:
                info = sf.info(audio_path)
                if not info.duration:
                    logger.warning(f"{path}:{line_no} invalid duration in audio file")
                    continue
                if info.duration > max_duration:
                    logger.warning(f"{path}:{line_no} audio file duration={info.duration} exceeds max_duration ({max_duration})")
                    continue

            except Exception as e:
                logger.warning(f"{path}:{line_no} failed to read audio: {e}")
                continue                

            src_lang = code2lang.get(fields[1], "")
            if not src_lang:
                logger.warning(f"{path}:{line_no} bad src_lang: {fields[1]}")
                continue

            asr = fields[2].strip()

            if not src_lang or not asr:
                logger.warning(f"{path}:{line_no} bad src_lang or empty asr")
                continue

            if len(fields) == 5: #STT 
                tgt_lang = code2lang.get(fields[3], "")
                stt = fields[4].strip()
                if not tgt_lang or not stt:
                    logger.warning(f"{path}:{line_no} bad tgt_lang or empty asr")
                    continue

            else: #ASR
                tgt_lang = None
                stt = None

            sample = {
                "audio_path": audio_path,
                "src_lang": src_lang,
                "asr": asr,
                "duration": info.duration,
                "tgt_lang": tgt_lang,
                "stt": stt,
            }

            samples.append(sample)

    logger.info(f"samples: {len(samples)}")
    stt_count = sum(1 for x in samples if x["stt"] is not None)
    logger.info("### Task stats ###")
    logger.info(f"ASR samples: {len(samples) - stt_count}")
    logger.info(f"STT samples: {stt_count}")

    if samples:
        durations = [x["duration"] for x in samples]
        total_duration = sum(durations)
        logger.info("### Audio duration stats ###")
        logger.info(f"sum: {total_duration}")
        logger.info(f"max: {max(durations)}")
        logger.info(f"min: {min(durations)}")
        logger.info(f"avg: {total_duration / len(durations)}")

    return samples


def log_sample(samples, idx, prompt, target, tokenizer):
    prompt_ids = samples[idx]["prompt_ids"]
    target_ids = samples[idx]["target_ids"]

    # Map prompt_ids to tokens
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    prompt_mapping = ", ".join(f"{id_}:'{tok}'" for id_, tok in zip(prompt_ids.tolist(), prompt_tokens))

    # Map target_ids to tokens
    target_tokens = tokenizer.convert_ids_to_tokens(target_ids)
    target_mapping = ", ".join(f"{id_}:'{tok}'" for id_, tok in zip(target_ids.tolist(), target_tokens))

    logger.info(
        f"### idx={idx} #######\n"
        f"### prompt #######\n{prompt}\n"
        f"### target #######\n{target}\n"
        f"### prompt_ids ###\n{prompt_mapping}\n"
        f"### target_ids ###\n{target_mapping}\n"
        f"##################"
    )

class BatchedLengthSampler(BatchSampler):
    def __init__(self, dataset, batch_size=4, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        """
        When shuffle is False, batches are build without sorting
        When shuffle is True, batches are build using the 'total_length' field of the dataset
        """
        if not shuffle:
            self.indices = list(range(len(dataset)))

        else:
            # Extract total_length for each sample
            lengths = np.array([s["total_length"] for s in dataset])

            # sort indices by length
            sorted_indices = np.argsort(lengths)

            # Create batches of sorted indices (contain samples of similar lengths)
            batches = [
                sorted_indices[i:i+self.batch_size] 
                for i in range(0, len(sorted_indices), self.batch_size)
            ]

            # Randomize batches
            np.random.shuffle(batches)

            # flat list of indices
            self.indices = np.concatenate(batches) #[idx for batch in batches for idx in batch]

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i:i+self.batch_size]

    def __len__(self):
        return len(self.dataset)


class Dataset(Dataset):
    """
    PyTorch Dataset for audio-to-LLM SFT training.
    """
    def __init__(
        self,
        file_path: str,
        tokenizer,
        audio_token="<extra_id_0>",
        # sample_rate=16000,
        # downsample_ratio=320,
        # conv_stride=30,
        # max_seq_len=1000,
        seed=42,
    ):
        """
        Read audio embedding cache metadata.

        Args:
            file_path (str): Path to meta.json
            cache_dir (str): Directory containing .pt embedding files

        Returns:
            dict with keys:
                - embedder: embedder configuration
                - samples: list of samples with resolved pt_path
        """

        #random seed for reproducibility
        np.random.seed(seed)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"meta.json not found: {file_path}")

        if file_path.endswith('.json'):
            with open(file_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.data = meta["samples"]
        else:
            self.data = read_samples_from_tsv(file_path)

        for idx in range(len(self.data)):
            sample = self.data[idx]

            prompt = build_prompt(audio_token=audio_token, src_lang=sample['src_lang'], tgt_lang=sample['tgt_lang'], asr=sample["asr"] if sample.get("tgt_lang") else None)
            prompt_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False).input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)
            self.data[idx]["prompt_ids"] = prompt_ids

            target = build_target(asr=sample['asr'], stt=sample['stt'])
            target_ids = tokenizer(target, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False).input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)
            self.data[idx]["target_ids"] = target_ids

            if "n_audio_embs" in sample:
                n_tokens_audio = sample['n_audio_embs']
                self.data[idx]["total_length"] = n_tokens_audio + len(prompt_ids) + len(target_ids)

            if idx % 50000 == 0:
                log_sample(self.data, idx, prompt, target, tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct", use_fast=True)

    # Create dataset from file
    ds = Dataset(file_path=sys.argv[1], tokenizer=tokenizer)
    print(f"Dataset size: {len(ds)} samples")

    # Create sampler from datset
    sampler = BatchedLengthSampler(ds, shuffle=True)
    print(f"Sampler size: {len(sampler)} samples")

    # Iterate over sampler and print batch info
    for i, idx in enumerate(sampler):
        print(f"Batch {i}")
        for id in idx:
            e = ds[id]
            n_prompt = len(e["prompt_ids"])
            n_target = len(e["target_ids"])
            n_audio = e["total_length"] - n_prompt - n_target
            duration = e["duration"]
            print(f"\tidx={id}\tn_audio={n_audio}, n_prompt={n_prompt}, n_target={n_target}, n_total={e['total_length']}, duration={duration}")
