# Dataset.py

import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from typing import Iterator, List, Dict, Optional

from torch.utils.data import Dataset, BatchSampler
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger("Dataset")

WHISPER_FRAMES = 1500

code2lang={
    "fr": "French",
    "ca": "Catalan",
    "de": "German",
    "es": "Spanish",
    "en": "English",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese",
    "zh-CN": "Chinese"
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

def build_template(
        type="instruct", task="asr", 
        audio_token="<audio>", bos_token="<s>", eos_token="</s>", 
        src_lang=None, tgt_lang=None, 
        asr_text=None, stt_text=None
    ):
    if type not in {'instruct', 'declarative', 'oneline'}:
        raise ValueError("unknown type template: use 'instruct' OR 'declarative' OR 'oneline'")

    if task not in {'asr', 'ast', 'stt'}:
        raise ValueError("unknown task template: use 'asr' OR 'ast' OR 'stt'")

    ####################################################################
    ### oneline prompt #################################################
    ####################################################################
    if type == "oneline":
        # Automatic Speech Recognition
        if task == "asr":
            prompt=f"{audio_token}<|{task}|>" 
            target=f"<|{src_lang}|>{asr_text}" if src_lang is not None and asr_text is not None else None

        # Automatic Speech Translation
        elif task == "ast":
            if tgt_lang is None:
                raise ValueError("tgt_lang must exist for ast task")
            
            prompt=f"{audio_token}<|{task}|><|{tgt_lang}|>" 
            target=f"<|{src_lang}|>{stt_text}" if src_lang is not None and stt_text is not None else None

        # Speech Transcription and Translation
        elif task == "stt":
            if src_lang is None or tgt_lang is None:
                raise ValueError("src_lang/tgt_lang must exist for stt task")
            if asr_text is None:
                raise ValueError("asr_text must exist for stt task")
            
            prompt=f"{audio_token}<|{task}-asr|><|{src_lang}|>{asr_text}<|{task}|><|{tgt_lang}|>" 
            target=f"{stt_text}" if stt_text is not None else None

        # Text to Text Translation (No audio involved)
        elif task == "ttt":
            if tgt_lang is None:
                raise ValueError("tgt_lang must exist for ast task")
            
            prompt=f"{asr_text}<|{task}|><|{tgt_lang}|>"
            target=f"<|{src_lang}|>{stt_text}" if src_lang is not None and stt_text is not None else None

        return bos_token+prompt, target+eos_token if target is not None else None

    ####################################################################
    ### instruct prompt ################################################
    ####################################################################
    elif type == "instruct":
        if task == "asr":
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

        elif task == "stt":
            prompt = (
                f"<|im_start|>system\n"
                f"You are a professional {src_lang} interpreter.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Translate the following speech into {tgt_lang}:\n"
                f"{audio_token}\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            target = f"{stt_text}<|im_end|>"

        elif task == "2stt":
            prompt = (
                f"<|im_start|>system\n"
                f"You are a professional {src_lang} interpreter.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Transcribe the following speech:\n"
                f"{audio_token}\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{asr_text}\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Translate into {tgt_lang}:\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            target = f"{stt_text}<|im_end|>"
        return prompt, target

    ####################################################################
    ### declarative prompt #############################################
    ####################################################################
    elif type == "declarative":
        if task == "asr":
            prompt = (
                f"{bos_token}<|task:asr|><|src_lang:{src_lang}|>\n"
                f"<|speech|>\n" 
                f"{audio_token}\n" 
                f"<|transcription|>\n"
            )
            target = asr_text + eos_token if asr_text is not None else None

        elif task == "stt":
            prompt = (
                f"{bos_token}<|task:stt|><|src_lang:{src_lang}|><|tgt_lang:{tgt_lang}|>\n"
                f"<|speech|>\n"
                f"{audio_token}\n"
                f"<|translation|>\n"
            )
            target = stt_text + eos_token if stt_text is not None else None

        elif task == "2stt":
            prompt = (
                f"{bos_token}<|task:asr|><|src_lang:{src_lang}|>\n"
                f"<|speech|>\n"
                f"{audio_token}\n"
                f"<|transcription|>\n"
                f"{asr_text}\n"
                f"<|task:stt|><|tgt_lang:{tgt_lang}|>\n"
                f"<|translation|>\n"
            )
            target = stt_text + eos_token if stt_text is not None else None

        return prompt, target



def read_samples_from_tsv(path: str, max_duration: float = 30.0, sep: str = "\t", use_tqdm=True):
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
        for line_no, line in enumerate(tqdm(f, desc=f"Reading {Path(path).name}", unit=" sample", disable=not use_tqdm), start=1):
        
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

            src_lang = fields[1] #code2lang.get(fields[1], "")
            if not src_lang:
                logger.warning(f"{path}:{line_no} bad src_lang: {fields[1]}")
                continue

            asr = fields[2].strip()

            if not src_lang or not asr:
                logger.warning(f"{path}:{line_no} bad src_lang or empty asr")
                continue

            if len(fields) == 5: #STT 
                tgt_lang = fields[3] #code2lang.get(fields[3], "")
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
        bos_token="<bos>",
        eos_token="<eos>",
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
        self.is_cached = Path(file_path).name == "meta.json"

        if self.is_cached:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            self.info = self.meta['info']
            self.data = self.meta['samples']
            file_path_dir = Path(file_path).parent

        else:
            self.info = None
            self.data = read_samples_from_tsv(file_path)


        for idx in range(len(self.data)):
            sample = self.data[idx]

            prompt, target = build_template(
                type="declarative", task="asr", 
                audio_token=audio_token, bos_token=bos_token, eos_token=eos_token, 
                src_lang=sample['src_lang'], tgt_lang=sample['tgt_lang'], 
                asr_text=sample["asr"], stt_text=sample['stt']
            )

            prompt_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False).input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)
            self.data[idx]["prompt_ids"] = prompt_ids

            target_ids = tokenizer(target, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False).input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)
            self.data[idx]["target_ids"] = target_ids

            if self.is_cached: #convert the pt_path into an absolute path
                self.data[idx]["pt_path"] = file_path_dir / self.data[idx]["pt_path"] 

            else: #tsv dataset
                conv_kernel = 30
                conv_stride = 30
                n_tokens_audio = (WHISPER_FRAMES - conv_kernel) // conv_stride + 1
                self.data[idx]["total_length"] = n_tokens_audio + len(prompt_ids) + len(target_ids)

            if idx % 50000 == 0:
                log_sample(self.data, idx, prompt, target, tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
    sampler = BatchedLengthSampler(ds, batch_size=4, shuffle=True)
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
