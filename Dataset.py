# Dataset.py

import torch
import logging
import numpy as np
import soundfile as sf
from collections import defaultdict

from torch.utils.data import Dataset, Sampler
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
}


def build_prompt(audio_token="<[audio]>", src_lang=None, tgt_lang=None, bos_token="<bos>"):
    if src_lang is not None and src_lang not in code2lang:
        raise ValueError(f"Source language code '{src_lang}' not found.")        
    if tgt_lang is not None and tgt_lang not in code2lang:
        raise ValueError(f"Target language code '{tgt_lang}' not found.")        

    src_lang = code2lang.get(src_lang)
    tgt_lang = code2lang.get(tgt_lang)

    prompt = f"{audio_token}\nTask:\n"
    
    if src_lang and tgt_lang:
        prompt += (
            f"1. Transcribe the speech.\n"
            f"2. Translate the transcription into {tgt_lang}.\n"
        )
    elif src_lang:
        prompt += "Transcribe the speech.\n"
    elif tgt_lang:
        prompt += f"Translate the speech into {tgt_lang}.\n"
    else:
        raise ValueError("No src_lang or tgt_lang provided")
    
    prompt += "Answer:\n"
    return bos_token+prompt

def build_target(asr=None, stt=None, asr_start_token="<asr>", asr_end_token="</asr>", stt_start_token="<stt>", stt_end_token="</stt>", eos_token="<eos>"):
    if (asr is None or asr == "") and (stt is None or stt == ""):
        raise ValueError("No ASR or STT text provided.")

    target = ""

    if asr is not None and asr != "":
        target += f"{asr_start_token} {asr} {asr_end_token}\n"

    if stt is not None and stt != "":
        target += f"{stt_start_token} {stt} {stt_end_token}\n"

    return target+eos_token
 

class BatchedLengthSampler(Sampler):
    def __init__(self, dataset, batch_size=4, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        if not shuffle:
            self.all_indices = list(range(len(dataset)))

        else:
            # Extract total_length for each sample
            lengths = np.array([s["total_length"] for s in dataset])

            # sort indices by length
            if self.shuffle:
                sorted_indices = np.argsort(lengths)
                #sorted_lengths = lengths[sorted_indices]
            else:
                sorted_indices = np.arange(len(lengths))

            # Create batches of sorted indices (contain samples of similar lengths)
            batches = [
                sorted_indices[i:i+self.batch_size] 
                for i in range(0, len(sorted_indices), self.batch_size)
            ]

            # Randomize batches
            if self.shuffle:
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
    Builds prompt/target, estimates audio token lengths, and filters by max_seq_len.
    """
    def __init__(
        self,
        file_path: str,
        tokenizer,
        add_tokens={},
        sample_rate=16000,
        downsample_ratio=320,
        stack_size=8,
        max_seq_len=1000,
        seed=42,
    ):
        self.tokenizer = tokenizer
        self.asr_start_token = add_tokens['asr_start_token']
        self.asr_end_token = add_tokens['asr_end_token']
        self.stt_start_token = add_tokens['stt_start_token']
        self.stt_end_token = add_tokens['stt_end_token']
        self.audio_token = add_tokens['audio_token']
        self.sample_rate = sample_rate
        self.downsample_ratio = downsample_ratio
        self.stack_size = stack_size
        self.max_seq_len = max_seq_len

        #random seed for reproducibility
        np.random.seed(seed)

        self.tasks = defaultdict(int)

        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i,line in enumerate(f):
                parts = line.strip("\n").split("\t")
                if len(parts) < 5:
                    continue
                audio_path, src_lang, asr, tgt_lang, stt = parts[:5]

                task = "asr+stt" if asr and stt else "asr" if asr else "stt" if stt else None
                self.tasks[task] += 1

                src_lang = src_lang if src_lang else None
                tgt_lang = tgt_lang if tgt_lang else None

                prompt = build_prompt(self.audio_token, src_lang, tgt_lang, self.tokenizer.bos_token)
                prompt_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                ).input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)

                target = build_target(asr, stt, self.asr_start_token, self.asr_end_token, self.stt_start_token, self.stt_end_token, self.tokenizer.eos_token)
                target_ids = tokenizer(
                    target,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                ).input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)

                if i % 10000 == 0:
                    logger.info(f"sample={i}\n### prompt #######\n{prompt}\n### target #######\n{target}\n### prompt_ids ###\n{prompt_ids}\n### target_ids ###\n{target_ids}\n##################")

                audio_time, n_audio = self.audio_length_in_embeddings(audio_path)
                total_length = n_audio + len(prompt_ids) + len(target_ids)
                if total_length > max_seq_len:
                    logger.info(f"Skipped audio by len={n_audio} {audio_path}")
                    continue

                self.data.append({
                    "audio_path": audio_path,
                    "prompt_ids": prompt_ids,
                    "target_ids": target_ids,
                    "total_length": total_length,
                    "audio_time": audio_time,
                })
                
            logger.info(f"Read dataset {file_path} with {len(self.data)} samples ({dict(self.tasks)})")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


    def audio_length_in_embeddings(self, filepath):
        """
        Estimate number of tokens produced from an audio file
        after audio embedding + frame stacking (no chunking).
        """
        try:
            info = sf.info(filepath)
            if not info.duration:
                return 0, 0
            
            if self.downsample_ratio == 160: #whisper (this should be better done!!)
                n_tokens = (WHISPER_FRAMES + self.stack_size - 1) // self.stack_size
                return info.duration, n_tokens

            # total audio samples
            n_samples = int(info.duration * self.sample_rate)

            # number of frame-level embeddings
            # (audio encoder internal downsampling)
            n_frames = (n_samples + self.downsample_ratio - 1) // self.downsample_ratio

            # number of tokens after stacking frames
            n_tokens = (n_frames + self.stack_size - 1) // self.stack_size

            return info.duration, n_tokens

        except Exception:
            return 0, 0


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
            audio_time = e["audio_time"]
            print(f"\tidx={id}\tn_audio={n_audio}, n_prompt={n_prompt}, n_target={n_target}, n_total={e['total_length']}, audio_time={audio_time}")
