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


def build_prompt(src_lang=None, tgt_lang=None, audio_token="<extra_id_0>", bos_token="<bos>"):
    """
    Build prompt for audio-to-text or audio-to-text+translation.

    - src_lang: source language code (required)
    - tgt_lang: target language code (optional, only if translation is desired)
    - bos_token: token to prepend at the start of the prompt

    The prompt always includes expected tags Transcription and optionally Translation.
    """
    if src_lang is None:
        raise ValueError("Source language (src_lang) must be provided")
    if src_lang not in code2lang:
        raise ValueError(f"Source language code '{src_lang}' not found.")        
    if tgt_lang is not None and tgt_lang not in code2lang:
        raise ValueError(f"Target language code '{tgt_lang}' not found.")        

    src_name = code2lang[src_lang]
    tgt_name = code2lang[tgt_lang] if tgt_lang else None

    lines = [
        "Task:"
    ]

    if tgt_name:
        lines.append(f"Transcribe the {src_name} speech Input and translate it into {tgt_name}.")
    else:
        lines.append(f"Transcribe the {src_name} speech Input.")

    lines.extend([
        "\nInput:",
        audio_token,
        "\nOutput:",
        # "Transcription]"
    ])

    # if tgt_name:
    #     lines.append("Translation")

    # join lines with newline and prepend BOS token
    prompt = bos_token + "\n" + "\n".join(lines) + "\n"
    return prompt


def build_target(asr=None, stt=None, asr_token="<extra_id_1>", stt_token="<extra_id_2>", eos_token="<|im_end|>"):
    """
    Build target string for transcription and optional translation.
    Tags Transcription and Translation are included to match the prompt.
    """
    if (asr is None or asr.strip() == "") and (stt is None or stt.strip() == ""):
        raise ValueError("No ASR or STT text provided.")

    parts = []

    if asr and asr.strip():
        parts.append("{asr_token}\n" + asr.strip())

    if stt and stt.strip():
        parts.append("{stt_token}\n" + stt.strip())

    # join with newline and append EOS token
    target = "\n".join(parts) + eos_token
    return target
 

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
        audio_token="<extra_id_0>",
        asr_token="<extra_id_1>",
        stt_token="<extra_id_2>",
        sample_rate=16000,
        downsample_ratio=320,
        stack_size=8,
        max_seq_len=1000,
        seed=42,
    ):
        self.tokenizer = tokenizer
        self.audio_token = audio_token
        self.asr_token = asr_token
        self.stt_token = stt_token
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

                prompt = build_prompt(src_lang, tgt_lang, audio_token=self.audio_token, bos_token=self.tokenizer.bos_token)
                prompt_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                ).input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)

                target = build_target(asr, stt, asr_token=self.asr_token, stt_token=self.stt_token, eos_token=self.tokenizer.eos_token)
                target_ids = tokenizer(
                    target,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                ).input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)

                if i % 50000 == 0:
                    # Map prompt_ids to tokens
                    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
                    prompt_mapping = ", ".join(f"{id_}:'{tok}'" for id_, tok in zip(prompt_ids.tolist(), prompt_tokens))

                    # Map target_ids to tokens
                    target_tokens = tokenizer.convert_ids_to_tokens(target_ids)
                    target_mapping = ", ".join(f"{id_}:'{tok}'" for id_, tok in zip(target_ids.tolist(), target_tokens))

                    logger.info(
                        f"sample={i}\n"
                        f"### prompt #######\n{prompt}\n"
                        f"### target #######\n{target}\n"
                        f"### prompt_ids ###\n{prompt_mapping}\n"
                        f"### target_ids ###\n{target_mapping}\n"
                        f"##################"
                    )

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

    sys.exit()
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
