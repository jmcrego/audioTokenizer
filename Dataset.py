# Dataset.py

import torch
import logging
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger("Dataset")

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
        asr_token="[ASR]",
        stt_token="[STT]",
        end_token="[END]",
        sample_rate=16000,
        downsample_ratio=320,
        stack_size=8,
        max_seq_len=1000,
        seed=42,
    ):
        self.tokenizer = tokenizer
        self.asr_token = asr_token
        self.stt_token = stt_token
        self.end_token = end_token
        self.sample_rate = sample_rate
        self.downsample_ratio = downsample_ratio
        self.stack_size = stack_size
        self.max_seq_len = max_seq_len

        #random seed for reproducibility
        np.random.seed(seed)

        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                audio_path, lang, asr, tgt_lang, stt = parts[:5]
                tgt_lang = None #ATTENTION!!! i only use ASR task
                stt = None #ATTENTION!!! i only use ASR task

                prompt_ids = tokenizer(
                    self.build_prompt(lang, tgt_lang),
                    return_tensors="pt",
                    padding=False,
                    truncation=False
                ).input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)

                target_ids = tokenizer(
                    self.build_target(asr, stt),
                    return_tensors="pt",
                    padding=False,
                    truncation=False
                ).input_ids[0].long() #tensor([ t₁, t₂, t₃, … ], dtype=torch.long)

                audio_time, total_length = self.audio_length_in_tokens(audio_path) + len(prompt_ids) + len(target_ids)
                if total_length > max_seq_len:
                    continue

                self.data.append({
                    "audio_path": audio_path,
                    "prompt_ids": prompt_ids,
                    "target_ids": target_ids,
                    "total_length": total_length,
                    "audio_time": audio_time,
                })
            logger.debug(f"Read dataset {file_path} with {len(self.data)} samples")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


    def build_prompt(self, lang, tgt_lang):
        if lang and tgt_lang:
            return f"\nTranscribe then translate into {tgt_lang}.\n{self.asr_token}"
        elif lang:
            return f"\nTranscribe.\n{self.asr_token}"
        elif tgt_lang:
            return f"\nTranslate into {tgt_lang}.\n{self.stt_token}"
        else:
            raise ValueError("No lang or tgt_lang provided")

    def build_target(self, asr, stt):
        if asr and stt:
            return f"{asr} {self.stt_token} {stt} {self.end_token}"
        elif asr:
            return f"{asr} {self.end_token}"
        elif stt:
            return f"{stt} {self.end_token}"
        else:
            raise ValueError("No ASR or STT text provided")

    def audio_length_in_tokens(self, filepath):
        """
        Estimate number of tokens produced from an audio file
        after audio embedding + frame stacking (no chunking).
        """
        try:
            info = sf.info(filepath)
            if not info.duration:
                return 0, 0

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
            print(f"\tidx={id}\tn_audio={n_audio}, n_prompt={n_prompt}, n_target={n_target}, n_total={e['total_length']}\taudio_time={audio_time}")
