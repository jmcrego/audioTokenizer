import soundfile as sf
import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from torch.utils.data import Sampler
import numpy as np

class BucketedLengthSampler(Sampler):
    def __init__(self, dataset, batch_size, bucket_size=1000, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        # Extract total_length for each sample
        self.lengths = np.array([s["total_length"] for s in dataset])
        self.sorted_indices = np.argsort(self.lengths)

    def __iter__(self):
        # Split sorted indices into buckets
        buckets = [
            self.sorted_indices[i:i+self.bucket_size]
            for i in range(0, len(self.sorted_indices), self.bucket_size)
        ]

        all_indices = []
        for b in buckets:
            if self.shuffle:
                b = np.random.permutation(b)
            all_indices.extend(b.tolist())

        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i+self.batch_size]

    def __len__(self):
        return len(self.dataset)

import torch
from torch.utils.data import Dataset
import soundfile as sf

class AudioDataset(Dataset):
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
        chunk_size=3200,
        stride=1600,
        stack_size=8,
        max_seq_len=1000
    ):
        self.tokenizer = tokenizer
        self.asr_token = asr_token
        self.stt_token = stt_token
        self.end_token = end_token
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stride = stride
        self.stack_size = stack_size
        self.max_seq_len = max_seq_len

        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                audio_path, lang, asr, tgt_lang, stt = parts[:5]

                prompt_ids = tokenizer(
                    self.build_prompt(lang, tgt_lang),
                    return_tensors="pt",
                    padding=False,
                    truncation=False
                ).input_ids[0].long()

                target_ids = tokenizer(
                    self.build_target(asr, stt),
                    return_tensors="pt",
                    padding=False,
                    truncation=False
                ).input_ids[0].long()

                total_length = self.audio_length_in_tokens(audio_path) + len(prompt_ids) + len(target_ids)
                if total_length > max_seq_len:
                    continue

                self.data.append({
                    "audio_path": audio_path,
                    "prompt_ids": prompt_ids,
                    "target_ids": target_ids,
                    "total_length": total_length,
                    "text": ""  # dummy for SFTTrainer
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

    # --- helper functions ---
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
        """Estimate number of tokens for audio after embedding/projector"""
        try:
            info = sf.info(filepath)
            if not info.duration:
                return 0
            total_samples = int(info.duration * self.sample_rate)
            n_chunks = max(0, (total_samples - self.chunk_size) // self.stride + 1)
            n_tokens = (n_chunks + self.stack_size - 1) // self.stack_size
            return n_tokens
        except:
            return 0


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct",
        use_fast=True
    )

    # Create dataset from file
    eval_dataset = AudioDataset(
        file_path=sys.argv[1],
        tokenizer=tokenizer,
        max_seq_len=50
    )

    # Print dummy column names compatible with SFTTrainer
    # We'll assume the same columns as in build_dataset
    column_names = ["audio_path", "prompt_ids", "target_ids", "total_length", "text"]
    print(column_names)

    # Inspect some samples
    for i, e in enumerate(eval_dataset):
        n_prompt = len(e["prompt_ids"])
        n_target = len(e["target_ids"])
        n_audio = e["total_length"] - n_prompt - n_target
        print(f"n_audio={n_audio}, n_prompt={n_prompt}, n_target={n_target}, n_total={e['total_length']}")
