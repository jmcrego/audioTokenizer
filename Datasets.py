
import torch
from torch.utils.data import Dataset
import soundfile as sf

def build_prompt(lang, tgt_lang, asr_token, stt_token):
    if lang and tgt_lang:
        prompt = f"\nTranscribe then translate into {tgt_lang}.\n{asr_token}"
    elif lang:
        prompt = f"\nTranscribe.\n{asr_token}"
    elif tgt_lang:
        prompt = f"\nTranslate into {tgt_lang}.\n{stt_token}"
    else:
        raise ValueError("No lang or tgt_lang provided")
    return prompt

def build_target(asr, stt, stt_token, end_token):
    if asr and stt:
        target = f"{asr} {stt_token} {stt} {end_token}"
    elif asr:
        target = f"{asr} {end_token}"
    elif stt:
        target = f"{stt} {end_token}"
    else:
        raise ValueError("No ASR or STT text provided")
    return target

class AudioDataset(Dataset):
    """
    Map-style audio-text dataset with pre-tokenized prompt + target.
    Computes n_audio_tokens from duration so that total_length
    can be used for length bucketing.
    """

    def __init__(
        self,
        path,
        tokenizer,
        asr_token="[ASR]",
        stt_token="[STT]",
        end_token="[END]",
        sample_rate=16000,
        chunk_size=3200,
        stride=1600,
        stack_size=8,
        max_seq_len=1000,
    ):
        self.tokenizer = tokenizer
        self.asr_token = asr_token
        self.stt_token = stt_token
        self.end_token = end_token
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stride = stride
        self.stack_size = stack_size

        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            n_filtered = 0
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue

                audio_path, lang, asr, tgt_lang, stt = parts[:5]

                # build and tokenize prompt
                prompt_ids = tokenizer(build_prompt(lang, tgt_lang, self.asr_token, self.stt_token), return_tensors="pt", padding=False, truncation=False).input_ids[0].long()

                # build and tokenize target (labels)
                target_ids = tokenizer(build_target(asr, stt, self.stt_token, self.end_token), return_tensors="pt", padding=False, truncation=False).input_ids[0].long()

                # compute total length estimating the length in tokens of the audio too
                total_length = self.audio_length_in_tokens(audio_path) + len(prompt_ids) + len(target_ids)

                if total_length > max_seq_len:
                    n_filtered += 1
                    continue

                self.data.append({
                    "audio_path": audio_path, # string, path to audio file
                    "prompt_ids": prompt_ids, # tensor, tokenized prompt
                    "target_ids": target_ids, # tensor, tokenized target/labels
                    "total_length": total_length, # int, n_audio_tokens + len(prompt_ids) + len(target_ids)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def audio_length_in_tokens(self, filepath):
        """
        Given an audio file, this function returns an estimation of the number of tokens after feature extractor/embedding and projector
        """
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

    tokenizer = AutoTokenizer.from_pretrained("/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct", use_fast=True)
    dataset = AudioDataset(sys.argv[1], tokenizer, max_seq_len=100)
    for i,e in enumerate(dataset): 
        n_prompt = len(e["prompt_ids"])
        n_target = len(e["target_ids"])
        n_audio = e["total_length"] - n_prompt - n_target
        print(f"n_audio={n_audio}, n_prompt={n_prompt}, n_target={n_target}, n_total={e['total_length']}")

