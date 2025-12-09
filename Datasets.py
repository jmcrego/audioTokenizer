
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
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue

                audio_path, lang, asr, tgt_lang, stt = parts[:5]

                # build and tokenize prompt
                prompt_ids = tokenizer(build_prompt(lang, tgt_lang, self.asr_token, self.stt_token), return_tensors="pt", padding=False, truncation=False).input_ids[0]

                # build and tokenize target (labels)
                target_ids = tokenizer(build_target(asr, stt, self.stt_token, self.end_token), return_tensors="pt", padding=False, truncation=False).input_ids[0]

                # compute total length estimating the length in tokens of the audio too
                total_length = self.audio_length_in_tokens(audio_path) + len(prompt_ids) + len(target_ids)

                self.data.append({
                    "audio_path": audio_path,
                    "input_ids": prompt_ids,
                    "labels": target_ids,
                    "total_length": total_length,
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
    dataset = AudioIterableDataset(sys.argv[1], tokenizer)
    for i,e in enumerate(dataset): 
        pass

