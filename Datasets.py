import soundfile as sf
import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

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

def audio_length_in_tokens(filepath, sample_rate=16000, chunk_size=3200, stride=1600, stack_size=8):
    """Estimate number of tokens for audio after embedding/projector"""
    try:
        info = sf.info(filepath)
        if not info.duration:
            return 0
        total_samples = int(info.duration * sample_rate)
        n_chunks = max(0, (total_samples - chunk_size) // stride + 1)
        n_tokens = (n_chunks + stack_size - 1) // stack_size
        return n_tokens
    except:
        return 0

def build_dataset(file_path: str,
                  tokenizer: PreTrainedTokenizerBase,
                  asr_token="[ASR]",
                  stt_token="[STT]",
                  end_token="[END]",
                  sample_rate=16000,
                  chunk_size=3200,
                  stride=1600,
                  stack_size=8,
                  max_seq_len=1000) -> Dataset:
    """Build a Hugging Face Dataset from a tab-separated file"""

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            audio_path, lang, asr, tgt_lang, stt = parts[:5]

            # tokenize prompt
            prompt_ids = tokenizer(
                build_prompt(lang, tgt_lang, asr_token, stt_token),
                return_tensors="pt",
                padding=False,
                truncation=False
            ).input_ids[0].long()

            # tokenize target
            target_ids = tokenizer(
                build_target(asr, stt, stt_token, end_token),
                return_tensors="pt",
                padding=False,
                truncation=False
            ).input_ids[0].long()

            total_length = audio_length_in_tokens(
                audio_path, sample_rate, chunk_size, stride, stack_size
            ) + len(prompt_ids) + len(target_ids)

            if total_length > max_seq_len:
                continue

            data.append({
                "audio_path": audio_path,
                "prompt_ids": prompt_ids.tolist(),  # convert tensor -> list for Dataset
                "target_ids": target_ids.tolist(),
                "total_length": total_length
            })

    return Dataset.from_list(data)


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct",
        use_fast=True
    )

    eval_dataset = build_dataset(sys.argv[1], tokenizer, max_seq_len=50)

    # Now train_dataset.column_names will exist, compatible with SFTTrainer
    print(eval_dataset.column_names)
    for i,e in enumerate(eval_dataset): 
        n_prompt = len(e["prompt_ids"])
        n_target = len(e["target_ids"])
        n_audio = e["total_length"] - n_prompt - n_target
        print(f"n_audio={n_audio}, n_prompt={n_prompt}, n_target={n_target}, n_total={e['total_length']}")
