import torch
import torchaudio
from torch.utils.data import Dataset, IterableDataset

def get_target(parts, asr_token="[ASR]", stt_token="[STT]", end_token="[END]"):
    _, _, asr, _, stt = parts
    if asr and not stt:
        return f"{asr_token} {asr} {end_token}"
    elif not asr and stt:
        return f"{stt_token} {stt} {end_token}"
    elif asr and stt:
        return f"{asr_token} {asr} {stt_token} {stt} {end_token}"
    else:
        raise ValueError(f"Invalid dataset entry with neithr asr nor stt: {parts}")

class AudioDataset(Dataset):
    """
    Dataset for loading audio-text pairs from a TSV file.
    Each line in the TSV should contain 5 fields: 
    audio_file_path \t lang \t transcription \t tgt_lang \t translation
    (lang, transcription) and (tgt_lang, translation) can be empty strings if not available (one of them must be present).
    """
    def __init__(self, path, sep_token="<sep>", end_token="<end>"):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.samples = [line for line in f]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        parts = self.samples[idx].strip().split("\t")
        if len(parts) == 5:
            return {
                "audio": parts[0], 
                "lang": parts[1] if parts[1] else None,
                "tgt_lang": parts[3] if parts[3] else None,
                "target": get_target(parts)
            }
        else:
            raise ValueError(f"Invalid line {idx} in {self.path}: {self.samples[idx]}")

if __name__ == "__main__":
    # Simple test
    dataset = AudioDataset("../audioLLM/data/covost_v2.es_en.tsv ")
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample[0]: {sample}")
