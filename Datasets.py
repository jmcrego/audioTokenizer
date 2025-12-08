import torch
from torch.utils.data import Dataset

def get_target(parts, asr_token="[ASR]", stt_token="[STT]", end_token="[END]"):
    if len(parts) < 5:
        raise ValueError(f"Error: entry must contain at least 5 fields {parts}")
    
    _, _, asr, _, stt = parts[:5]
    if asr and stt:
        return f"{asr} {stt_token} {stt} {end_token}"
    elif asr:
        return f"{asr} {end_token}"
    elif stt:
        return f"{stt} {end_token}"
    else:
        raise ValueError(f"Either 'lang' or 'tgt_lang' must be provided in the dataset: {parts}")

class AudioDataset(Dataset):
    """
    Dataset for loading audio-text pairs from a TSV file.
    Each line in the TSV should contain 5 fields: 
    audio_file_path \t lang \t transcription \t tgt_lang \t translation
    (lang, transcription) and (tgt_lang, translation) can be empty strings if not available (one of them must be present).
    """
    def __init__(self, path, asr_token="[ASR]", stt_token="[STT]", end_token="[END]"):
        self.path = path
        self.asr_token = asr_token
        self.stt_token = stt_token
        self.end_token = end_token
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    self.samples.append(parts[:5])
                else:
                    pass

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        parts = self.samples[idx]
        if len(parts) >= 5:
            return {
                "audio": parts[0], 
                "lang": parts[1] if parts[1] else None,
                "tgt_lang": parts[3] if parts[3] else None,
                "target": get_target(parts, asr_token=self.asr_token, stt_token=self.stt_token, end_token=self.end_token)
            }
        else:
            raise ValueError(f"Invalid line {idx} in {self.path}: {self.samples[idx]}")

if __name__ == "__main__":
    import sys
    # Simple test
    dataset = AudioDataset(sys.argv[1])
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample[0]: {sample}")
