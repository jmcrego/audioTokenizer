import torch
import torchaudio
from torch.utils.data import Dataset, IterableDataset

def get_target(parts, sep_token="<sep>", end_token="<end>"):
    _, _, transcription, _, translation = parts
    ### target options:
    # transcription <end>
    # transcription <sep> translation <end>
    # translation <end>
    target = transcription if transcription else ""
    if translation:
        target += (" " + sep_token + " " + translation) if target else translation
    target += " " + end_token
    return target

def get_waveform(audio_path, sample_rate=40000):
    waveform, sr = torchaudio.load(audio_path)
    # convert to 40kHz if needed
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    return waveform  # [1, T]

class AudioDataset(Dataset):
    """
    Dataset for loading audio-text pairs from a TSV file.
    Each line in the TSV should contain: audio_file_path \t lang \t transcription \t tgt_lang \t translation
    (lang, transcription) and (tgt_lang, translation) can be empty strings if not available (one of them must be present).
    """
    def __init__(self, path, split="train", sep_token="<sep>", end_token="<end>", sample_rate=40000):
        self.sample_rate = sample_rate
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 5:
                    self.samples.append({
                        "audio_path": parts[0], 
                        "lang": parts[1] if parts[1] else None,
                        "tgt_lang": parts[3] if parts[3] else None,
                        "target": get_target(parts, sep_token, end_token)
                    })
                else:
                    raise ValueError(f"Invalid line in {path}: {line}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load audio waveform
        waveform = get_waveform(sample["audio_path"], self.sample_rate)
        # return sample dict
        return {
            "audio": waveform,  # [1, T]
            "lang": sample["lang"],
            "tgt_lang": sample["tgt_lang"],
            "target": sample["target"],
        }
    
class AudioDatasetIterable(IterableDataset):
    """
    Iterable Dataset for loading audio-text pairs from a TSV file.
    Each line in the TSV should contain: audio_file_path \t lang \t transcription \t translation
    transcription and translation can be empty strings if not available (one of them must be present).
    """
    def __init__(self, path, sep_token="<sep>", end_token="<end>", sample_rate=40000):
        self.path = path
        self.sep_token = sep_token
        self.end_token = end_token
        self.sample_rate = sample_rate

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 5:
                    # Load audio waveform
                    waveform = get_waveform(parts[0], self.sample_rate)
                    # return sample dict
                    return {
                        "audio": waveform,  # [1, T]
                        "lang": parts[1] if parts[1] else None,
                        "tgt_lang": parts[3] if parts[3] else None,
                        "target": get_target(parts, self.sep_token, self.end_token),
                    }
                else:
                    raise ValueError(f"Invalid line in {self.path}: {line}")

if __name__ == "__main__":
    # Simple test
    dataset = AudioDataset("data/train.tsv")
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample audio shape: {sample['audio'].shape}")
    print(f"Sample target: {sample['target']}")
