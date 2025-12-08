import torch
from torch.utils.data import Dataset, IterableDataset
import soundfile as sf

class AudioIterableDataset(IterableDataset):
    """
    Dataset for loading audio-text pairs from a TSV file.
    Each line in the TSV should contain 5 fields: 
    audio_file_path \t lang \t transcription \t tgt_lang \t translation
    (lang, transcription) and (tgt_lang, translation) can be empty strings if not available (one of them must be present).
    """
    def __init__(self, path, tokenizer, bucket_size=32768, 
                 asr_token="[ASR]", stt_token="[STT]", end_token="[END]", 
                 sample_rate=16000, chunk_size=3200, stride=1600, stack_size=16):
        self.path = path
        self.tokenizer = tokenizer
        self.bucket_size = bucket_size
        self.asr_token = asr_token
        self.stt_token = stt_token
        self.end_token = end_token
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stride = stride
        self.stack_size = stack_size
    
    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            samples = []
            for line in f:
                parts = line.strip().split('\t')

                if len(parts) >= 5:
                    audio, prompt, target = self.get_audio_prompt_and_target(parts)
                    duration, n_audio = self.audio_length_in_tokens(audio)
                    if n_audio is not None:
                        print(f"{audio}, duration={duration}, n_audio={n_audio}, target={self.tokenizer.decode(target)}")
                        length = n_audio + len(prompt) + len(target)
                        samples.append({"length": length, "audio": audio, "prompt": prompt, "target": target}) 
                        if len(samples) >= self.bucket_size:
                            yield from sorted(samples, key=lambda x: x["length"])
                            samples = []
            if len(samples):
                yield from sorted(samples, key=lambda x: x["length"])
                samples = []

    def get_audio_prompt_and_target(self, parts):
        if len(parts) < 5:
            raise ValueError(f"Error: entry must contain at least 5 fields {parts}")

        audio, lang, asr, tgt_lang, stt = parts[:5]
        audio.replace('clip','clips')

        if len(lang) and len(tgt_lang):
            prompt = f"\nTranscribe then translate into {tgt_lang}.\n{self.asr_token}"
        elif len(lang):
            prompt = f"\nTranscribe.\n{self.asr_token}"
        elif len(tgt_lang):
            prompt = f"\nTranslate into {tgt_lang}.\n{self.stt_token}"
        else:
            raise ValueError(f"Either 'lang' or 'tgt_lang' must be provided in the dataset: {parts}")
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False).input_ids[0]

        if len(asr) and len(stt):
            target = f"{asr} {self.stt_token} {stt} {self.end_token}"
        elif len(asr):
            target = f"{asr} {self.end_token}"
        elif len(stt):
            target = f"{stt} {self.end_token}"
        else:
            raise ValueError(f"Either 'lang' or 'tgt_lang' must be provided in the dataset: {parts}")

        target = self.tokenizer(target, return_tensors="pt", padding=False, truncation=False).input_ids[0]

        return audio, prompt, target

    def audio_length_in_tokens(self, filepath, sample_rate=16000, chunk_size=3200, stride=1600, stack_size=8):
        """Get duration of an audio file in seconds, and compute a prediction of number of tokens"""
        try:
            info = sf.info(filepath)
            if info.duration:
                total_samples = int(info.duration * sample_rate)
                n_chunks = max(0, (total_samples - chunk_size) // stride + 1) # number of chunks
                n_tokens = (n_chunks + stack_size - 1) // stack_size  # number of frames/embeddings/tokens
                return info.duration, n_tokens
        except Exception as e:
            return None, None
        return None, None

if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct", use_fast=True)
    dataset = AudioIterableDataset(sys.argv[1], tokenizer)
    for i,e in enumerate(dataset): 
        pass

