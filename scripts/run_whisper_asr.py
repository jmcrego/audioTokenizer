#!/usr/bin/env python3
import argparse
import os
import re
import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm
import librosa
import jiwer
import unicodedata

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dataset import Dataset

class UnicodeNormalize(jiwer.AbstractTransform):
    def process_string(self, s: str):
        return unicodedata.normalize("NFKC", s)

class RemoveTags(jiwer.AbstractTransform):
    def process_string(self, s: str):
        # More robust: handles nested brackets, empty tags                                                                                                                                                                                                                                    
        s = re.sub(r"\<[^>]*\>", "", s)  # Remove <anything>                                                                                                                                                                                                                                  
        s = re.sub(r"\[[^\]]*\]", "", s)  # Remove [anything]                                                                                                                                                                                                                                 
        return s

transform = jiwer.Compose([ 
    UnicodeNormalize(), 
    RemoveTags(), 
    jiwer.ToLowerCase(), 
    jiwer.RemovePunctuation(), 
    jiwer.RemoveWhiteSpace(replace_by_space=True), 
    jiwer.Strip(), 
    jiwer.RemoveEmptyStrings() 
])

def load_dataset(dataset_path):
    """
    Dataset format:
      one audio file path per line
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


@torch.no_grad()
def transcribe_file(model, processor, audio_path, args):
    audio = load_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(args.device)
    generated_ids = model.generate(input_features, language=args.language, task="transcribe", max_new_tokens=256)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Run ASR using HuggingFace Whisper", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium", help="HF Whisper model")
    parser.add_argument("--meta", type=str, default="/lustre/fsn1/projects/rech/eut/ujt99zo/josep/datasets/covost2.test7x10.tsv_asr_cache/meta.json", help="json file (meta.json) with samples containing: audio files/transcriptions")
    parser.add_argument("--dataset", type=str, help="Text file with audio paths (one per line)")
    parser.add_argument("--file_path", type=str, help="Single audio file")
    parser.add_argument("--language", default=None, help="Force language (e.g. en, fr, de)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--output", type=str, default=None, help="Save transcripts")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 and args.device == "cuda" else torch.float32

    if not args.dataset and not args.file_path and not args.meta:
        raise ValueError("You must provide --dataset or --file_path or --meta")

    print(f"Loading model: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    model.eval()

    outputs = []

    if args.file_path:
        print(f"Transcribing: {args.file_path}")
        text = transcribe_file(model, processor, args.file_path, args)
        outputs.append((args.file_path, text))
        print(text)

    elif args.dataset:
        with open(args.dataset, "r", encoding="utf-8") as f:
            audio_files = [line.strip() for line in f if line.strip()]
        print(f"Transcribing {len(audio_files)} files")

        for audio_path in tqdm(audio_files):
            if not os.path.isfile(audio_path):
                print(f"[WARN] Missing file: {audio_path}")
                continue

            text = transcribe_file(model, processor, audio_path, args)
            outputs.append((audio_path, text))
            print(f"{audio_path}\t{text}")

    elif args.meta:
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        samples = meta['samples']
        file_path_dir = Path(args.meta).parent       
        print(f"Transcribing {len(samples)} files")

        hyps = []
        refs = []
        for sample in tqdm(samples):
            audio_path = file_path_dir / sample['audio_path']
            if not audio_path.is_file():
                print(f"[WARN] Missing file: {audio_path}")
                continue

            text = transcribe_file(model, processor, audio_path, args)
            hyps.append(text)
            refs.append(sample['asr'])
            outputs.append((audio_path, text))
            print(f"{str(audio_path)}\t{text}")

        refs_transformed = [ transform(x) or "EMPTY" for x in refs]
        hyps_transformed = [ transform(x) or "EMPTY" for x in hyps]

        # Word-level metrics                                                                                                                                                                                                                                                                          
        word_output = jiwer.process_words(refs_transformed, hyps_transformed)
        print(f"WER: {word_output.wer:.4f}")

        # Character-level metrics                                                                                                                                                                                                                                                                     
        char_output = jiwer.process_characters(refs_transformed, hyps_transformed)
        print(f"CER: {char_output.cer:.4f}")


    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for path, text in outputs:
                f.write(f"{path}\t{text}\n")

        print(f"Saved transcripts to {args.output}")


if __name__ == "__main__":
    main()
