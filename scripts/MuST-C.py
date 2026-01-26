import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import subprocess
import numpy as np
import shutil
import json
import soxr 
import os
import re
import sys
from scipy.io.wavfile import write
from utils import load_audio_ffmpeg, extract_fragments


pattern = re.compile(
    r"duration:\s*([0-9.]+).*?"
    r"offset:\s*([0-9.]+).*?"
    r"wav:\s*([^}\s]+)"
)

def parse_line(line: str) -> dict:
    line = line.strip().lstrip("- ").strip("{}")
    m = pattern.search(line)
    if not m:
        raise ValueError(f"Cannot parse line: {line}")
    duration = float(m.group(1))
    beg = float(m.group(2))
    audio_name = m.group(3)
    return beg, beg + duration, audio_name

def build_segments_dict(segments_path, source_path, target_path):
    """Read segments, source, target files and group by audio_name."""
    segments_dict = defaultdict(list)

    with segments_path.open("r", encoding="utf-8") as f_seg, \
         source_path.open("r", encoding="utf-8") as f_src, \
         target_path.open("r", encoding="utf-8") as f_tgt:

        n_segments = 0
        for seg, src, tgt in zip(f_seg, f_src, f_tgt):
            print(seg)
            beg, end, audio_name = parse_line(seg)
            segments_dict[audio_name].append({
                "beg": float(beg),
                "end": float(end),
                "src": src.strip(),
                "tgt": tgt.strip()
            })
            n_segments += 1

        print(f"Found {n_segments} segments in {len(segments_dict)} audio files")
        
    return segments_dict


def get_audio_dict(base_path):
    print(base_path)
    wav_stem2path = {}
    for audio_name in base_path.glob("*.wav"):
        audio_stem = Path(audio_name).stem
        if audio_stem in wav_stem2path:
            print(f"repeated entry {audio_stem}")
        wav_stem2path[audio_stem] = base_path / audio_name
    print(f"Found {len(set(wav_stem2path.keys()))} wav files")

    return wav_stem2path


def main():
    parser = argparse.ArgumentParser(description="Extract MuST-C audio fragments and build TSV.")
    parser.add_argument("--idir", type=str, default="/lustre/fsmisc/dataset/MUST-C", help="Input path")
    parser.add_argument("--odir", type=str, default="/lustre/fsn1/projects/rech/eut/ujt99zo/josep/datasets", help="Output path")
    args = parser.parse_args()

    base_path = Path(args.idir)
    out_path = Path(args.odir)

    lang_pairs = {tuple(p.name.split("-")) for p in base_path.iterdir() if p.is_dir() and len(p.name.split("-")) == 2 and all(len(x) == 2 for x in p.name.split("-"))}
    data_sets = ["dev", "tst-COMMON", "tst-HE", "train"]

    # tsv_file = out_path / f"MuST-C.tsv"
    # with tsv_file.open("w", encoding="utf-8") as f_tsv:
    json_file = out_path / f"MuST-C.json"
    with json_file.open("w", encoding="utf-8") as f_json:

        n_entries = 0
        t_entries = 0
        for lsrc, ltgt in lang_pairs:
            if lsrc == ltgt:
                continue

            for data_set in data_sets:
                print(f"---------- {lsrc}-{ltgt}:{data_set} ----------")
                source_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"{data_set}.{lsrc}"
                target_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"{data_set}.{ltgt}"
                segments_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"{data_set}.yaml"

                wav_stem2path = get_audio_dict(base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "wav")
                print(wav_stem2path)

                n_created = 0
                n_exist = 0
                t_audio = 0
                n_skipped = 0

                segments_dict = build_segments_dict(segments_path, source_path, target_path)

                for audio_stem, segments in tqdm(segments_dict.items(), desc=f"Processing {lsrc}-{ltgt}:{data_set}", unit="file"):
                    results, n, m, duration, s = extract_fragments(wav_stem2path[audio_stem], segments, out_path / "audios" / "MuST-C")
                    n_created += n
                    n_exist += m
                    t_audio += duration
                    n_skipped += s

                    for ofile_name, seg in results:
                        out_file = str(out_path / "audios" / "MuST-C" / ofile_name)
                        # f_tsv.write(f"{out_file}\t{lsrc}\t{seg['src']}\t{ltgt}\t{seg['tgt']}\t{data_set}\n")
                        f_json.write(
                            json.dumps({
                                "audio_file": out_file,
                                "set": data_set,
                                "transcription": {
                                    "lang": lsrc, 
                                    "text": seg['src']
                                },
                                "translation": {
                                    "lang": ltgt,
                                    "text": seg['tgt']
                                }
                            }, ensure_ascii=False) + "\n"
                        )
                print(f"Created {n_created} files ({n_exist} existing), total duration {t_audio:.1f} secs, skipped {n_skipped} segments")

            n_entries += n_created + n_exist
            t_entries += t_audio

    print(f"Total entries {n_entries}")
    print(f"Total duration {t_entries:.1f} secs ({t_entries/n_entries:.1f} secs/file)")

if __name__ == "__main__":
    main()
