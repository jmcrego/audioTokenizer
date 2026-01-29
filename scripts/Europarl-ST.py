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
from scipy.io.wavfile import write
from utils import load_audio_ffmpeg, extract_fragments


def build_segments_dict(segments_path, source_path, target_path):
    """Read segments, source, target files and group by audio_name."""
    segments_dict = defaultdict(list)

    with segments_path.open("r", encoding="utf-8") as f_seg, \
         source_path.open("r", encoding="utf-8") as f_src, \
         target_path.open("r", encoding="utf-8") as f_tgt:

        n_segments = 0
        for seg, src, tgt in zip(f_seg, f_src, f_tgt):
            audio_name, beg, end = seg.strip().split(" ")
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
    m4a_stem2path = {}
    slangs = [p.name for p in base_path.iterdir() if p.is_dir()]
    for slang in slangs:
        audios_path = base_path / slang / "audios"
        for audio_name in audios_path.glob("*.m4a"):
            audio_stem = Path(audio_name).stem
            if audio_stem in m4a_stem2path:
                print(f"repeated entry {audio_stem}")
            m4a_stem2path[audio_stem] = audios_path / audio_name
    print(f"Found {len(set(m4a_stem2path.keys()))} m4a files")
    return m4a_stem2path


def main():
    parser = argparse.ArgumentParser(description="Extract Europarl-ST audio fragments and build TSV.")
    parser.add_argument("--idir", type=str, default="/lustre/fsmisc/dataset/Europarl-ST/v1.1", help="Input path")
    parser.add_argument("--odir", type=str, default="/lustre/fsn1/projects/rech/eut/ujt99zo/josep/datasets", help="Output path")
    args = parser.parse_args()

    base_path = Path(args.idir)
    out_path = Path(args.odir)
    langs = [p.name for p in base_path.iterdir() if p.is_dir()]
    data_sets = ["dev", "test", "train"]

    m4a_stem2path = get_audio_dict(base_path)

    # tsv_file = out_path / f"Europarl-ST_v1.1.tsv"
    # with tsv_file.open("w", encoding="utf-8") as f_tsv:
    json_file = out_path / f"Europarl-ST_v1.1.json"
    with json_file.open("w", encoding="utf-8") as f_json:

        n_entries = 0
        t_entries = 0
        for lsrc, ltgt, data_set in [(s, t, d) for s in langs for t in langs for d in data_sets if s != t]:
            print(f"---------- {lsrc}-{ltgt}:{data_set} ----------")
            segments_path = base_path / lsrc / ltgt / data_set / "segments.lst"
            source_path = base_path / lsrc / ltgt / data_set / f"segments.{lsrc}"
            target_path = base_path / lsrc / ltgt / data_set / f"segments.{ltgt}"

            n_created = 0
            n_exist = 0
            t_audio = 0
            n_skipped = 0
            segments_dict = build_segments_dict(segments_path, source_path, target_path)
            for audio_stem, segments in tqdm(segments_dict.items(), desc=f"Processing {lsrc}-{ltgt}:{data_set}", unit="file"):
                #en.20081117.22.1-112
                #[{'beg': 0.0, 'end': 15.98, 'src': 'Signor Presidente, ...', 'tgt': '. Senhor Presidente, ...'}, ...]

                results, n, m, duration, s = extract_fragments(m4a_stem2path[audio_stem], segments, out_path / "audios" / "Europarl-ST_v1.1")
                n_created += n
                n_exist += m
                t_audio += duration
                n_skipped += s
                #('en.20081117.22.1-112___0.00___15.98.wav', {'beg': 0.0, 'end': 15.98, 'src': 'Signor Presidente, ....', 'tgt': '. Senhor Presidente, ...'})
                for ofile_name, seg in results:
                    out_file = str(out_path / "audios" / "Europarl-ST_v1.1" / ofile_name)
                    # f_tsv.write(f"{out_file}\t{lsrc}\t{seg['src']}\t{ltgt}\t{seg['tgt']}\t{data_set}\n")
                    f_json.write(
                        json.dumps({
                            "audio_file": out_file,
                            "split": data_set,
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
