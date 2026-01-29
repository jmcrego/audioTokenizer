# coding=utf-8                                                                                                                                                                                                                                                                                    
import os
import io
import sys
import json
import argparse
import tarfile
import urllib.request
from pathlib import Path
from collections import defaultdict
import csv
csv.field_size_limit(sys.maxsize)

ALLOWED = {"test.tsv", "dev.tsv", "train.tsv", "validated.tsv", "other.tsv", "test.tsv.old", "dev.tsv.old", "train.tsv.old", "validated.tsv.old", "other.tsv.old"}

def clean_field(s: str) -> str:
    if not s:
        return ""
    return (
        s.replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .strip()
    )

def read_covost_tsv(tsv_path):
    """
    TSV parser for CoVoST files.
    """
    name2entry = {}

    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            if not line: # skip empty lines
                continue

            parts = line.split("\t")
            if len(parts) != 3: # Expect exactly 3 columns
                continue

            path, translation, split = parts

            path = path.strip()
            translation = translation.strip()
            split = split.strip()

            if not path or not translation or not split: # skip incomplete entries
                continue

            if any("\n" in x for x in (path, translation, split)): # Guard against hidden newlines
                continue

            name2entry[path] = {
                "translation": translation,
                "split": split,
            }

    return name2entry

def read_audio_files(mp3_dir, name2entry):
    name2path = {}
    for path in mp3_dir.rglob("*.mp3"):
        if path.name in name2path:
            continue
        if path.name not in name2entry:
            continue
        name2path[path.name] = path 
    return name2path


def main():
    parser = argparse.ArgumentParser(description="Link CoVoST TSV file with corresponding CommonVoice audio files.")
    parser.add_argument("--c2", type=str, default="./data/covost2", help="Directory where CoVoST TSV files with translations (built by download_covost_tsv.py)")
    parser.add_argument("--cv", type=str, default="/lustre/fsmisc/dataset/CommonVoice/cv-corpus-22.0-2025-06-20", help="Directory with CommonVoice audio files")
    parser.add_argument("--verify", action="store_true", help="Verify linked file exists (slows down the script)")
    args = parser.parse_args()

    c2_tsv_files = list(Path(args.c2).glob("covost_v2.??_??.tsv"))
    print(f"Found {len(c2_tsv_files)} CoVoST TSV files in {args.c2}")

    # ------------------------------------------------------------------
    # Parse CommonVoice TSVs and link to CoVoST entries
    # ------------------------------------------------------------------

    out_path = Path(args.c2) / "covost_v2.jsonl"
    with open(out_path, "w", encoding="utf-8") as fdo:
        total_linked = 0

        for c2_tsv_file in c2_tsv_files:
            src_lang = c2_tsv_file.name.split(".")[1].split("_")[0]
            tgt_lang = c2_tsv_file.name.split(".")[1].split("_")[1]

            # ------------------------------------------------------------------
            # Load CoVoST translation table
            # ------------------------------------------------------------------
            c2_name2entry = read_covost_tsv(c2_tsv_file)
            print(f" - Loaded {len(c2_name2entry)} CoVoST entries from {c2_tsv_file}")

            # ------------------------------------------------------------------
            # Locate ALL CommonVoice audio files given src_lang
            # ------------------------------------------------------------------
            clips_dir = Path(args.cv) / src_lang / "clips"
            cv_name2path = read_audio_files(clips_dir, c2_name2entry)
            print(f" + Resolved {len(cv_name2path)} audio files from {clips_dir}")

            json_lines = []

            dir_lang = Path(args.cv) / src_lang
            for cv_tsv in list(dir_lang.glob("*.tsv")) + list(dir_lang.glob("*.tsv.old")):

                if cv_tsv.name not in ALLOWED:
                    continue

                print(f"\tParsing {cv_tsv}")
                seen = set()
                n_missing = 0
                n_errors = 0
                n_repeated = 0
                with open(cv_tsv, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")

                    if reader.fieldnames is None:
                        n_errors += 1
                        continue

                    required_cols = {"path", "sentence"}
                    if not required_cols.issubset(reader.fieldnames):
                        n_errors += 1
                        continue

                    for row in reader:

                        rel_path = row.get("path")
                        if rel_path is None:
                            n_errors += 1
                            continue

                        fname = Path(rel_path.strip()).name
                        if fname in seen:
                            n_repeated += 1
                            continue

                        transc = row.get("sentence")
                        if transc is None:
                            n_errors += 1
                            continue

                        path = cv_name2path.get(fname) #Path object or None
                        if path is None:
                            n_errors += 1
                            continue

                        entry = c2_name2entry.get(fname)
                        if entry is None:
                            n_errors += 1
                            continue

                        transl = entry.get("translation")
                        if transl is None:
                            n_errors += 1
                            continue

                        split = entry.get("split")
                        if split is None:
                            n_errors += 1
                            continue

                        if "\n" in str(path) or "\n" in transc or "\n" in transl or "\n" in split:
                            #print(f"skipping {cv_tsv} line with \\n:\npath={str(path)}\ntransc={transc}\ntransl={transl}\nsplit={split}")
                            n_errors += 1
                            continue

                        if args.verify and not path.is_file():
                            print(f"\tskipping missing linked file {path}")
                            n_missing += 1
                            continue

                        # write to jsonl file
                        json_lines.append({
                            "audio_file": str(clean_field(str(path))),
                            "split": clean_field(split),
                            "transcription": {
                                "lang": clean_field(src_lang),
                                "text": clean_field(transc),
                            },
                            "translation": {
                                "lang": clean_field(tgt_lang),
                                "text": clean_field(transl),
                            },
                        })

                        seen.add(fname)
                        total_linked += 1

            print(f"\t{len(json_lines)} entries found from {cv_tsv}, errors={n_errors} repeated={n_repeated} missing={n_missing} entries")

            print(json.dumps(json_lines, ensure_ascii=False), file=fdo)

            # ------------------------------------------------------------------
            # Summary
            # ------------------------------------------------------------------
            pct = 100.0 * total_linked / max(1, len(c2_name2entry))
            print(f"Total {total_linked} out of {len(c2_name2entry)} ({pct:.2f}%) entries written to {out_path}")

if __name__ == "__main__":
    main()

