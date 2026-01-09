# coding=utf-8                                                                                                                                                                                                                                                                                    
import os
import io
import sys
import argparse
import tarfile
import urllib.request
from pathlib import Path
from collections import defaultdict
import csv
csv.field_size_limit(sys.maxsize)

def get_langs(file):
    name = os.path.basename(file)
    # Remove prefix
    if name.startswith("covost_v2."):
        name = name[len("covost_v2."):]
    # Remove suffix
    if name.endswith(".tsv"):
        name = name[:-len(".tsv")]
    src_lang, tgt_lang = name.split("_")
    parts = src_lang.split("-")
    src_lang = parts[0] if len(parts) > 1 else src_lang
    parts = tgt_lang.split("-")
    tgt_lang = parts[0] if len(parts) > 1 else tgt_lang
    assert "\t" not in src_lang
    assert "\t" not in tgt_lang
    return src_lang, tgt_lang

def read_covost_tsv(tsv_path):
    """
    Read CoVoST TSV with columns:
    path    translation     split
    """
    name2entry = {}

    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
            # `newline=""` is important for csv.reader to handle embedded newlines
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)

            expected_fields = {"path", "translation", "split"}
            if not expected_fields.issubset(reader.fieldnames):
                raise ValueError(
                    f"Invalid TSV header. Expected {expected_fields}, got {reader.fieldnames}"
                )

            for nrow, row in enumerate(reader, start=2):  # start=2 (line after header)
                path = row["path"].strip()
                translation = row["translation"].strip()
                split = row["split"].strip()

                if not path:
                    continue

                if path in name2entry:
                    raise ValueError(f"Duplicate entry for '{path}' at line {nrow}")

                name2entry[path] = {
                    "translation": translation,
                    "split": split,
                }

    return name2entry

def read_covost_tsv_linewise(tsv_path):
    """
    Robust TSV parser for broken CoVoST files.
    Enforces: one physical line = one entry.
    """
    name2entry = {}

    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t")

            # Expect exactly 3 columns
            if len(parts) != 3:
                # malformed line → skip
                continue

            path, translation, split = parts

            path = path.strip()
            translation = translation.strip()
            split = split.strip()

            if not path or not translation or not split:
                continue

            # Guard against hidden newlines (paranoia)
            if any("\n" in x for x in (path, translation, split)):
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
        name2path[path.name] = path   # path.name = file name (not path)
        #print(f"{path.name} {str(path)}")
    # print(f"Found {len(name2path)} files in {mp3_dir}")
    return name2path

def main():
    parser = argparse.ArgumentParser(description="Link CoVoST 2 TSV file with corresponding CommonVoice audio files.")
    parser.add_argument("--tsv", type=str, required=True, help="TSV file with translations (built by download_covost_tsv.py)")
    parser.add_argument("--cv", type=str, default="/lustre/fsmisc/dataset/CommonVoice/cv-corpus-22.0-2025-06-20", help="Directory with CommonVoice audio files")
    parser.add_argument("--verify", action="store_true", help="Verify linked file exists (slows down the script)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------
    src_lang, tgt_lang = get_langs(args.tsv)
    print(f"Source language: {src_lang}, Target language: {tgt_lang}")

    # ------------------------------------------------------------------
    # Load CoVoST translation table
    # ------------------------------------------------------------------
    name2entry = read_covost_tsv_linewise(args.tsv)
    print(f"Loaded {len(name2entry)} CoVoST entries")

    # ------------------------------------------------------------------
    # Locate CommonVoice audio files
    # ------------------------------------------------------------------
    clips_dir = Path(args.cv) / src_lang / "clips"
    name2path = read_audio_files(clips_dir, name2entry)
    print(f"Resolved {len(name2path)} audio files")

    # ------------------------------------------------------------------
    # Parse CommonVoice TSVs and link
    # ------------------------------------------------------------------
    dir_lang = Path(args.cv) / src_lang
    out_path = args.tsv + ".cv-linked.tsv" #args.tsv[:-4] + ".linked.tsv"

    seen = set()
    total_linked = 0

    with open(out_path, "w", encoding="utf-8", newline="") as fdo:
        writer = csv.writer(fdo, delimiter="\t")

        print(f"Parsing {dir_lang}/*.tsv files")
        for cv_tsv in list(dir_lang.glob("*.tsv")) + list(dir_lang.glob("*.tsv.old")):
            linked_in_file = 0
            n_missing = 0
            with open(cv_tsv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")

                if reader.fieldnames is None:
                    #print(f"\tskipping empty file {cv_tsv}")
                    continue

                required_cols = {"path", "sentence"}
                if not required_cols.issubset(reader.fieldnames):
                    #print(f"\tskipping bad header file {cv_tsv}")
                    continue

                for row in reader:

                    rel_path = row.get("path")
                    if rel_path is None:
                        continue

                    fname = Path(rel_path.strip()).name
                    if fname in seen:
                        continue

                    transc = row.get("sentence")
                    if transc is None:
                        continue

                    path = name2path.get(fname) #Path object or None
                    if path is None:
                        continue

                    entry = name2entry.get(fname)
                    if entry is None:
                        continue

                    transl = entry.get("translation")
                    if transl is None:
                        continue

                    split = entry.get("split")
                    if split is None:
                        continue

                    if "\n" in str(path) or "\n" in transc or "\n" in transl or "\n" in split:
                            #print(f"skipping {cv_tsv} line with \\n:\npath={str(path)}\ntransc={transc}\ntransl={transl}\nsplit={split}")
                            continue

                    if args.verify and not path.is_file():
                        #print(f"\tskipping missing linked file {path}")
                        n_missing += 1
                        continue

                    writer.writerow([
                        str(path),
                        src_lang.strip(),
                        transc.strip(),
                        tgt_lang.strip(),
                        transl.strip(),
                        split.strip(),
                        len(str(path)),
                    ])

                    seen.add(fname)
                    linked_in_file += 1
                    total_linked += 1

            if linked_in_file:
                print(f"\t{linked_in_file} entries found from {cv_tsv}, {n_missing} missing files")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    pct = 100.0 * total_linked / max(1, len(name2entry))
    print(
        f"Total {total_linked} out of {len(name2entry)} "
        f"({pct:.2f}%) entries written to {out_path}"
    )

if __name__ == "__main__":
    main()

