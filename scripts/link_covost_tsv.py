# coding=utf-8                                                                                                                                                                                                                                                                                    
import os
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
    return src_lang, tgt_lang

def read_covost_tsv(tsv_path):
    """
    Read CoVoST TSV with columns:
    path    translation     split
    """
    name2entry = {}

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

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

    # print(f"Found {len(name2entry)} entries in {tsv_path}")
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
    name2entry = read_covost_tsv(args.tsv)
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
    out_path = args.tsv[:-4] + ".linked.tsv"

    seen = set()
    total_linked = 0

    with open(out_path, "w", encoding="utf-8", newline="") as fdo:
        writer = csv.writer(fdo, delimiter="\t")

        for cv_tsv in list(dir_lang.glob("*.tsv")) + list(dir_lang.glob("*.tsv.old")):
            #print(f"Parsing file {cv_tsv}")
            linked_in_file = 0

            with open(cv_tsv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")

                if reader.fieldnames is None:
                    print(f"\tskipping empty file {cv_tsv}")
                    continue

                required_cols = {"path", "sentence"}
                if not required_cols.issubset(reader.fieldnames):
                    print(f"\tskipping bad header file {cv_tsv}")
                    continue

                for row in reader:
                    rel_path = row.get("path", "").strip()
                    transc = row.get("sentence", "").strip()

                    if not rel_path or not transc:
                        continue

                    path1 = Path(args.cv) / src_lang / "clips" / rel_path
                    fname = path1.name

                    if fname in seen:
                        continue

                    if args.verify and not path1.is_file():
                        print(f"\tskipping missing linked file {path1}")
                        continue

                    if fname not in name2entry or fname not in name2path:
                        continue

                    entry = name2entry[fname]
                    transl = entry.get("translation", "").strip()
                    split = entry.get("split", "").strip()

                    if not transl or not split:
                        continue

                    writer.writerow([
                        str(name2path[fname]),
                        src_lang,
                        transc,
                        tgt_lang,
                        transl,
                        split,
                    ])

                    seen.add(fname)
                    linked_in_file += 1
                    total_linked += 1

            print(f"\t{linked_in_file} entries found from {cv_tsv}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    pct = 100.0 * total_linked / max(1, len(name2entry))
    print(
        f"Total {total_linked} out of {len(name2entry)} "
        f"({pct:.2f}%) entries written to {out_path}"
    )

# def main():
#     parser = argparse.ArgumentParser(description="Link CoVoST 2 TSV file with corresponding CommonVoice audio files.")
#     parser.add_argument("--tsv", type=str, required=True, help="tsv file with translations (built by download_covost_tsv.py)")
#     parser.add_argument("--cv", type=str, default="/lustre/fsmisc/dataset/CommonVoice/cv-corpus-22.0-2025-06-20", help="Directory with CommonVoice audio files")
#     parser.add_argument("--verify", action="store_true", help="Verify linked file exists (slows down the script)")
#     args = parser.parse_args()

#     src_lang, tgt_lang = get_langs(args.tsv)
#     print(src_lang, tgt_lang)

#     ### read from args.tsv the valid entries:
#     # name: common_voice_es_19764307.mp3
#     # entry ['common_voice_es_19764307.mp3', 'Lady Faustina, Countess of Benavente, then ordered them to compose a zarzuela.', 'test']
#     name2entry = read_covost_tsv(args.tsv)
#     ### read audio mp3 files from args.csv / src_lang / clips:
#     # name: common_voice_es_19764307.mp3
#     # path /lustre/fsmisc/dataset/CommonVoice/cv-corpus-22.0-2025-06-20/es/20/common_voice_es_19764307.mp3
#     name2path = read_audio_files(Path(args.cv) / src_lang / "clips", name2entry)
#     # Now read CommonVoice TSVs under the source language as indicated by *.tsv{.old} (contain file / lang / transcript)
#     dir_lang = Path(args.cv) / src_lang

#     fdo = open(args.tsv[:-4] + '.linked.tsv', 'w')
#     seen = set()
#     N = 0
#     for cv_tsv in list(dir_lang.glob("*.tsv")) + list(dir_lang.glob("*.tsv.old")): #tsv.old are parsed after tsv files                                                                                                                                                                            
#         n = 0
#         print(f"Parsing file {cv_tsv}")
#         with open(cv_tsv, "r", encoding="utf-8") as f:
#             reader = csv.reader(f, delimiter="\t")
#             try:
#                 header = next(reader)
#             except StopIteration:
#                 print(f"\tskipping empty file {cv_tsv}")
#                 continue

#             if "path" not in header or "sentence" not in header:
#                 print(f"\tskipping bad header file {cv_tsv}")
#                 continue

#             # Expected columns: client_id path sentence_id sentence sentence_domain up_votes down_votes age gender accents variant locale segment
#             for row in reader:
#                 if len(row) < 4:
#                     continue

#                 path1 = Path(args.cv) / src_lang / 'clips' / row[1]
#                 fname = path1.name

#                 #if fname in seen:
#                 if str(fname) in seen:
#                     #print(f"Repeated entry {fname}")
#                     continue

#                 if args.verify and not path1.is_file():
#                     print(f"\tskipping missing linked file {str(path1)}")
#                     continue

#                 transc = row[3]
#                 if not transc:
#                     continue

#                 if fname in name2entry and fname in name2path:
#                     path = name2path[fname]
#                     entry = name2entry[fname]
#                     transl = entry["translation"]
#                     split = entry["split"]

#                     if not transl:
#                         continue
#                     if not split:
#                         continue

#                     fdo.write(
#                         str(path) + '\t' +
#                         src_lang + '\t' +
#                         transc + '\t' +
#                         tgt_lang + '\t' +
#                         transl + '\t' +
#                         split + '\n'
#                     )
#                     n += 1
#                     N += 1
#                     seen.add(fname)

#         if n:
#             print(f"\t{n} entries found from {cv_tsv}")

#     fdo.close()
#     print(f"Total {N} out of {len(name2entry)} ({100*N/len(name2entry):.2f}%) entries left in {args.tsv[:-4] + '.linked.tsv'}")

if __name__ == "__main__":
    main()

