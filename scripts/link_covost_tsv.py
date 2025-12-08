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
    print(f"{src_lang} {tgt_lang}")
    return src_lang, tgt_lang

def read_covost_tsv(file):
    # Read CoVoST TSV (translation table)
    name2entry = {}

    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

        # Expected columns: path, translation, split
        # Example header: ["path", "translation", "split"]
        try:
            for nrow, row in enumerate(reader):
                print(f"TSV {nrow} => {row}")
                if not row:
                    continue
                file_name = row[0]
                name2entry[file_name] = row
        except Exception as e:
            raise Exception(f"Error on line {nrow}: {e}")


    print(f"Found {len(name2entry)} entries in {file}")
    return name2entry

def main():
    parser = argparse.ArgumentParser(description="Link CoVoST 2 TSV file with corresponding CommonVoice audio files.")
    parser.add_argument("--tsv", type=str, required=True, help="tsv file with translations (built by download_covost_tsv.py)")
    parser.add_argument("--cv", type=str, default="/lustre/fsmisc/dataset/CommonVoice/cv-corpus-22.0-2025-06-20", help="Directory with CommonVoice audio files")
    parser.add_argument("--verify", action="store_true", help="Verify linked file exists (slows down the script)")
    args = parser.parse_args()

    src_lang, tgt_lang = get_langs(args.tsv)
    name2entry = read_covost_tsv(args.tsv)

    # Now read CommonVoice TSVs under the source language
    dir_lang = Path(args.cv) / src_lang

    fdo = open(args.tsv[:-4] + '.linked.tsv', 'w')

    seen = set()
    N = 0
    for cv_tsv in list(dir_lang.glob("*.tsv")) + list(dir_lang.glob("*.tsv.old")): #tsv.old are parsed after tsv files                                                                                                                                                                            
        n = 0
        #print(f"\tParsing file {cv_tsv}")
        with open(cv_tsv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                #print(f"\tskipping empty file {cv_tsv}")
                continue

            if "path" not in header or "sentence" not in header:
                #print(f"\tskipping bad header file {cv_tsv}")
                continue


            # Expected columns: client_id       path    sentence_id     sentence        sentence_domain up_votes        down_votes      age     gender  accents variant locale  segment
            for row in reader:
                if len(row) < 4:
                    continue

                print(row)
                path = Path(args.cv) / src_lang / 'clips' / row[1]
                print(path)
                fname = os.path.basename(path)

                #if path in seen:                                                                                                                                                                                                                                                                 
                if str(fname) in seen:
                    #print(f"Repeated entry {fname}")
                    continue

                if args.verify and not path.is_file():
                    print(f"\tskipping missing linked file {str(path)}")
                    continue

                transc = row[3]

                if fname in name2entry:
                    entry = name2entry[fname]
                    transl = entry[1]
                    split = entry[2]

                    fdo.write(
                        str(path) + '\t' +
                        src_lang + '\t' +
                        transc + '\t' +
                        tgt_lang + '\t' +
                        transl + '\t' +
                        split + '\n'
                    )
                    n += 1
                    N += 1
                    seen.add(fname)
                    #seen.add(str(path))

        if n:
            print(f"\t{n} entries found from {cv_tsv}")

    fdo.close()
    print(f"Total {N} out of {len(name2entry)} ({100*N/len(name2entry):.2f}%) entries left in {args.tsv[:-4] + '.linked.tsv'}")

if __name__ == "__main__":
    main()

