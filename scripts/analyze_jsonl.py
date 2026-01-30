import json
from pathlib import Path
from collections import Counter, defaultdict

def print_matrix(title, matrix):
    """
    Print a matrix (mapping src -> mapping tgt -> count) with sources as rows
    and targets as columns, aligned.
    """
    print(f"\n{title}")
    if not matrix:
        print("  (none)")
        return
    srcs = sorted(matrix.keys())
    tgts = sorted({t for s in matrix for t in matrix[s].keys()})
    if not srcs or not tgts:
        print("  (none)")
        return
    src_w = max(len("src"), max(len(s) for s in srcs))
    # determine width per target column (consider header and max count width)
    tgt_col_w = {}
    for t in tgts:
        max_count_w = max(len(str(matrix[s].get(t, 0))) for s in srcs)
        tgt_col_w[t] = max(len(t), max_count_w)
    # header
    header_cells = ["src".ljust(src_w)] + [t.rjust(tgt_col_w[t]) for t in tgts]
    print("  " + "  ".join(header_cells))
    # separator
    sep_cells = ["-" * src_w] + ["-" * tgt_col_w[t] for t in tgts]
    print("  " + "  ".join(sep_cells))
    # rows
    for s in srcs:
        cells = [s.ljust(src_w)] + [str(matrix[s].get(t, 0)).rjust(tgt_col_w[t]) for t in tgts]
        print("  " + "  ".join(cells))

def analyze_jsonl(input_path):
    """
    Analyze a JSONL file and print dataset statistics.
    """
    #verify input path exists
    input_path = Path(input_path)
    if not input_path.is_file():
        print(f"Input file {input_path} does not exist.")
        return


    top_level_fields = Counter()
    split_counts = Counter()

    transcription_langs = Counter()
    translation_matrix = defaultdict(Counter)
    # per-split translation counts: split -> (src -> Counter(tgt->count))
    translation_by_split = defaultdict(lambda: defaultdict(Counter))

    # counts of empty text lines
    empty_transcription_count = 0
    empty_translation_count = 0

    text_length_stats = defaultdict(list)  # lengths only, no content

    num_entries = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)
            num_entries += 1

            # Top-level fields
            for key in entry.keys():
                top_level_fields[key] += 1

            # Split
            split = entry.get("split") or "none"
            split_counts[split] += 1

            # Transcription
            if "transcription" in entry:
                t_lang = entry["transcription"].get("lang") or "unknown"
                transcription_langs[t_lang] += 1
                if "text" in entry["transcription"]:
                    txt = entry["transcription"]["text"]
                    text_length_stats["transcription.text"].append(len(txt))
                    if len(txt) == 0:
                        empty_transcription_count += 1
                        print(f"Warning: transcription.text length 0 for entry idx {num_entries}:")
                        print(json.dumps(entry, ensure_ascii=False, indent=2))

            # Translation
            if "translation" in entry:
                # source language is the transcription language when present
                if "transcription" in entry:
                    src_lang = entry["transcription"].get("lang") or "unknown"
                else:
                    src_lang = "unknown"
                tgt_lang = entry["translation"].get("lang") or "unknown"
                translation_matrix[src_lang][tgt_lang] += 1
                translation_by_split[split][src_lang][tgt_lang] += 1
                if "text" in entry["translation"]:
                    txt = entry["translation"]["text"]
                    text_length_stats["translation.text"].append(len(txt))
                    if len(txt) == 0:
                        empty_translation_count += 1
                        print(f"Warning: translation.text length 0 for entry idx {num_entries}:")
                        print(json.dumps(entry, ensure_ascii=False, indent=2))

    # Output
    print(f"Total entries: {num_entries}\n")

    print("Top-level fields:")
    for field, count in top_level_fields.items():
        print(f"  {field}: {count}")

    print("\nSplit distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count}")

    print("\nTranscription languages:")
    for lang, count in transcription_langs.items():
        print(f"  {lang}: {count}")

    # Print overall translation matrix
    print_matrix("Translation languages (overall src -> tgt):", translation_matrix)
    # Print per-split translation matrices
    for sp in sorted(translation_by_split.keys()):
        print_matrix(f"Translation languages (split={sp})", translation_by_split[sp])

    # Summary of empty text lines
    print("\nEmpty text lines:")
    print(f"  transcriptions: {empty_transcription_count}")
    print(f"  translations: {empty_translation_count}")

    print("\nText length statistics (characters):")
    for field, lengths in text_length_stats.items():
        if lengths:
            print(
                f"  {field}: "
                f"min={min(lengths)}, "
                f"max={max(lengths)}, "
                f"avg={sum(lengths)/len(lengths):.1f}"
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze JSONL dataset.")
    parser.add_argument("--input", type=str, default="data.jsonl", help="Input JSONL file")
    args = parser.parse_args()

    analyze_jsonl(args.input)
