import json
from pathlib import Path
from collections import Counter, defaultdict

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
    translation_langs = Counter()

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
            split_counts[entry.get("split")] += 1

            # Transcription
            if "transcription" in entry:
                t_lang = entry["transcription"].get("lang") or "unknown"
                transcription_langs[t_lang] += 1
                if "text" in entry["transcription"]:
                    txt = entry["transcription"]["text"]
                    text_length_stats["transcription.text"].append(len(txt))
                    if len(txt) == 0:
                        print("Warning: transcription.text length 0 for entry:")
                        print(json.dumps(entry, ensure_ascii=False, indent=2))

            # Translation
            if "translation" in entry:
                # source language is the transcription language when present
                if "transcription" in entry:
                    src_lang = entry["transcription"].get("lang") or "unknown"
                else:
                    src_lang = "unknown"
                tgt_lang = entry["translation"].get("lang") or "unknown"
                pair = f"{src_lang}-{tgt_lang}"
                translation_langs[pair] += 1
                if "text" in entry["translation"]:
                    txt = entry["translation"]["text"]
                    text_length_stats["translation.text"].append(len(txt))
                    if len(txt) == 0:
                        print("Warning: translation.text length 0 for entry:")
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

    print("\nTranslation languages:")
    for pair, count in translation_langs.items():
        print(f"  {pair}: {count}")

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
