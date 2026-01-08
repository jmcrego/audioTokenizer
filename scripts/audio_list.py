import argparse
import sys
import random
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

code2lang={
    "fr": "French",
    "ca": "Catalan",
    "de": "German",
    "es": "Spanish",
    "en": "English",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese",
    "zh=CN": "Chinese"
}

def get_audio_duration(filepath):
    """Get duration of an audio file in seconds."""
    try:
        info = sf.info(filepath)
        return filepath, info.duration
    except Exception as e:
        sys.stderr.write(f"Error reading {filepath}: {e}\n")
        return filepath, None

def find_audio_files_by_lang(base_path, langs, max_files_lang, min_duration_file, output):
    """
    Find audio files by language and compute their durations efficiently.
    Args:
        base_path (str): Base path with LANG placeholder
        langs (str or list): Comma-separated string or list of language codes
    
    Returns:
        dict: Nested dictionary {lang: {path/filename: duration}}
    """
    if isinstance(langs, str):
        langs = [lang.strip() for lang in langs.split(',')]

    with open(output, 'w') as fdo:
        fdo.write(f"base_path={base_path}\n")
        fdo.write(f"langs={langs}\n")
        # fdo.write(f"max_files_lang={max_files_lang}\n")
        # fdo.write(f"min_duration_file={min_duration_file}\n")
        # total_duration = 0
        total_files = 0
        for lang in langs:
            lang_path = Path(base_path.replace('LANG', lang))
            
            if not lang_path.exists():
                sys.stderr.write(f"Warning: Path does not exist for language {lang}: {lang_path}\n")
                continue
            
            # Find all .mp3 files
            files = list(lang_path.rglob('*.mp3'))
            sys.stderr.write(f"{lang}: found {len(files)} files\n")
            random.shuffle(files)
            
            if not files:
                continue

            bar = tqdm(total=len(files), desc=f"{lang} files", unit=" file")
            # total_lang_duration = 0
            total_lang_files = 0
            for filepath in files:
                # filepath, duration = get_audio_duration(filepath)

                # if duration is None:
                #     continue

                # if min_duration_file is not None and duration < min_duration_file:
                #     continue

                # fdo.write(f"{lang}\t{duration:.2f}\t{filepath}\n")
                fdo.write(f"{code2lang[lang]}\t{filepath}\n")
                bar.update(1)
                total_lang_files += 1
                # total_lang_duration += duration

                # if max_files_lang is not None and total_lang_files >= max_files_lang:
                #     break

            # force close the bar
            bar.close()

            total_files += total_lang_files
            # total_duration += total_lang_duration

            # sys.stderr.write(f"{lang}: total files {total_lang_files}, total duration {total_lang_duration:.2f}s ({total_lang_duration/3600:.2f}h)\n")
            # fdo.write(f"{lang}\tTotalDuration={total_lang_duration/3600:.2f}h TotalFiles={total_lang_files}\n")
    
        # sys.stderr.write(f"total files {total_files}, total duration {total_duration:.2f}s ({total_duration/3600:.2f}h)\n")
        # fdo.write(f"TotalDuration={total_duration/3600:.2f}h TotalFiles={total_files}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find audio files by language and compute durations.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-path", type=str, required=True, help="Base path with LANG placeholder")
    parser.add_argument("--langs", type=str, required=True, help="Comma-separated list of language codes")
    parser.add_argument("--max-files-lang", type=int, default=None, help="Maximum number of files per language")
    parser.add_argument("--min-duration-file", type=float, default=None, help="Minimum duration for a file (seconds)")
    parser.add_argument("--output", type=str, required=True, help="Output file")
    args = parser.parse_args()
    
    print("Starting audio file search and duration computation...")
    audio_files = find_audio_files_by_lang(
        args.base_path, 
        args.langs, 
        args.max_files_lang,
        args.min_duration_file,
        args.output,
    )