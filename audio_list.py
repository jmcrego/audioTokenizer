import argparse
import sys
import random
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

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
        fdo.write(f"max_files_lang={max_files_lang}\n")
        fdo.write(f"min_duration_file={min_duration_file}\n")
        for lang in langs:
            sys.stderr.write(f"Processing language: {lang}\n")
            lang_path = Path(base_path.replace('LANG', lang))
            total_duration = 0
            
            if not lang_path.exists():
                sys.stderr.write(f"Warning: Path does not exist for language {lang}: {lang_path}\n")
                continue
            
            # Find all .mp3 files
            files = list(lang_path.rglob('*.mp3'))
            sys.stderr.write(f"Found {len(files)} files\n")
            random.shuffle(files)
            
            if not files:
                continue

            bar = tqdm(total=max_files_lang or len(files), desc=f"{lang} files", unit=" file")
            n_files = 0
            for filepath in files:
                filepath, duration = get_audio_duration(filepath)
                if duration is not None:
                   if min_duration_file is None or duration > min_duration_file:
                        total_duration += duration
                fdo.write(f"{lang}\t{duration:.2f}\t{filepath}\n")
                bar.update(1)
                n_files += 1
                if max_files_lang is not None and n_files >= max_files_lang:
                    break


            sys.stderr.write(f"Lang {lang}, Total files {n_files}, Total duration {total_duration:.2f}s ({total_duration/3600:.2f}h)\n")
            fdo.write(f"{lang} TotalDuration={total_duration/3600:.2f}h\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find audio files by language and compute durations")
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