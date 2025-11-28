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

def find_audio_files_by_lang(base_path, langs, max_files_lang, output):
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

            if max_files_lang is not None and len(files) > max_files_lang:
                random.shuffle(files)
                files = files[:max_files_lang]
                sys.stderr.write(f"Shuffled and Kept {len(files)} files\n")
            
            if not files:
                continue
            
            for filepath in tqdm(files, total=len(files), desc=f"{lang} files", unit=" file"):
                filepath, duration = get_audio_duration(filepath)
                if duration is not None:
                    total_duration += duration
                fdo.write(f"{lang}\t{duration:.2f}\t{filepath}\n")

            sys.stderr.write(f"Lang {lang}, Total files {len(files)}, Total duration for {lang}: {total_duration:.2f}s ({total_duration/3600:.2f}h)\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find audio files by language and compute durations")
    parser.add_argument("--base-path", type=str, required=True, help="Base path with LANG placeholder")
    parser.add_argument("--langs", type=str, required=True, help="Comma-separated list of language codes")
    parser.add_argument("--max-files-lang", type=int, default=None, help="Maximum number of files per language")
    parser.add_argument("--output", type=str, required=True, help="Output file")
    args = parser.parse_args()
    
    print("Starting audio file search and duration computation...")
    audio_files = find_audio_files_by_lang(
        args.base_path, 
        args.langs, 
        args.max_files_lang,
        args.output,
    )