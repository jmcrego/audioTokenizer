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
        return info.duration
    except Exception as e:
        sys.stderr.write(f"Error reading {filepath}: {e}\n")
        return None

def find_audios(path):
    name2path = {}
    if not path.exists():
        sys.stderr.write(f"Warning: Path does not exist: {path}\n")
        return name2path  
    # Find recursively all .mp3 files
    for file in list(path.rglob('*.mp3')):
        name2path[Path(file).name] = file
    sys.stderr.write(f"Found {len(name2path)} audio files in {path}\n")
    return name2path

def read_paths(path, name2path):
    path_transc = []
    with open(str(path), 'r') as fdi:
        for i, l in enumerate(fdi):
            if i==0:
                continue
            parts = l.strip().split('\t')
            if len(parts) < 3:
                continue
            name = parts[1]
            name = name.split('/')[-1]
            if name not in name2path:
                continue
            transc = parts[3].strip()
            if len(transc) == 0:
                continue
            path_transc.append((name2path[name], transc))

    sys.stderr.write(f"Found {len(path_transc)} files in {path}\n")
    return path_transc

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
        total_duration = 0
        total_files = 0
        for lang in langs:
            name2path = find_audios(Path(base_path) / lang / 'clips')
            path_transc = read_paths(Path(base_path) / lang / 'train.tsv', name2path)
            random.shuffle(path_transc)

            total_lang_duration = 0
            total_lang_files = 0
            bar = tqdm(total=max_files_lang or len(path_transc), desc=f"{lang} files", unit=" file")
            for path, transc in path_transc:
                if min_duration_file is not None:
                    duration = get_audio_duration(path)
                    if duration is None:
                        continue
                    if min_duration_file is not None and duration < min_duration_file:
                        continue
                else:
                    duration = '-'
                fdo.write(f"{lang}\t{duration:.2f}\t{path}\t{transc}\n")
                bar.update(1)
                total_lang_duration += duration
                total_lang_files += 1
                if max_files_lang is not None and total_lang_files >= max_files_lang:
                    break

            # force close the bar
            bar.close()

            total_files += total_lang_files
            total_duration += total_lang_duration

            sys.stderr.write(f"{lang}: total files {total_lang_files}, total duration {total_lang_duration:.2f}s ({total_lang_duration/3600:.2f}h)\n")
            fdo.write(f"{lang}\tTotalDuration={total_lang_duration/3600:.2f}h TotalFiles={total_lang_files}\n")
    
        sys.stderr.write(f"total files {total_files}, total duration {total_duration:.2f}s ({total_duration/3600:.2f}h)\n")
        fdo.write(f"TotalDuration={total_duration/3600:.2f}h TotalFiles={total_files}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find audio files by language and compute durations.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-path", type=str, default="/lustre/fsmisc/dataset/CommonVoice/cv-corpus-22.0-2025-06-20", help="Base path for CommonVoice corpus")
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