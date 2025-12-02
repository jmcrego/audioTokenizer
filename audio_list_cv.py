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

def find_audio_files_by_lang(base_path, lang, file_set):
    fout = file_set.replace('.tsv',f'.{lang}.tsv')
    with open(fout, 'w') as fdo:
        fdo.write(f"lang={lang} base_path={base_path}\n")

        name2path = find_audios(Path(base_path) / lang / 'clips')
        path_transc = read_paths(Path(base_path) / lang / file_set, name2path)
        random.shuffle(path_transc)

        total_files = 0
        bar = tqdm(total=len(path_transc), desc=f"{lang} files", unit=" file")
        for path, transc in path_transc:
            if not Path(path).is_file():
                continue
            fdo.write(f"{lang}\t{path}\t{transc}\n")
            bar.update(1)
            total_files += 1

        # force close the bar
        bar.close()
        sys.stderr.write(f"{lang}: total files {total_files}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find audio files by language in CommonVoice corpus.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-path", type=str, default="/lustre/fsmisc/dataset/CommonVoice/cv-corpus-22.0-2025-06-20", help="Base path for CommonVoice corpus")
    parser.add_argument("--langs", type=str, required=True, help="Comma-separated list of language codes")
    parser.add_argument("--set", type=str, required=True, help="set to use (ex: train.tsv)")
    args = parser.parse_args()
    
    for lang in args.langs.split(','):
        audio_files = find_audio_files_by_lang(
            args.base_path, 
            lang, 
            args.set,
        )