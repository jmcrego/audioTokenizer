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


def find_audio_files_by_lang(base_path, lang, file_set, output_dir):
    """
    Search audio files for a given language and write results to a TSV output file.

    Args:
        base_path (str | Path): Root dataset directory.
        lang (str): Language code (e.g., "fr", "en").
        file_set (str): TSV file listing transcript pairs (e.g., "train.tsv").
        output_dir (str | Path): Directory where the output TSV will be written.
    """
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)   # ensure output dir exists

    # compute output filename
    output_file = output_dir / file_set.replace(".tsv", f".{lang}.tsv")

    # All paths
    clips_dir = base_path / lang / "clips"
    input_tsv = base_path / lang / file_set

    # sanity checks
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {clips_dir}")

    if not input_tsv.is_file():
        raise FileNotFoundError(f"Transcript TSV not found: {input_tsv}")

    # find audio paths
    name2path = find_audios(clips_dir)
    path_transc = read_paths(input_tsv, name2path)
    random.shuffle(path_transc)

    # write output file
    with output_file.open("w", encoding="utf-8") as fdo:

        # header line
        fdo.write(f"lang={lang}\tbase_path={base_path}\n")

        total_written = 0

        for path, transc in tqdm(path_transc, desc=f"{lang} files", unit="file"):
            path = Path(path)

            # if not path.is_file():    # skip missing audio
            #     continue

            fdo.write(f"{path}\t{lang}\t{transc}\n")
            total_written += 1

    sys.stderr.write(f"{lang}: total files written {total_written}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find audio files by language in CommonVoice corpus.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-path", type=str, default="/lustre/fsmisc/dataset/CommonVoice/cv-corpus-22.0-2025-06-20", help="Base path for CommonVoice corpus")
    parser.add_argument("--langs", type=str, required=True, help="Comma-separated list of language codes")
    parser.add_argument("--set", type=str, required=True, help="set to use (ex: train.tsv)")
    parser.add_argument("--odir", type=str, default=".", help="Output directory")
    args = parser.parse_args()
    
    for lang in args.langs.split(','):
        audio_files = find_audio_files_by_lang(
            args.base_path, 
            lang, 
            args.set,
            args.odir,
        )