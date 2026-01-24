import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import subprocess
import numpy as np
import shutil
import soxr 
import os
from scipy.io.wavfile import write

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    raise RuntimeError("ffmpeg not found in PATH")


def load_audio_ffmpeg_aligned(path, sample_rate=None, channel=-1, norm=True):
    """
    Load audio using FFmpeg to mimic preprocess_audio behavior.
    Returns float32 numpy array in [-1,1], mono if needed, optionally resampled.
    """
    cmd = [
        "ffmpeg", "-i", str(path), "-f", "f32le",
        "-ac", "2",  # keep stereo initially for channel handling
        "-acodec", "pcm_f32le", "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)

    # Parse original sample rate from ffmpeg stderr
    sr = None
    for line in proc.stderr.decode().splitlines():
        if "Hz" in line:
            sr = int(line.split("Hz")[0].split()[-1])
            break
    if sr is None:
        raise RuntimeError("Could not determine sample rate from FFmpeg")

    # Reshape stereo
    if wav.size % 2 == 0:
        wav = wav.reshape((-1, 2))
        if channel == -1:
            wav = wav.mean(axis=1)
        elif channel < 2:
            wav = wav[:, channel]

    # Optional resampling
    if sample_rate is not None and sr != sample_rate:
        wav = soxr.resample(wav, sr, sample_rate)
        sr = sample_rate

    # Optional normalization
    if norm:
        wav /= np.max(np.abs(wav)) + 1e-9

    return wav, sr


def load_audio_ffmpeg(path):
    # This version :
    # Does not resample (leaves audio as it is)
    # Mono forced immediately (mixes everything into mono automatically)
    # No normalization, ffmpeg outputs float32 PCM, usually in [-1,1], but no guarantee your pipeline does dynamic normalization like preprocess_audio.

    cmd = [ FFMPEG, "-i", str(path), "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1" ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Parse sample rate from ffmpeg stderr
    sr = None
    for line in proc.stderr.decode().splitlines():
        if "Hz" in line:
            sr = int(line.split("Hz")[0].split()[-1])
            break
    if sr is None:
        raise RuntimeError("Could not determine sample rate")

    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    return wav, sr


def extract_fragments(ifile_path, segments, audio_out_path):
    """
    Load an audio file once, slice all fragments in memory, and write them out.
    segments: list of dicts with keys 'beg', 'end', 'src', 'tgt'
    Returns: list of (ofile_name, segment_dict)
    """

    if not ifile_path.exists():
        print(f"Missing input audio: {ifile_path}")
        return []

    segments.sort(key=lambda s: s["beg"])

    ### check if all output segments are already in place

    all_segments_exist = True
    for seg in segments:

        duration_sec = seg["end"] - seg["beg"]

        if duration_sec <= 0:
            print(f"Skipping invalid segment {seg} in {ifile_path}")
            continue

        if duration_sec > 30.0:
            print(f"Skipping long segment {seg} in {ifile_path}")
            continue

        ofile_name = f"{ifile_path.stem}___{seg['beg']:.2f}___{seg['end']:.2f}.wav"
        ofile_path = audio_out_path / ofile_name

        if not ofile_path.exists():
            all_segments_exist = False
            break

    if not all_segments_exist:
        try:
            wav, sample_rate = load_audio_ffmpeg(ifile_path)
        except Exception as e:
            print(f"Failed to read {ifile_path}: {e}")
            return []
    
    results = []
    n_exist = 0
    n_created = 0
    for seg in segments:

        duration_sec = seg["end"] - seg["beg"]

        if duration_sec <= 0:
            print(f"Skipping invalid segment {seg} in {ifile_path}")
            continue

        if duration_sec > 30.0:
            print(f"Skipping long segment {seg} in {ifile_path}")
            continue

        ofile_name = f"{ifile_path.stem}___{seg['beg']:.2f}___{seg['end']:.2f}.wav"
        ofile_path = audio_out_path / ofile_name

        if not ofile_path.exists():
            ofile_path.parent.mkdir(parents=True, exist_ok=True)
            # Slice waveform in memory
            beg_sample = int(seg["beg"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            fragment = wav[beg_sample:end_sample]
            # Write as wav soundfile
            tmp_path = str(ofile_path) + ".tmp"
            write(tmp_path, sample_rate, fragment)
            os.replace(tmp_path, ofile_path)  # atomic on most OSes to avoid mid-write files if crashes when writing
            n_created += 1
        else:
            n_exist += 1

        results.append((ofile_name, seg))
    return results, n_created, n_exist


def build_segments_dict(segments_path, source_path, target_path):
    """Read segments, source, target files and group by audio_name."""
    segments_dict = defaultdict(list)

    with segments_path.open("r", encoding="utf-8") as f_seg, \
         source_path.open("r", encoding="utf-8") as f_src, \
         target_path.open("r", encoding="utf-8") as f_tgt:

        n_segments = 0
        for seg, src, tgt in zip(f_seg, f_src, f_tgt):
            audio_name, beg, end = seg.strip().split(" ")
            segments_dict[audio_name].append({
                "beg": float(beg),
                "end": float(end),
                "src": src.strip(),
                "tgt": tgt.strip()
            })
            n_segments += 1
        print(f"Found {n_segments} segments in {len(segments_dict)} audio files")
        
    return segments_dict


def get_audio_dict(base_path):
    m4a_stem2path = {}
    slangs = [p.name for p in base_path.iterdir() if p.is_dir()]
    for slang in slangs:
        audios_path = base_path / slang / "audios"
        for audio_name in audios_path.glob("*.m4a"):
            audio_stem = Path(audio_name).stem
            if audio_stem in m4a_stem2path:
                print(f"repeated entry {audio_stem}")
            m4a_stem2path[audio_stem] = audios_path / audio_name
    print(f"Found {len(set(m4a_stem2path.keys()))} m4a files")
    return m4a_stem2path


def main():
    parser = argparse.ArgumentParser(description="Extract Europarl-ST audio fragments and build TSV.")
    parser.add_argument("--idir", type=str, default="/lustre/fsmisc/dataset/Europarl-ST/v1.1", help="Input path")
    parser.add_argument("--odir", type=str, default="/lustre/fsn1/projects/rech/eut/ujt99zo/josep/datasets", help="Output path")
    parser.add_argument("--lp", type=str, default="en-fr,en-es,en-de,en-it,en-pt,fr-en,fr-es,fr-de,fr-it,fr-pt,de-en,de-fr,de-es,de-it,de-pt,pt-en,pt-es,pt-de,pt-fr,pt-it,it-en,it-es,it-de,it-fr,it-pt", help="Comma-separated list of language pairs (i.e. en-fr,es-it)")
    parser.add_argument("--data_sets", type=str, default="test,dev,train", help="Comma-separated list of sets (i.e. test,dev")
    args = parser.parse_args()

    base_path = Path(args.idir)
    out_path = Path(args.odir)
    langs = [p.name for p in base_path.iterdir() if p.is_dir()]

    m4a_stem2path = get_audio_dict(base_path)

    tsv_file = out_path / f"Europarl-ST_v1.1.tsv"
    with tsv_file.open("w", encoding="utf-8") as f_tsv:

        for slang, tlang, data_set in [(s, t, d) for s in langs for t in langs for d in ["dev", "test", "train"] if s != t]:
            print(f"---------- {slang}-{tlang}:{data_set} ----------")
            segments_path = base_path / slang / tlang / data_set / "segments.lst"
            source_path = base_path / slang / tlang / data_set / f"segments.{slang}"
            target_path = base_path / slang / tlang / data_set / f"segments.{tlang}"

            n_created = 0
            n_exist = 0
            segments_dict = build_segments_dict(segments_path, source_path, target_path)
            for audio_stem, segments in tqdm(segments_dict.items(), desc=f"Processing {slang}-{tlang}:{data_set}", unit="file"):
                #en.20081117.22.1-112
                #[{'beg': 0.0, 'end': 15.98, 'src': 'Signor Presidente, ...', 'tgt': '. Senhor Presidente, ...'}, ...]

                results, n, m = extract_fragments(m4a_stem2path[audio_stem], segments, out_path / "audios")
                n_created += n
                n_exist += m
                #('en.20081117.22.1-112___0.00___15.98.wav', {'beg': 0.0, 'end': 15.98, 'src': 'Signor Presidente, ....', 'tgt': '. Senhor Presidente, ...'})
                for ofile_name, seg in results:
                    out_file = out_path / "audios" / ofile_name
                    f_tsv.write(f"{out_file}\t{slang}\t{seg['src']}\t{tlang}\t{seg['tgt']}\t{data_set}\n")

            print(f"Created {n_created} files ({n_exist} existing)")


if __name__ == "__main__":
    main()
