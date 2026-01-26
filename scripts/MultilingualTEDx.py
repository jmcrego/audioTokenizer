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
import json
import soxr 
import os
import sys
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
        return [], 0, 0, 0

    segments.sort(key=lambda s: s["beg"])

    ### check if all output segments are already in place

    all_segments_exist = True
    for seg in segments:

        duration_sec = seg["end"] - seg["beg"]

        if duration_sec <= 0:
            continue

        if duration_sec > 30.0:
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
            return [], 0, 0, 0
    
    results = []
    n_exist = 0
    n_created = 0
    t_audio = 0
    for seg in segments:

        duration_sec = seg["end"] - seg["beg"]
        
        if duration_sec <= 0:
            # print(f"Skipping invalid segment {seg} in {ifile_path}")
            continue

        if duration_sec > 30.0:
            # print(f"Skipping long segment {seg} in {ifile_path}")
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

        t_audio += duration_sec
        results.append((ofile_name, seg))

    return results, n_created, n_exist, t_audio


def build_segments_dict(segments_path, source_path, target_path):
    """Read segments, source, target files and group by audio_name."""
    segments_dict = defaultdict(list)

    with segments_path.open("r", encoding="utf-8") as f_seg, \
         source_path.open("r", encoding="utf-8") as f_src, \
         target_path.open("r", encoding="utf-8") as f_tgt:

        n_segments = 0
        for seg, src, tgt in zip(f_seg, f_src, f_tgt):
            _, audio_name, beg, end = seg.strip().split(" ")
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
    print(base_path)
    flac_stem2path = {}
    for audio_name in base_path.glob("*.flac"):
        audio_stem = Path(audio_name).stem
        if audio_stem in flac_stem2path:
            print(f"repeated entry {audio_stem}")
        flac_stem2path[audio_stem] = base_path / audio_name
    print(f"Found {len(set(flac_stem2path.keys()))} flac files")

    return flac_stem2path


def main():
    parser = argparse.ArgumentParser(description="Extract MultilingualTEDx audio fragments and build TSV.")
    parser.add_argument("--idir", type=str, default="/lustre/fsmisc/dataset/MultilingualTEDx", help="Input path")
    parser.add_argument("--odir", type=str, default="/lustre/fsn1/projects/rech/eut/ujt99zo/josep/datasets", help="Output path")
    args = parser.parse_args()

    base_path = Path(args.idir)
    out_path = Path(args.odir)

    lang_pairs = {tuple(p.name.split("-")) for p in base_path.iterdir() if p.is_dir() and len(p.name.split("-")) == 2 and all(len(x) == 2 for x in p.name.split("-"))}
    data_sets = ["valid", "test", "train"]

    # tsv_file = out_path / f"MultilingualTEDx.tsv"
    # with tsv_file.open("w", encoding="utf-8") as f_tsv:
    json_file = out_path / f"MultilingualTEDx.json"
    with json_file.open("w", encoding="utf-8") as f_json:

        n_entries = 0
        t_entries = 0
        for lsrc, ltgt in lang_pairs:
            if lsrc == ltgt:
                continue

            for data_set in data_sets:
                print(f"---------- {lsrc}-{ltgt}:{data_set} ----------")
                source_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"{data_set}.{lsrc}"
                target_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"{data_set}.{ltgt}"
                segments_path = base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "txt" / f"segments"

                flac_stem2path = get_audio_dict(base_path / f"{lsrc}-{ltgt}" / "data" / data_set / "wav")

                n_created = 0
                n_exist = 0
                t_audio = 0

                segments_dict = build_segments_dict(segments_path, source_path, target_path)

                for audio_stem, segments in tqdm(segments_dict.items(), desc=f"Processing {lsrc}-{ltgt}:{data_set}", unit="file"):

                    results, n, m, duration = extract_fragments(flac_stem2path[audio_stem], segments, out_path / "audios"/ "MultilingualTEDx")
                    n_created += n
                    n_exist += m
                    t_audio += duration

                    for ofile_name, seg in results:
                        out_file = str(out_path / "audios" / ofile_name)
                        # f_tsv.write(f"{out_file}\t{lsrc}\t{seg['src']}\t{ltgt}\t{seg['tgt']}\t{data_set}\n")
                        f_json.write(
                            json.dumps({
                                "audio_file": out_file,
                                "set": data_set,
                                "transcription": {
                                    "lang": lsrc, 
                                    "text": seg['src']
                                },
                                "translation": {
                                    "lang": ltgt,
                                    "text": seg['tgt']
                                }
                            }, ensure_ascii=False) + "\n"
                        )
                print(f"Created {n_created} files ({n_exist} existing), total duration {t_audio:.1f} secs")

            n_entries += n_created + n_exist
            t_entries += t_audio

    print(f"Total entries {n_entries}")
    print(f"Total duration {t_entries:.1f} secs ({t_entries/n_entries:.1f} secs/file)")

if __name__ == "__main__":
    main()
