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
            return [], 0, 0, 0
    
    results = []
    n_exist = 0
    n_created = 0
    t_audio = 0
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

        t_audio += duration_sec
        results.append((ofile_name, seg))

    return results, n_created, n_exist, t_audio
