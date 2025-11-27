
#!/usr/bin/env python3
import time
import queue
import logging
import argparse
import numpy as np
import sounddevice as sd

from AudioEmbedder import AudioEmbedder
from AudioProcessor import AudioProcessor
from AudioTokenizer import AudioTokenizer
from Utils import secs2human

def record_mic_stream(chunk_duration=5., sample_rate=16000):
    """
    Generator that yields consecutive audio chunks from the microphone.
    Uses a background callback to ensure no audio is lost.
    Args:
        sample_rate (int): Sampling rate in Hz.
        chunk_duration (float): Duration of each yielded chunk in seconds.
    Yields:
        np.ndarray: 1D float32 array of audio samples of length chunk_duration*sample_rate
    """
    if sd is None:
        raise ImportError("sounddevice is required for microphone input")

    q = queue.Queue()
    nchunks = 0 # num of chunks retrieved

    def callback(indata, frames, time, status):
        """This callback runs in a separate thread and puts audio into the queue."""
        nonlocal nchunks # allow writing to outer variables
        if frames != sample_rate * chunk_duration:
            print("Warning: audio buffer irregularity")
        if status:
            print(f"Microphone status: {status}")
        logging.info(f"sample_rate={sample_rate} Hz, "
                    f"frames={frames}, "
                    f"time=[{secs2human(nchunks*frames/sample_rate)}, {secs2human((nchunks+1)*frames/sample_rate)}], "
                    f"chunk_latency={time.currentTime - time.inputBufferAdcTime:.6f} sec"
        )
        nchunks += 1
        q.put(indata.copy())

    chunk_frames = int(chunk_duration * sample_rate)
    buffer = np.zeros((0, 1), dtype=np.float32)

    with sd.InputStream(blocksize=int(sample_rate*chunk_duration), samplerate=sample_rate, channels=1, callback=callback):
        print(f"Recording indefinitely. Tokenizing every {chunk_duration} sec. Press CTRL-C to stop.")
        while True:
            # Fill buffer until we have enough frames for one chunk
            while buffer.shape[0] < chunk_frames:
                buffer = np.vstack([buffer, q.get()])
            # Take exactly one chunk from buffer
            chunk = buffer[:chunk_frames]
            buffer = buffer[chunk_frames:]  # keep remaining audio
            yield chunk.flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize audio using pretrained centroids")
    parser.add_argument("--model", type=str, default="utter-project/mhubert-147")
    parser.add_argument("--centroids", type=str, default="centroids.mhubert-147.100.npy")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration of each audio chunk in seconds (only if streaming)")
    parser.add_argument("--wav", type=str, default=None, help="Audio file to tokenize (otherwise streaming is run)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_processor = AudioProcessor(top_db=30, stride=320, receptive_field=400)
    audio_embedder = AudioEmbedder(audio_processor, model=args.model)
    audio_tokenizer = AudioTokenizer(audio_embedder, args.centroids)

    if args.wav is None:
        try:
            for i, chunk in enumerate(record_mic_stream(chunk_duration=args.duration, sample_rate=16000)):
                t = time.time()
                tokens = audio_tokenizer(chunk)
                logging.info(f"Chunk {i+1}, process took {time.time()-t:.3f} sec, tokens={tokens.shape[0]}\n{tokens}")
        except KeyboardInterrupt:
            print("\nStreaming stopped.")

    else:
        t = time.time()
        tokens = audio_tokenizer(args.wav)
        logging.info(f"Tokenization took {time.time()-t:.3f} sec, tokens={tokens.shape[0]}\n{tokens}")
