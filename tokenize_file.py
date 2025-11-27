
#!/usr/bin/env python3
import time
import queue
import torch
import logging
import numpy as np
#import sounddevice as sd

from Utils import load_wav

if __name__ == "__main__":
    import argparse
    from AudioTokenizer import AudioTokenizer

    parser = argparse.ArgumentParser(description="Tokenize audio using pretrained centroids")
    parser.add_argument("--model", type=str, default="utter-project/mhubert-147")
    parser.add_argument("--centroids", type=str, default="centroids.mhubert-147.100.npy")
    parser.add_argument("--wav", type=str, required=True, help="Audio file to tokenize")
    parser.add_argument("--channel", type=int, default=1, help="Channel to consider")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    tokenizer = AudioTokenizer(args.model, args.centroids)
    logging.info(f"Loaded tokenizer with model={args.model} centroids={args.centroids}")
    wav = load_wav(args.wav, args.channel) 
    logging.info(f"Loaded audio file {args.wav}")
    t = time.time()
    tokens = tokenizer(wav)
    logging.info(f"Process took {time.time()-t:.3f} sec, tokens={tokens.shape[0]}\n{tokens}")

