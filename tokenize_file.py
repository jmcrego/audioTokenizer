
#!/usr/bin/env python3
import time
import logging
import argparse

from AudioEmbedder import AudioEmbedder
from AudioProcessor import AudioProcessor
from AudioTokenizer import AudioTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize audio using pretrained centroids")
    parser.add_argument("--model", type=str, default="utter-project/mhubert-147")
    parser.add_argument("--centroids", type=str, default="centroids.mhubert-147.100.npy")
    parser.add_argument("--wav", type=str, required=True, help="Audio file to tokenize")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_processor = AudioProcessor(top_db=30, stride=320, receptive_field=400)
    audio_embedder = AudioEmbedder(audio_processor, model=args.model)
    audio_tokenizer = AudioTokenizer(audio_embedder, args.centroids)

    t = time.time()
    tokens = audio_tokenizer(args.wav)
    logging.info(f"Process took {time.time()-t:.3f} sec, tokens={tokens.shape[0]}\n{tokens}")

