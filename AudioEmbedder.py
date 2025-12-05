#!/usr/bin/env python3
"""
Class to extract embeddings from audio using mHuBERT, wav2vec2, or Whisper encoder.
Supports WAV/MP3 files or raw numpy audio arrays.
"""

import torch
import logging
import numpy as np
import soundfile as sf
import soxr

logger = logging.getLogger("audio_embedder")

def arguments(args):
    args.pop('self', None)  # None prevents KeyError if 'self' doesn't exist
    return args


def preprocess_audio(audio_input, sample_rate=16000, channel=0):
    """Load WAV from file or an audio chunk (float32 numpy array), convert to mono (channel), resample (sample_rate), ..."""

    if isinstance(audio_input, str):
        wav, sr = sf.read(audio_input)
    elif isinstance(audio_input, np.ndarray):
        wav = audio_input
        sr = sample_rate 
    else:
        raise ValueError("audio_input must be a path or np.ndarray")
    logger.debug(f"wav size={wav.shape} sr={sr} time={wav.shape[0]/sr:.2f} sec")

    # -----------------------------
    # --- mono CHANNEL ------------
    # -----------------------------
    if len(wav.shape) > 1: 
        if channel == -1:
            wav = np.mean(wav, axis=1)
        else:
            wav = wav[:, channel]
        logger.debug(f"handled channels, wav size={wav.shape} time={wav.shape[0]/sr:.2f} sec")

    # -----------------------------
    # --- RESAMPLE ----------------
    # -----------------------------
    if sr != sample_rate:
        wav = soxr.resample(wav, sr, sample_rate)
        logger.debug(f"resampled, wav size={wav.shape} sr={sample_rate} time={wav.shape[0]/sample_rate:.2f} sec")

    # -----------------------------
    # --- Normalize audio amplitude
    # -----------------------------
    wav = wav / max(1e-8, np.abs(wav).max())

    # -----------------------------
    # --- ENSURE float32 dtype ----
    # -----------------------------
    wav = wav.astype(np.float32)

    return wav


class AudioEmbedder:
    """
    Audio embeddings extractor with chunk/stride support.
    Models supported: 'mhubert-147', 'wav2vec2-xlsr-53', 'whisper'
    """

    def __init__(self, 
                 model: str = "utter-project/mhubert-147",
                 l2_norm: bool=False, 
                 half_precision: bool=False,
                 device: str = "cpu",
                 chunk_size: int = 3200, #number of samples of each chunk passed to the model (it contains N/320 embeddings)
                 stride: int = 1600): #number of samples to move for the next chunk (must be <= chunk_size to not lose sammples), allows chunk overlap for smooth embeddings
        self.meta = arguments(locals())
        logger.info(f"Initializing {self.meta}")
        assert chunk_size % 320 == 0, f"chunk_size ({chunk_size}) 
        #chunk_size must be a multiple of 320 (model stride)" #For mHuBERT/wav2vec2 model stride is 320 (number of samples for an embedding)
        #larger chunks allow for larger context when computing its internal embeddings, as the model sees all the chunk when computing embeddings
        assert stride <= chunk_size , f"stride {stride} must be <= chunk_size ({chunk_size})"
        #stride allows the overlap chunks (its embeddings), thus reducing the problem of truncating the sound of a word
        #i could add averaging for overlapping frames (embeddings) to make the embeddings smoother and prevent duplicated frame effects

        self.device = torch.device(device)
        self.l2_norm = l2_norm
        self.half_precision = half_precision
        self.model = model.lower()
        self.chunk_size = chunk_size
        self.stride = stride

        if "mhubert" in model.lower():
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model)
            self.embedder = HubertModel.from_pretrained(model)
            self.D = self.embedder.config.hidden_size

        elif "wav2vec2" in model.lower():
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model)
            self.embedder = Wav2Vec2Model.from_pretrained(model)
            self.D = self.embedder.config.hidden_size

        elif "whisper" in self.model.lower():
            from transformers import WhisperFeatureExtractor, WhisperModel
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model)
            self.embedder = WhisperModel.from_pretrained(model).encoder
            self.D = self.embedder.config.d_model

        else:
            raise ValueError(f"Unknown model: {model}")

        self.sample_rate = self.feature_extractor.sampling_rate
        self.embedder.to(self.device)
        if self.half_precision:
            self.embedder = self.embedder.half()  # for A100/H100
        self.embedder = torch.compile(self.embedder)
        self.embedder.eval()


    def __call__(self, audio_inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract embeddings from a batch of audio files or numpy arrays with chunk/stride.
        
        Args:
            audio_inputs: List of str paths or np.ndarray audio chunks.
        
        Returns:
            embeddings: torch.Tensor [B, T, D] padded to the longest sequence
            mask: torch.BoolTensor [B, T] indicating valid frames
        """
        all_chunks = []
        lengths = []

        # Preprocess and slice each audio into overlapping chunks
        for audio in audio_inputs:
            wav = preprocess_audio(audio, sample_rate=self.sample_rate)
            n_samples = len(wav)

            # Split into overlapping chunks
            chunks = []
            for start in range(0, n_samples, self.stride):
                end = start + self.chunk_size
                chunk = wav[start:end]
                if len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
                chunks.append(chunk)
            all_chunks.append(np.stack(chunks))  # [n_chunks, chunk_size]
            lengths.append(len(chunks))  # number of chunks per audio

        # Concatenate all chunks for batch processing
        batch_chunks = np.concatenate(all_chunks, axis=0)  # [C, cs]  
        # C ~ Total chunks
        # cs ~ chunk size (number of samples in a chunk)

        # Feature extraction
        input_dict = self.feature_extractor(batch_chunks, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        inputs = input_dict.input_values if "whisper" not in self.model else input_dict.input_features
        inputs = inputs.pin_memory().to(self.device, non_blocking=True) #[B, F] (for raw audio) or [B, F, f] (for Whisper)
        #C ~ batch size (number of chunks)
        #F ~ time dimension (number of frames per audio chunk)
        #f ~ feature dimension (for spectrograms)

        if self.half_precision:
            inputs = inputs.half()

        # Forward pass
        with torch.inference_mode():
            out = self.embedder(inputs).last_hidden_state  # [C, E, D] (each frame is now an embedding of size D)
        #E ~ number of embeddings (same as previously number of frames)
        #D ~ embedding dimension

        # Optional L2 normalization
        if self.l2_norm:
            out = torch.nn.functional.normalize(out, dim=-1)

        # Split outputs back into original audios (A)
        embeddings = []
        masks = []
        idx = 0
        for n_chunks in lengths: #n_chunks is the number of chunks on each audio file
            emb_audio = out[idx: idx + n_chunks]  # [nC, E, D]
            #nC ~ number of chunks in this audio file
            idx += n_chunks

            # Flatten chunks along time dimension
            emb_audio = emb_audio.reshape(-1, self.D)  # [nC*E, D]
            #nC*E is the number of embeddings in current audio file
            embeddings.append(emb_audio)

            mask = torch.ones(emb_audio.shape[0], dtype=torch.bool, device=self.device) #[nC*E]
            masks.append(mask) 
        #embeddings ~ [A, nC*E, D] (nC*E is different on each list element)
        #masks = [A, nC*E]


        # Pad all sequences to the max length
        max_len = max(e.shape[0] for e in embeddings)
        padded_embeddings = torch.stack([torch.nn.functional.pad(e, (0,0,0,max_len - e.shape[0])) for e in embeddings])
        #[A, T_max, D] # total number of frames/embeddings for this audio
        padded_masks = torch.stack([torch.nn.functional.pad(m, (0,max_len - m.shape[0])) for m in masks])
        #[A, T_max]

        return padded_embeddings, padded_masks




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract audio embeddings from file or array.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="utter-project/mHuBERT-147", help="Path or HuggingFace model name (i.e. openai/whisper-small, utter-project/mhubert-147, facebook/wav2vec2-xlsr-53 models)")
    parser.add_argument("--wav", type=str, help="Path to WAV/MP3 file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_embedder = AudioEmbedder(model=args.model, device=args.device)
    embeddings, masks = audio_embedder([args.wav])
    print(f"embeddings {embeddings.shape}")
