#!/usr/bin/env python3

import torch
import logging
import numpy as np
import soundfile as sf
import soxr

logger = logging.getLogger("audio_embedder")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def preprocess_audio(audio_input, sample_rate=16000, channel=0):
    """Load WAV from file or an audio chunk (float32 numpy array), convert to mono (channel), resample (sample_rate), normalize, ..."""

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
    logger.debug(f"preprocess returns wav {wav.shape} type={wav.__class__.__name__} dtype={wav.dtype}")

    return wav

def get_model_stride(embedder, feature_extractor, model_name):
    model_name = model_name.lower()
    if "whisper" in model_name:
        return feature_extractor.hop_length
    else:    
        stride = 1
        for layer in embedder.feature_extractor.conv_layers:
            stride *= layer.conv.stride[0]
        return stride

class AudioEmbedder:
    """
    Audio embeddings extractor with chunk/stride support.
    Models supported: 'mhubert-147', 'wav2vec2-xlsr-53', 'whisper'
    """

    def __init__(self, 
                 model: str = "utter-project/mhubert-147",
                 l2_norm: bool=False, 
                 half_precision: bool=False,
                 chunk_size: int = 3200, #number of samples of each chunk passed to the model (the chunk will contain N/320 embeddings)
                 stride: int = 1600, #number of samples to move for the next chunk (must be <= chunk_size to not lose sammples), allows chunk overlap for smooth embeddings
                 device: str = "cpu",):
        meta = {k: v for k, v in locals().items() if k != "self"}
        logger.info(f"Initializing {meta}")

        assert stride <= chunk_size , f"stride {stride} must be <= chunk_size ({chunk_size})"
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

        self.model_stride = get_model_stride(self.embedder, self.feature_extractor, self.model)
        #model_stride is the downsampling factor from audio samples to embeddings (how many audio samples used for one embedding)
        assert chunk_size % self.model_stride == 0, f"chunk_size ({chunk_size}) must be a multiple of model stride ({self.model_stride})"
        #chunk_size must be a multiple of model stride to avoid padding

        self.sample_rate = self.feature_extractor.sampling_rate
        self.embedder.to(self.device)
        if self.half_precision:
            self.embedder = self.embedder.half()  # for A100/H100
#        self.embedder = torch.compile(self.embedder)
        self.embedder.eval()
        logger.debug(f"Read model {model} model_stride={self.model_stride} D={self.D}")


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

        # ----------------------------------------------
        # --- preprocess audios and build chunks -------
        # ----------------------------------------------
        t = time.time()
        for audio in audio_inputs:
            wav = preprocess_audio(audio, sample_rate=self.sample_rate)
            n_samples = len(wav)
            # 1. Compute chunk start positions
            starts = np.arange(0, n_samples, self.stride)
            # 2. Pad wav ONCE so all end slices exist
            padded_len = starts[-1] + self.chunk_size
            if padded_len > n_samples:
                wav = np.pad(wav, (0, padded_len - n_samples))
            # 3. Extract chunks: fast vectorized slicing
            chunks = np.stack([wav[s:s + self.chunk_size] for s in starts])
            # results
            all_chunks.append(chunks) # [n_chunks, chunk_size]
            lengths.append(len(chunks)) # number of chunks per audio input
        logger.debug(f"preprocess took {time.time()-t:.2f} sec")

        # ----------------------------------------------
        # --- concat all chunks and extract features ---
        # ----------------------------------------------
        t = time.time()
        # Concatenate all chunks for batch processing
        batch_chunks = np.concatenate(all_chunks, axis=0)  # [C, cs] # C ~ Total chunks; cs ~ chunk size (number of samples in a chunk)
        # Feature extraction
        input_dict = self.feature_extractor(batch_chunks, sampling_rate=self.sample_rate, return_tensors="pt", padding=False)
        inputs = input_dict.input_values if "whisper" not in self.model else input_dict.input_features
        inputs = inputs.pin_memory().to(self.device, non_blocking=True) #[C, F] (for raw audio) or [C, F, f] (for Whisper)
        #C ~ batch size (total number of chunks)
        #F ~ time dimension (number of frames per audio chunk)
        #f ~ feature dimension (for spectrograms)
        logger.debug(f"feature extraction took {time.time()-t:.2f} sec")

        if self.half_precision:
            inputs = inputs.half()

        # ----------------------------------------------
        # --- extract embeddings from all features -----
        # ----------------------------------------------
        t = time.time()
        # Forward pass
        with torch.inference_mode():
            out = self.embedder(inputs).last_hidden_state  # [C, E, D] # E ~ number of embeddings in chunk (frames) # D ~ embedding dimension

        # Optional L2 normalization (only for computing clusters)
        if self.l2_norm:
            out = torch.nn.functional.normalize(out, dim=-1)
        logger.debug(f"embedding took {time.time()-t:.2f} sec")

        # ----------------------------------------------
        # --- back to original format ------------------
        # ----------------------------------------------
        t = time.time()
        # Split outputs back into original audios (B), each audio input is an entry in batch
        embeddings = []
        masks = []
        idx = 0
        for n_chunks in lengths: #n_chunks (nC) is the number of chunks on each audio file
            emb_audio = out[idx: idx + n_chunks]  # [nC_i, E, D] #nC_i ~ number of chunks in this audio file
            idx += n_chunks

            # Flatten chunks along time dimension
            emb_audio = emb_audio.reshape(-1, self.D)  # [nC_i*E, D] # nC_i*E is the number of embeddings in current audio file
            embeddings.append(emb_audio)

            mask = torch.ones(emb_audio.shape[0], dtype=torch.bool, device=self.device) #[nC_i*E]
            masks.append(mask) 
        #embeddings ~ [B, nC_i*E, D] (nC_i*E is different on each list element)
        #masks = [B, nC_i*E]

        # ----------------------------------------------
        # --- add padding and return tensors -----------
        # ----------------------------------------------
        # Pad all sequences to the max length of embeddings (T)
        max_len = max(e.shape[0] for e in embeddings)
        padded_embeddings = torch.stack([torch.nn.functional.pad(e, (0,0,0,max_len - e.shape[0])) for e in embeddings]) #[B, T, D] 
        padded_masks = torch.stack([torch.nn.functional.pad(m, (0,max_len - m.shape[0])) for m in masks]) #[B, T]
        logger.debug(f"formatting took {time.time()-t:.2f} sec")

        return padded_embeddings, padded_masks




if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Extract audio embeddings from file or array.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="utter-project/mHuBERT-147", help="Path or HuggingFace model name (i.e. openai/whisper-small, utter-project/mhubert-147, facebook/wav2vec2-xlsr-53 models)")
    parser.add_argument("--wav", type=str, help="Comma separated list of paths to audio files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_embedder = AudioEmbedder(model=args.model, device=args.device)
    t = time.time()
    embeddings, masks = audio_embedder(args.wav.split(','))
    print(f"Output embeddings {embeddings.shape}, took {time.time()-t:.2f} sec")
