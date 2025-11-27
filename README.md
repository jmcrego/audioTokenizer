# audioTokenizer
Build discrete tokens from speech

Workflow diagram:
```
numpy WAV (float32, 16 kHz)                       → [samples]
        ↓
processor                                         → tensor [bs=1, samples]
        ↓
CNN feature encoder (stride=320, RF≈400)          → Latent features [bs, seq_len, D]
        ↓
Transformer encoder                               → Hidden states [bs, seq_len, D] 
        ↓
K-means tokenizer                                 → Discrete tokens [bs, seq_len]
```

Caveats:
* Processor handles resampling, channel handling (mono), amplitude normalization, Silence removal and padding. Returns torch.Tensor. For Whisper it may produce log-mel features instead.
* CNN feature encoder extracts accoustic patterns from audio chunks. A stack of strided conv layers that downsample the input. [use pretrain model FROZEN]
* Transformer encoder refines features. It applies global context, self-attention across the entire sequence. [use pretrain model FROZEN]
* K-means tokenizer creates discrete acoustic units by mapping each embedding to the nearest centroid. [must be TRAINED FROM SCRATCH using speech files]
* D depends on the model used (Ex: mHuBERT base: 768, mHuBERT large: 1024, wav2vec2 base: 768, wav2vec2 large: 1024).
* One embedding/token = 20 ms of audio (stride = 320 samples / 16000 = 0.02 sec = 20 ms)


## Install

### Create conda environment and install dependencies
```
conda create -n audio_tokenizer python=3.11 -y
conda activate audio_tokenizer

pip install -r requirements.txt
# Test everything
python -c "import torch; import torchaudio; import numpy; import sounddevice; import transformers; print('All OK')"

```

## Use

### Building centroids from audio files:
```
python train_kmeans.py --model utter-project/mhubert-147 --data data --k 512 --max-iter 1000
```

### Tokenize an entire audio file:
```
python tokenize_file.py --model utter-project/mhubert-147 --centroids centroids.mhubert-147.512.npy --wav data/common_voice_fr_18916222.mp3
```

### Stream audio tokens obtained from your mic:
```
python tokenize_mic.py --model utter-project/mhubert-147 --centroids centroids.mhubert-147.100.npy --duration 1
```
