# audioTokenizer
Build discrete tokens from speech

## Create conda environment and install dependencies
```
conda create -n audio_tokenizer python=3.11 -y
conda activate audio_tokenizer

pip install -r requirements.txt
# Test everything
python -c "import torch; import torchaudio; import numpy; import sounddevice; import transformers; print('All OK')"

```

## Use:
### Building centroids from audio files:
```
python train_kmeans.py --model utter-project/mhubert-147 --data data --k 512 --max-iter 1000
```

### Tokenize an entire audio file:
```
python tokenizer_audio_file.py --model utter-project/mhubert-147 --centroids centroids.mhubert-147.512.npy --wav data/common_voice_fr_18916222.mp3
```

### Stream audio tokens obtained from your mic:
```
python tokenizer_mic_stream.py --model utter-project/mhubert-147 --centroids centroids.mhubert-147.100.npy --duration 1
```
