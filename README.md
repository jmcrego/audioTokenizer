# audioTokenizer
Build discrete tokens from speech

# Building centroids from audio files:
```
python train_kmeans.py --model utter-project/mhubert-147 --data data --k 512 --max-iter 1000
```

# Tokenize an entire audio file:
```
python tokenizer_audio_file.py --model utter-project/mhubert-147 --centroids centroids.mhubert-147.512.npy --wav data/common_voice_fr_18916222.mp3
```

# Stream audio tokens obtained from your mic:
```
python tokenizer_mic_stream.py --model utter-project/mhubert-147 --centroids centroids.mhubert-147.100.npy --duration 1
```
