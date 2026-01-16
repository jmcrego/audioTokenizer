import json

config = {
    "audio": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium"
    },
    "projector": {
        "path": None,
        "conv_kernel": 30,
        "conv_stride": 30,
        "rmsnorm_pre": True,
        "act": None, #"silu",
        "rmsnorm_pos": True,
        "scale": 0.1,
        "use_bias": False,
    },
    "llm": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B",
        "audio_token": "<extra_id_0>", 
    },
    "embeddings": {
        "path": None,
        "special_tokens": [
            "<|task:asr|>", 
            "<|task:stt|>", 
            "<|task:2stt|>", 
            "<|speech|>", 
            "<|transcription|>", 
            "<|translation|>",
            "<|src_lang:en|>", 
            "<|src_lang:fr|>", 
            "<|src_lang:de|>", 
            "<|src_lang:ca|>", 
            "<|src_lang:it|>", 
            "<|src_lang:es|>", 
            "<|src_lang:pt|>", 
            "<|src_lang:nl|>", 
            "<|src_lang:ru|>", 
            "<|src_lang:ja|>", 
            "<|src_lang:ko|>", 
            "<|src_lang:ar|>", 
            "<|src_lang:zh-CN|>", 
            "<|tgt_lang:en|>", 
            "<|tgt_lang:fr|>", 
            "<|tgt_lang:de|>", 
            "<|tgt_lang:ca|>", 
            "<|tgt_lang:it|>", 
            "<|tgt_lang:es|>", 
            "<|tgt_lang:pt|>", 
            "<|tgt_lang:nl|>", 
            "<|tgt_lang:ru|>", 
            "<|tgt_lang:ja|>", 
            "<|tgt_lang:ko|>", 
            "<|tgt_lang:ar|>", 
            "<|tgt_lang:zh-CN|>", 
        ], ### added in the tokenizer, learned embeddigns
    },
    "lora": {
        "path": None,
        "r": 16,
        "lora_alpha": 32,
        "target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
}

with open(f"config.json", "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)
