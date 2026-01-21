import json

config = {
    "audio": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium"
    },
    "projector": {
        "path": None,
        "conv_kernel": 15,
        "conv_stride": 15,
        "rmsnorm_pre": True,
        "act": "silu", #None
        "rmsnorm_pos": True,
        "scale": 0.1,
        "use_bias": False,
    },
    "llm": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B",
        "audio_token": "<extra_id_0>", 
        "pad_token" : "<|im_end|>"
    },
    "embeddings": {
        "path": None,
        "special_tokens": [
            "<|asr|>", 
            "<|ast|>", 
            "<|stt|>", 
            "<|stt-asr|>", 
            "<|en|>", 
            "<|fr|>", 
            "<|de|>", 
            "<|ca|>", 
            "<|it|>", 
            "<|es|>", 
            "<|pt|>", 
            "<|nl|>", 
            "<|ru|>", 
            "<|ja|>", 
            "<|ko|>", 
            "<|ar|>", 
            "<|zh-CN|>", 
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
