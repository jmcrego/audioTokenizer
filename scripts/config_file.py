import json

config = {
    "audio": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium"
    },
    "projector": {
        "path": None,
        "rmsnorm_pre": True, #False to discard it
        "conv_kernel": 15,
        "conv_stride": 15,
        "linear": True,
        "act": "silu",       #None to discard it
        "rmsnorm_pos": True, #False to discard it
        "scale": 0,          #0 to discard it
        "use_bias": False,   #False to discard it
    },
    "llm": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B",
        "audio_token": "<extra_id_0>", #must exist (unused) in vocab
        "pad_token" : "<|im_end|>"     #must be different to eos token
    },
    "embeddings": {
        "path": None,
        "special_tokens": [  #will be added in tokenizer, learned embeddigns
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
        ], 
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
