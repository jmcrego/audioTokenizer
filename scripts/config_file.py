import json

config = {
    "audio": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147",
    },
    "projector": {
        "path": None,
        "stack_size": 8,
        "middle_dim": 2048,
        "rmsnorm_pre": True,
        "rmsnorm_mid": False,
        "rmsnorm_pos": True,
    },
    "llm": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct",
        "token_map": {
            "<asr>": 5,
            "</asr>": 6,
            "<stt>": 7,
            "</stt>": 8,
            "<[audio]>": 9,
        },
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
    "asr_start_token": "<asr>",
    "asr_end_token": "</asr>",
    "stt_start_token": "<stt>",
    "stt_end_token": "</stt>",
    "audio_token": "<[audio]>",
}

with open(f"config.json", "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)
