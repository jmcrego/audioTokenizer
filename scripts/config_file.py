import json

config = {
    "audio": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/openai/whisper-medium"
    },
    "projector": {
        "path": None,
        "stack_size": 8,
        "middle_dim": 2048,
        "rmsnorm_pre": True,
        "rmsnorm_mid": False,
        "rmsnorm_pos": False,
        "scale": 0.03,
    },
    "llm": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct",
        "add_tokens": {
            "path": None,
            "asr_start_token": "<asr>",
            "asr_end_token": "</asr>",
            "stt_start_token": "<stt>",
            "stt_end_token": "</stt>",
            "audio_token": "<[audio]>",
        }
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
