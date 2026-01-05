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
        "rmsnorm_pos": True,
        "scale": 0,
        "use_bias": False,
    },
    "llm": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct",
        "audio_token": "<extra_id_0>"
        "asr_token": "<extra_id_1>"
        "stt_token": "<extra_id_2>"
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
