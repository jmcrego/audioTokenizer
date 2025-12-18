import json

config = {
    "audio": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147",
        "l2_norm": False,
    },
    "projector": {
        "path": None,
        "stack_size": 8,
        "rank_dim": 256,
    },
    "llm": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct"
    },
    "lora": {
        "path": None,
        "r": 16,
        "lora_alpha": 32,
        "target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "asr_token": "[ASR]",
    "stt_token": "[STT]"
}

with open(f"config.json", "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)
