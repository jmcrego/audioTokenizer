import json

config = {
    "audio": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147",
        "embedding_dim": 768,
        "l2_norm": False,
        "chunk_size" : 3200,
        "stride": 1600
    },
    "projector": {
        "path": none,
        "stack_size": 8,
        "embedding_dim": 2048,
        "rank_dim": 256,
        "max_seq_len": 1024
    },
    "llm": {
        "path": "/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct"
    },
    "lora": {
        "path": none,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
}

with open(f"config.json", "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)
