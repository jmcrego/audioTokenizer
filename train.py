# train.py

import os
import json
import torch
import logging
import argparse
import numpy as np

from AudioToLLM import AudioToLLM
from Trainer import Trainer
from Dataset import Dataset

logger = logging.getLogger("train")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_device_dtype():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # picks default CUDA device
        try:
            props = torch.cuda.get_device_properties(device)
            name = props.name.lower()
            if "h100" in name:
                dtype = torch.bfloat16
            elif "a100" in name:
                dtype = torch.bfloat16  # optional, you could also use fp16
            else:  # V100, T4, etc.
                dtype = torch.float16
        except Exception:
            dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    dtype = torch.float32
    return device, dtype    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a speech ASR/STT decoder (audio-embedder ➔ Projector ➔ LLM).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Model config file")
    # dataset paths
    parser.add_argument("--train", required=True, help="Training dataset file")
    parser.add_argument("--eval", default=None, help="Evaluation dataset file")
    # train/opt pars
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="Learning rate for LoRA layers")
    parser.add_argument("--lr_proj", type=float, default=5e-4, help="Learning rate for projector layers")
    parser.add_argument("--accum_steps", type=int, default=4, help="Accumulate this many steps before optimizing")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps (must be >0 for scheduler)")
    parser.add_argument("--max_epochs", type=int, default=0, help="Maximum number of training epochs (0 for no limit)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--eval_every", type=int, default=1000, help="Run evaluation after this many steps")
    parser.add_argument("--log_every", type=int, default=100, help="Logging after this many steps")
    parser.add_argument("--save_best_n", type=int, default=3, help="Save top N checkpoints")
    # output
    parser.add_argument("--output_dir", type=str, default="./sft_output", help="Output directory of training")
    parser.add_argument("--debug", action="store_true", help="Debug mode with more logging")
    args = parser.parse_args()

    if args.log_every % args.accum_steps != 0:
        raise ValueError(f"--log_every ({args.log_every}) must be a multiple of --accum_steps ({args.accum_steps})")
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logging
    log_filename = os.path.join(args.output_dir, f"train.log") #_{datetime.now().strftime('%Y%m%d_%H%M%S')}
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),  # save to file
            logging.StreamHandler()  # and print to console
        ]
    )

    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device count: {torch.cuda.device_count()}")

    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    # -----------------------------
    # Load model wrapper
    # -----------------------------

    model = AudioToLLM(
        config=config,
        device=device,
        dtype=dtype 
    )

    # -----------------------------
    # Datasets 
    # -----------------------------

    train_dataset = Dataset(
        file_path=args.train,
        tokenizer=model.tokenizer,
        asr_token=config["asr_token"],
        stt_token=config["stt_token"],
        sample_rate=model.audio_embedder.sample_rate,
        downsample_ratio=model.audio_embedder.downsample_ratio,
        stack_size=config["projector"]["stack_size"],
        max_seq_len=args.max_seq_len
    )

    eval_dataset = Dataset(
        file_path=args.eval,
        tokenizer=model.tokenizer,
        asr_token=config["asr_token"],
        stt_token=config["stt_token"],
        sample_rate=model.audio_embedder.sample_rate,
        downsample_ratio=model.audio_embedder.downsample_ratio,
        stack_size=config["projector"]["stack_size"],
        max_seq_len=args.max_seq_len
    ) if args.eval is not None else None

    # -----------------------------
    # Create Trainer
    # -----------------------------

    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        lr_proj=args.lr_proj,
        lr_lora=args.lr_lora,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        save_best_n=args.save_best_n,
        eval_every=args.eval_every,
        log_every=args.log_every,
        accum_steps=args.accum_steps,
        output_dir=args.output_dir,
    )

    # -----------------------------
    # Start training
    # -----------------------------

    trainer.train()

