# train.py

import os
import json
#import wandb
import torch
import logging
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

from AudioToLLM import AudioToLLM
from Trainer import Trainer
from Dataset import Dataset
from scripts.plot_learning import plot_logs

logger = logging.getLogger("train")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class JSONMetricsLogger:
    def __init__(self, path, plot=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.plot = plot

    def log(self, **data):
        data["timestamp"] = datetime.now().isoformat(timespec="seconds")

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        if self.plot:
            try:
                plot_logs(self.path, output_file=str(self.path) + ".png")
            except Exception as e:
                logger.warning("Plot generation failed (%s). Training continues.", type(e).__name__)


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
    # opt pars
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="Learning rate for LoRA layers")
    parser.add_argument("--lr_proj", type=float, default=5e-4, help="Learning rate for projector layers")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps (must be >0 for scheduler)")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of training epochs (0 for no limit)")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Scheduler warmup steps (use ~5%)")
    # train pars
    parser.add_argument("--batch_size", type=int, default=8, help="Number of sampels in a batch")
    parser.add_argument("--accum_steps", type=int, default=4, help="Accumulate this many batchs before optimizing")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluation (and saving checkpoint) after this many optimization steps")
    parser.add_argument("--log_every", type=int, default=10, help="Logging after this many optimization steps")
    parser.add_argument("--save_best_n", type=int, default=3, help="Save top N checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume previous training")
    # output
    parser.add_argument("--output_dir", type=str, default="./sft_output", help="Output directory of training")
    parser.add_argument("--debug", action="store_true", help="Debug mode with more logging")
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logging
    log_filename = os.path.join(args.output_dir, f"train.log") #_{datetime.now().strftime('%Y%m%d_%H%M%S')}
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='a', encoding='utf-8'),  # log to file (append mode)
            logging.StreamHandler()  # and log to console
        ]
    )
    
    logger.info("=" * 80)
    logger.info(f"Starting new run @ {datetime.now().isoformat(timespec='seconds')}")
    logger.info("=" * 80)

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    logger.info(
        "\n" + "=" * 80 +
        f"\nCONFIG FILE: {args.config}\n" +
        json.dumps(config, indent=2) +
        "\n" + "=" * 80
    )

    json_logger = JSONMetricsLogger(os.path.join(args.output_dir, "metrics.jsonl"))
    json_logger.log(
        type="run",
        config=config,
        train=args.train,
        eval=args.eval,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        max_seq_len=args.max_seq_len,
        resume=args.resume,
        output_dir=args.output_dir,
    )

    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device count: {torch.cuda.device_count()}")

    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")

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
        tokenizer=model.llm.tokenizer,
        audio_token=config['llm']['audio_token'],
        bos_token=model.llm.tokenizer.bos_token or "",
        eos_token=model.llm.tokenizer.eos_token or "",
    )

    eval_dataset = Dataset(
        file_path=args.eval,
        tokenizer=model.llm.tokenizer,
        audio_token=config['llm']['audio_token'],
        bos_token=model.llm.tokenizer.bos_token or "",
        eos_token=model.llm.tokenizer.eos_token or "",
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
        warmup_steps=args.warmup_steps,
        save_best_n=args.save_best_n,
        eval_every=args.eval_every,
        log_every=args.log_every,
        accum_steps=args.accum_steps,
        output_dir=args.output_dir,
        json_logger=json_logger,
        resume=args.resume,
    )

    # -----------------------------
    # Start training
    # -----------------------------

    trainer.train()

