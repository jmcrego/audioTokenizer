# infer.py

import torch
import json
import time
import logging
import argparse
from contextlib import nullcontext

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from train import get_device_dtype
from AudioToLLM import AudioToLLM

logger = logging.getLogger("infer")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Transcribe and/or translate audio using AudioToLLM (Hugging Face).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, required=True, help="Model config file")
    parser.add_argument("--audio_files", type=str, required=True, help="Comma separated list of paths to audio files")
    # Inference params
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of output tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    # Task params
    parser.add_argument("--task", type=str, default="transcribe", help="Task to perform, either: transcribe, translate2lang OR transcribe_translate2lang")
    parser.add_argument("--output", type=str, default=None, help="File to save outputs")
    parser.add_argument("--debug", action="store_true", help="Debug mode with more logging")
    args = parser.parse_args()

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Device count: {torch.cuda.device_count()}")

    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")

    # --------------------------------------------------
    # Config file
    # --------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    audio = config["audio"]
    projector = config["projector"]
    llm_path = config["llm"]["path"]
    lora_path = config["lora"]["path"]
    asr_token = config["asr_token"]
    stt_token = config["stt_token"]
    end_token = config["end_token"]

    # --------------------------------------------------
    # Task → prompt
    # --------------------------------------------------
    if args.task == "transcribe":
        prompt = f"\nTranscribe.\n{asr_token}"
    elif args.task.startswith("transcribe_translate2"):
        tgt_lang = args.task.split("2")[1]
        prompt = f"\nTranscribe then translate into {tgt_lang}.\n{asr_token}"
    elif args.task.startswith("translate2"):
        tgt_lang = args.task.split("2")[1]
        prompt = f"\nTranslate into {tgt_lang}.\n{stt_token}"
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    t0 = time.time()

    model = AudioToLLM(config, device, dtype, is_infer=True)
    logger.info(f"Loading took {time.time() - t0:.2f} sec")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    t1 = time.time()

    with open(args.output, "w", encoding="utf-8") if args.output else nullcontext() as out_file:
        for audio_file in args.audio_files.split(","):
            outputs = model.generate(
                [audio_file], 
                prompt, 
                max_new_tokens=args.max_new_tokens, 
                temperature=args.temperature, 
                top_p=args.top_p
            )
            text = outputs[0]

            if out_file:
                out_file.write(text + "\n")
            else:
                print(text)

    logger.info(f"Generation took {time.time() - t1:.2f} sec")
