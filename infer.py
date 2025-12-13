import torch
import logging
#from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from typing import Optional

from train_sft import get_device_dtype
from AudioToLLMGenerator import AudioToLLMGenerator

logger = logging.getLogger("infer")

if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe and translate audio using AudioToLLM with vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--audio_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147")
    parser.add_argument("--proj_path", type=str, required=True)
    parser.add_argument("--llm_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--audio_files", type=str, required=True, help="Comma separated list of paths to audio files")
    parser.add_argument("--max_output_tokens", type=int, default=128, help="Maximum number of output tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--task", type=str, default="transcribe", help="Task to perform: transcribe, translate2lang, transcribe_translate2lang")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            # logging.FileHandler(log_filename, mode='w', encoding='utf-8'),  # save to file
            logging.StreamHandler()  # and print to console
        ]
    )

    logger.info("CUDA available:", torch.cuda.is_available())
    logger.info("Device count:", torch.cuda.device_count())

    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")


    if args.task == "transcribe":
        prompt = "\nTranscribe.\n[ASR]"
    elif args.task.startswith("transcribe_translate2"):
        tgt_lang = args.task.split('2')[1]
        prompt = f"\nTranscribe then translate into {tgt_lang}.\n[ASR]"
    elif args.task.startswith("translate2"):
        tgt_lang = args.task.split('2')[1]
        prompt = f"\nTranslate into {tgt_lang}.\n[STT]"


    t = time.time()

    generator = AudioToLLMGenerator(
        audio_path=args.audio_path,
        proj_path=args.proj_path,
        llm_path=args.llm_path,
        lora_path=args.lora_path,
        device=device,
        dtype=dtype,
    )

    logging.info(f"Loading took {time.time() - t:.2f} sec")

    t = time.time()

    for audio_file in args.audio_files.split(","):
        output = generator(
            audio_file, 
            args.prompt, 
            max_output_tokens=args.max_output_tokens, 
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        print(output)

    logging.info(f"Generation took {time.time() - t:.2f} sec")
