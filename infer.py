import torch
import time
import logging
import argparse
from contextlib import nullcontext

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from train import get_device_dtype
from AudioToLLMGeneratorHF import AudioToLLMGeneratorHF
from AudioEmbedder import AudioEmbedder
from AudioToLLMProjector import AudioToLLMProjector

logger = logging.getLogger("infer")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Transcribe and/or translate audio using AudioToLLM (Hugging Face).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--audio_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147")
    parser.add_argument("--proj_path", type=str, required=True)
    parser.add_argument("--llm_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--audio_files", type=str, required=True, help="Comma separated list of paths to audio files")
    # Inference params
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum number of output tokens to generate")
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
    # Task → prompt
    # --------------------------------------------------
    if args.task == "transcribe":
        prompt = "\nTranscribe.\n[ASR]"
    elif args.task.startswith("transcribe_translate2"):
        tgt_lang = args.task.split("2")[1]
        prompt = f"\nTranscribe then translate into {tgt_lang}.\n[ASR]"
    elif args.task.startswith("translate2"):
        tgt_lang = args.task.split("2")[1]
        prompt = f"\nTranslate into {tgt_lang}.\n[STT]"
    else:
        raise ValueError(f"Unknown task: {args.task}")

    chunk_size = 3200  # must match audio embedder
    stride = 1600      # must match audio embedder
    stack_size = 8     # must match projector
    rank_dim = 256     # must match projector

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    t0 = time.time()

    audio_embedder = AudioEmbedder(
        audio_path=args.audio_path,
        l2_norm=False,
        chunk_size=chunk_size,
        stride=stride,
    )
    audio_embedder.to(device=device, dtype=dtype)    
    audio_embedder.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    llm_model.to(device=device)
    logger.info(f"Loaded llm_model from {args.llm_path}")

    if args.lora_path is not None:
        llm_model = PeftModel.from_pretrained(
            llm_model,
            args.lora_path,
        )
        llm_model.to(device=device, dtype=dtype)
        llm_model.eval()
        logger.info(f"Loaded LoRA adapters from {args.lora_path}")

    projector = AudioToLLMProjector(
        proj_path=args.proj_path,
        audio_embedding_dim=audio_embedder.D,
        stack_size=stack_size,
        rank_dim=rank_dim,
        llm_dimension=llm_model.config.hidden_size,
    )
    projector.to(device=device, dtype=dtype)

    generator = AudioToLLMGeneratorHF(
        model=llm_model,
        tokenizer=tokenizer,
        audio_embedder=audio_embedder,
        projector=projector,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    logger.info(f"Loading took {time.time() - t0:.2f} sec")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    t1 = time.time()

    with open(args.output, "w", encoding="utf-8") if args.output else nullcontext() as out_file:
        for audio_file in args.audio_files.split(","):
            outputs = generator.generate([audio_file], prompt)
            text = outputs[0]

            if out_file:
                out_file.write(text + "\n")
            else:
                print(text)

    logger.info(f"Generation took {time.time() - t1:.2f} sec")
