# infer.py

import torch
import json
import time
import logging
import argparse
from contextlib import nullcontext

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from train import get_device_dtype
from AudioToLLM import AudioToLLM
from Dataset import BatchedLengthSampler 
from Dataset import Dataset


logger = logging.getLogger("infer")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Transcribe and/or translate audio using AudioToLLM (Hugging Face).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, required=True, help="Model config file")
    parser.add_argument("--test", type=str, required=True, help="Testing dataset file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    # Inference params
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of output tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    # Task params
    #parser.add_argument("--task", type=str, default="transcribe", help="Task to perform, either: transcribe, translate2lang OR transcribe_translate2lang")
    #parser.add_argument("--output", type=str, default=None, help="File to save outputs")
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
    logger.info(f"Config: {config}")

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    t = time.time()

    model = AudioToLLM(config, device, dtype, is_infer=True)
    model.eval()
    logger.info(f"Loading model took {time.time() - t:.2f} sec")

    # -------------------------------------------------- 
    # Load dataset
    # --------------------------------------------------
    def collate_fn(batch):
        pad_token_id = model.tokenizer.pad_token_id
        audio_paths = [x["audio_path"] for x in batch]
        def ensure_tensor(x):
            return x.detach().clone() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)
        prompt_ids = pad_sequence([ensure_tensor(x["prompt_ids"]) for x in batch], batch_first=True, padding_value=pad_token_id)
        target_ids = pad_sequence([ensure_tensor(x["target_ids"]) for x in batch], batch_first=True, padding_value=pad_token_id)
        return {
            "audio_paths": audio_paths,
            "prompt_ids": prompt_ids,
            "target_ids": target_ids
        }

    test_dataset = Dataset(
        file_path=args.test,
        tokenizer=model.tokenizer,
        audio_token=config["llm"]["audio_token"],
        sample_rate=model.audio_embedder.sample_rate,
        downsample_ratio=model.audio_embedder.downsample_ratio,
        conv_stride=config["projector"]["conv_stride"],
        max_seq_len=args.max_seq_len
    )

    test_sampler = BatchedLengthSampler(test_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn
    )
    
    logger.info(f"Initialized Sampler and DataLoader for test with batch_size={args.batch_size} with {len(test_dataset)} samples")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    t = time.time()

    with torch.no_grad():
        for n_batch, batch in enumerate(test_loader):
            # ----------------------------
            # Move tensors to device
            # ----------------------------
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            audio_paths = batch["audio_paths"]

            # ----------------------------
            # 1) Embed audio
            # ----------------------------
            with torch.amp.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(**batch)

            prompt_ids = batch["prompt_ids"]
            target_ids = batch["target_ids"]

            # Decode prompt text (for logging only)
            prompt_texts = model.tokenizer.batch_decode(
                batch["prompt_ids"],
                skip_special_tokens=False,
            )

            # Decode targets (ground truth)
            target_texts = model.tokenizer.batch_decode(
                batch["target_ids"],
                skip_special_tokens=False,
            )

            # ----------------------------
            # 2) Run generation
            # ----------------------------
            genera_texts = model.generate(
                audio_files=audio_paths,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            B = len(audio_paths)
            for i in range(B):
                logger.info(f"nBatch: {n_batch} nSample: {i}")
                logger.info(f"AUDIO: {audio_paths[i]}")
                def replace_CR(text):
                    return text.replace("\n", "↵") if text is not None else None
                logger.info(f"TARGET: {replace_CR(target_texts[i])}")
                logger.info(f"PROMPT: {replace_CR(prompt_texts[i])}")
                logger.info(f"PREDIC: {replace_CR(genera_texts[i])}")
                logger.info("=" * 80)
                print(genera_texts[i])

    logger.info(f"Generation took {time.time() - t:.2f} sec")
