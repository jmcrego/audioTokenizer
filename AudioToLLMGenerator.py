import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from typing import Optional

from train_sft import get_device_dtype
from AudioToLLMWrapper import AudioToLLMWrapper

logger = logging.getLogger("AudioToLLMGeneration")

class AudioToLLMGeneration():
    """
    Generate text from audio using AudioToLLM model.
    """

    def __init__(
        self,
        audio_path: str,
        proj_path: str,
        llm_path: str,
        lora_path: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
    ):
        # the next should be read from wrapper config
        # chunk_size: int,
        # stride: int,
        # stack_size: int,
        # rank_dim: int,
        # max_seq_len: int,

        model = AudioToLLMWrapper(
            audio_path=audio_path,
            proj_path=proj_path,
            llm_path=llm_path,
            lora_path=lora_path,
            chunk_size=3200,
            stride=1600,
            stack_size=8,
            rank_dim=256,
            max_seq_len=1024,
            device=device,
            dtype=dtype,
        )

        self.audio_embedder = model.audio_embedder
        self.projector = model.projector
        self.llm = LLM.from_hf(model.llm_model, model.tokenizer)
        self.tokenizer = model.tokenizer

        # Reserve a large enough number of virtual tokens once
        max_virtual_tokens = 1024
        virtual_tokens = [f"<audio{i}>" for i in range(max_virtual_tokens)]
        self.tokenizer.add_tokens(virtual_tokens)
        self.virtual_token_ids = self.tokenizer.convert_tokens_to_ids(virtual_tokens)
        self.llm.model.resize_token_embeddings(len(self.tokenizer))

    def __call__(
        self, 
        audio_file, 
        prompt, 
        max_output_tokens=128, 
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    ):
        dtype = next(self.projector.parameters()).dtype
        device = next(self.projector.parameters()).device

        # --------------------------
        # Compute audio embeddings
        # --------------------------
        with torch.no_grad():
            embs, embs_mask = self.audio_embedder([audio_file])
            embs = embs.to(device=device, dtype=dtype)
            embs_mask = embs_mask.bool().to(device=device)

        proj_embs, proj_mask = self.projector(embs, embs_mask)
        n_virtual = int(proj_mask.sum())
        proj_embs = proj_embs[0, :n_virtual, :]

        if n_virtual > len(self.virtual_token_ids):
            raise ValueError(f"n_virtual={n_virtual} exceeds max_virtual_tokens={len(self.virtual_token_ids)}")

        # --------------------------
        # Assign projected embeddings to reserved virtual tokens
        # --------------------------
        with torch.no_grad():
            self.llm.model.get_input_embeddings().weight[self.virtual_token_ids[:n_virtual]] = proj_embs.to(
                self.llm.model.get_input_embeddings().weight.dtype
            )

        # --------------------------
        # Build prompt and generate
        # --------------------------
        full_prompt = " ".join([f"<audio{i}>" for i in range(n_virtual)]) + " " + prompt

        sampling_params = SamplingParams(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=["[END]"]
        )

        outputs = self.llm.generate(full_prompt, sampling_params)
        return outputs[0].text
        


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe and translate audio using AudioToLLM with vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--audio_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147")
    parser.add_argument("--proj_path", type=str, required=True)
    parser.add_argument("--llm_path", type=str, required=True)
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

    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
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

    generator = AudioToLLMGeneration(
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
        ):
        print(output)

    logging.info(f"Generation took {time.time() - t:.2f} sec")
