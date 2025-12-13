import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from train_sft import get_device_dtype
from AudioToLLMWrapper import AudioToLLMWrapper

logger = logging.getLogger("AudioToLLMGeneration")

def generate_from_audio(audio_files, prompt, 
                        model,
                        max_output_tokens=128, temperature=0.7, 
                        device="cuda"):
    dtype = next(projector.parameters()).dtype

                        audio_embedder, projector, llm, tokenizer,
    audio_embedder = model.audio_embedder
    projector = model.projector
    llm = LLM.from_hf(model.llm_model, model.tokenizer)
    tokenizer = model.tokenizer

    # Reserve a large enough number of virtual tokens once
    max_virtual_tokens = 1024
    virtual_tokens = [f"<audio{i}>" for i in range(max_virtual_tokens)]
    tokenizer.add_tokens(virtual_tokens)
    virtual_token_ids = tokenizer.convert_tokens_to_ids(virtual_tokens)
    llm.model.resize_token_embeddings(len(tokenizer))

    for audio_file in audio_files:
        try:
            # --------------------------
            # 1) Compute audio embeddings
            # --------------------------
            with torch.no_grad():
                embs, embs_mask = audio_embedder([audio_file])
                embs = embs.to(device=device, dtype=dtype)
                embs_mask = embs_mask.bool().to(device=device)

            proj_embs, proj_mask = projector(embs, embs_mask)
            n_virtual = int(proj_mask.sum())
            proj_embs = proj_embs[0, :n_virtual, :]

            if n_virtual > max_virtual_tokens:
                raise ValueError(f"n_virtual={n_virtual} exceeds max_virtual_tokens={max_virtual_tokens}")

            # --------------------------
            # 2) Assign projected embeddings to reserved virtual tokens
            # --------------------------
            with torch.no_grad():
                llm.model.get_input_embeddings().weight[virtual_token_ids[:n_virtual]] = proj_embs.to(
                    llm.model.get_input_embeddings().weight.dtype
                )

            # --------------------------
            # 3) Build prompt and generate
            # --------------------------
            full_prompt = " ".join(virtual_tokens[:n_virtual]) + " " + prompt

            sampling_params = SamplingParams(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )

            outputs = llm.generate(full_prompt, sampling_params)
            yield outputs[0].text

        except Exception as e:
            logging.error(f"Error processing {audio_file}: {e}")
            yield None


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
    parser.add_argument("--audio_files", type=str, help="Comma separated list of paths to audio files")
    parser.add_argument("--max_output_tokens", type=int, default=128, help="Maximum number of output tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--prompt", type=str, default="\nTranscribe.\n[ASR]")
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

    # -----------------------------
    # Load model wrapper
    # -----------------------------

    model = AudioToLLMWrapper(
        audio_path=args.audio_path,
        proj_path=args.proj_path,
        llm_path=args.llm_path,
        lora_path=args.lora_path,
        chunk_size=3200,
        stride=1600,
        stack_size=8,
        rank_dim=256,
        max_seq_len=1000,
        device=device,
        dtype=dtype,
    )

    start_time = time.time()
    audio_files = args.audio_files.split(",") if args.audio_files else []

    for output in generate_from_audio(audio_files, args.prompt, max_output_tokens=args.max_output_tokens, temperature=args.temperature, device=device):
        print(output)

    logging.info(f"Total generation time: {time.time() - start_time:.2f} sec")
