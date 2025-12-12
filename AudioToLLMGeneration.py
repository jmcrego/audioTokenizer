import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from AudioEmbedder import AudioEmbedder
from AudioToLLMProjector import AudioToLLMProjector

def generate_from_audio(audio_files, prompt, 
                        audio_embedder, projector, llm, tokenizer,
                        max_output_tokens=128, temperature=0.7, 
                        device="cuda"):
    dtype = next(projector.parameters()).dtype

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
    parser.add_argument("--audio_files", type=str, help="Comma separated list of paths to audio files")
    parser.add_argument("--prompt", type=str, default="translate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler()])

    start_time = time.time()
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # --------------------------
    # 1) Initialize audio embedder
    # --------------------------
    chunk_size = 3200
    stride = 1600
    stack_size = 8
    rank_dim = 256
    max_seq_len = 1024

    audio_embedder = AudioEmbedder(
        model=args.audio_path,
        l2_norm=False,
        chunk_size=chunk_size,
        stride=stride,
        device=device,
        dtype=dtype
    )

    # --------------------------
    # 2) Initialize LLM and tokenizer
    # --------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)

    llm = LLM.from_hf(llm_model, tokenizer)

    # --------------------------
    # 3) Initialize projector
    # --------------------------
    projector = AudioToLLMProjector(
        args.proj_path,
        audio_embedding_dim=audio_embedder.D,
        stack_size=stack_size,
        llm_dimension=llm_model.config.hidden_size,
        rank_dim=rank_dim,
        max_seq_len=max_seq_len,
        device=device,
        dtype=dtype
    )

    # --------------------------
    # 4) Run generation
    # --------------------------
    audio_files = args.audio_files.split(",") if args.audio_files else []
    for output in generate_from_audio(audio_files, args.prompt, 
                                      audio_embedder, projector, llm, tokenizer, 
                                      max_output_tokens=128, temperature=0.7,
                                      device=device):
        print(output)

    logging.info(f"Total generation time: {time.time() - start_time:.2f} sec")
