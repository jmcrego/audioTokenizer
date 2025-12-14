import torch
import logging
#from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from typing import Optional

from AudioToLLMWrapper import AudioToLLMWrapper

logger = logging.getLogger("AudioToLLMGenerator")

class AudioToLLMGenerator():
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
        

