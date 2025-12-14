import torch
import logging
from vllm import LLM, SamplingParams
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from AudioToLLMWrapper import AudioToLLMWrapper
from AudioEmbedder import AudioEmbedder
from AudioToLLMProjector import AudioToLLMProjector

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
#        lora_path: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
    ):

        chunk_size = 3200
        stride = 1600
        stack_size = 8
        rank_dim = 256

        self.audio_embedder = AudioEmbedder(
            audio_path=audio_path,
            l2_norm=False,
            chunk_size=chunk_size,
            stride=stride,
        )
        # Move to correct device and dtype
        self.audio_embedder.to(device=device, dtype=dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path)
        llm_model_hidden_size = self.llm_model.config.hidden_size
        
        self.llm_model_embedder = self.llm_model.get_input_embeddings()
        self.llm_model_embedder.to(device=device, dtype=dtype)
        #remove llm model from cpu memory (only need embeddings)
        del self.llm_model

        self.projector = AudioToLLMProjector(
            proj_path=proj_path,
            audio_embedding_dim=self.audio_embedder.D,
            stack_size=stack_size,
            rank_dim=rank_dim,
            llm_dimension=llm_model_hidden_size,
        )
        self.projector.to(device=device, dtype=dtype)

        llm_kwargs = {
            "model": llm_path,
            "dtype": "float16",  # V100 only supports float16, not bfloat16
            "enforce_eager": True,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.9,  # Adjust based on your GPU memory
        }
        
        # if max_model_len is not None:
        #     llm_kwargs["max_model_len"] = max_model_len
        
        self.llm = LLM(**llm_kwargs)


        # the next should be read from wrapper config
        # chunk_size: int,
        # stride: int,
        # stack_size: int,
        # rank_dim: int,
        # max_seq_len: int,

        # model = AudioToLLMWrapper(
        #     audio_path=audio_path,
        #     proj_path=proj_path,
        #     llm_path=llm_path,
        #     lora_path=lora_path,
        #     chunk_size=3200,
        #     stride=1600,
        #     stack_size=8,
        #     rank_dim=256,
        #     max_seq_len=1024,
        #     device=device,
        #     dtype=dtype,
        # )

        # self.audio_embedder = model.audio_embedder
        # self.projector = model.projector
        # # Load LLM with vLLM
        # self.llm = LLM(model=llm_path, enable_prompt_embeds=True)
        # self.tokenizer = model.tokenizer

        # Reserve a large enough number of virtual tokens once
        # max_virtual_tokens = 1024
        # virtual_tokens = [f"<audio{i}>" for i in range(max_virtual_tokens)]
        # self.tokenizer.add_tokens(virtual_tokens)
        # self.virtual_token_ids = self.tokenizer.convert_tokens_to_ids(virtual_tokens)
        # self.llm.model.resize_token_embeddings(len(self.tokenizer))

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
            embs, embs_mask = self.audio_embedder([audio_file]) # [1, T, D], [1, T]
            embs = embs.to(device=device, dtype=dtype) 
            embs_mask = embs_mask.bool().to(device=device) 

        proj_embs, proj_mask = self.projector(embs, embs_mask) # [1, N, llm_dim], [1, N]

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.long().to(device=device) # [1, L]
        prompt_embs = self.llm_model_embedder(prompt_ids) # [1, L, llm_dim]

        # remove masked projected embeddings
        proj_embs = proj_embs[proj_mask] # [N_valid, llm_dim]
        proj_embs = proj_embs.unsqueeze(0) # [1, N_valid, llm_dim]

        # Concatenate audio + text
        combined_embs = torch.cat([proj_embs, prompt_embs], dim=1) # [1, N_valid + L, llm_dim]

        sampling_params = SamplingParams(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=["[END]"]
        )

        outputs = self.llm.generate(
            prompts={
                "prompt_token_ids": [],  # Empty token IDs
                "prompt_embeds": combined_embs.cpu()  # Pass embeddings
            },
            sampling_params=sampling_params
        )

        # n_virtual = int(proj_mask.sum())
        # proj_embs = proj_embs[0, :n_virtual, :]

        # if n_virtual > len(self.virtual_token_ids):
        #     raise ValueError(f"n_virtual={n_virtual} exceeds max_virtual_tokens={len(self.virtual_token_ids)}")

        # --------------------------
        # Assign projected embeddings to reserved virtual tokens
        # --------------------------
        # with torch.no_grad():
        #     self.llm.model.get_input_embeddings().weight[self.virtual_token_ids[:n_virtual]] = proj_embs.to(
        #         self.llm.model.get_input_embeddings().weight.dtype
        #     )

        # --------------------------
        # Build prompt and generate
        # --------------------------
        # full_prompt = " ".join([f"<audio{i}>" for i in range(n_virtual)]) + " " + prompt


        # sampling_params = SamplingParams(
        #     temperature=temperature,
        #     max_output_tokens=max_output_tokens,
        #     top_p=top_p,
        #     top_k=top_k,
        #     stop=["[END]"]
        # )

        # outputs = self.llm.generate(full_prompt, sampling_params)
        return outputs[0].text
        

