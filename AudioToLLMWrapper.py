
import torch
import logging
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from AudioEmbedder import AudioEmbedder
from AudioToLLMProjector import AudioToLLMProjector

logger = logging.getLogger("AudioToLLMWrapper")

class AudioToLLMWrapper(torch.nn.Module):
    """
    Wrapper combining AudioEmbedder -> Projector -> LLM
    Only Projector is trainable.
    """
    def __init__(self, audio_path, proj_path, llm_path, chunk_size, stride, stack_size, rank_dim, max_seq_len, device, dtype):

        meta = {k: v for k, v in locals().items() if k != "self" and k != "__class__"}
        logger.info(f"Initializing {meta}")        
        super().__init__()

        ############################
        # Audio Embedder (frozen)
        ############################
        self.audio_embedder = AudioEmbedder(
            model=audio_path,
            l2_norm=False,
            chunk_size=chunk_size,
            stride=stride,
            device=device,
            dtype=dtype
        )
        self.audio_embedder.eval()
        for p in self.audio_embedder.parameters():
            p.requires_grad = False

        ############################
        # LLM (frozen) LoRa (trainable) + Tokenizer
        ############################
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path, dtype=dtype, low_cpu_mem_usage=True).to(device)

        lora_r = 16
        lora_alpha = 32
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_dropout = 0.05
        bias = "none"
        task_type = "CAUSAL_LM"

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
        )
        # Apply LoRa to LLM
        self.llm_model = get_peft_model(self.llm_model, lora_config)

        # Freeze base model, keep LoRA trainable
        for n, p in self.llm_model.named_parameters():
            p.requires_grad = ("lora" in n.lower())

        # self.llm_model.eval()
        # for p in self.llm_model.parameters():
        #     p.requires_grad = False

        # trl’s SFTTrainer expects a HuggingFace model, which always has:
        # self.config = self.llm_model.config
        # self.generation_config = getattr(self.llm_model, "generation_config", None)

        ############################
        # Projector (trainable)
        ############################
        self.projector = AudioToLLMProjector(
            proj_path,
            audio_embedding_dim=self.audio_embedder.D,
            stack_size=stack_size,
            llm_dimension=self.llm_model.config.hidden_size,
            rank_dim=rank_dim,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )

        # Ensure projector is trainable
        for p in self.projector.parameters():
            p.requires_grad = True

        logger.info(f"Read ModelsWrapper")


    def forward(self, audio_paths, prompt_ids, target_ids):
        """
        Vectorized forward pass for TRAINING.
        Always assumes target_ids is provided.
        
        audio_paths: list[str], length B
        prompt_ids: [B, T_prompt]
        target_ids: [B, L_labels]

        Returns:
            loss, logits, labels, attention_mask
        """

        device = self.llm_model.device
        dtype  = next(self.projector.parameters()).dtype

        # --------------------------------------------------------
        # 1) AUDIO → EMBEDDINGS (FROZEN)
        # --------------------------------------------------------
        with torch.no_grad():
            embs, embs_mask = self.audio_embedder(audio_paths)
            # embs: [B, S, D_audio]
            # mask: [B, S]

            embs = embs.to(device=device, dtype=dtype)
            embs_mask = embs_mask.bool().to(device)
            logger.debug(f"Audio embeddings: {embs.shape} dtype={embs.dtype} mask={embs_mask.shape}")

        # --------------------------------------------------------
        # 2) PROJECTOR (TRAINABLE)
        # --------------------------------------------------------
        proj_embs, proj_mask = self.projector(embs, embs_mask)
        proj_mask = proj_mask.bool()
        proj_embs = proj_embs.to(dtype)
        logger.debug(f"Projected embeddings: {proj_embs.shape} dtype={proj_embs.dtype} mask={proj_mask.shape} invalid embeddings={(~proj_mask).sum().item()}")

        B, S, D = proj_embs.shape

        # --------------------------------------------------------
        # 3) PROMPT EMBEDDINGS (FROZEN)
        # --------------------------------------------------------
        prompt_ids = prompt_ids.to(device)
        T_prompt = prompt_ids.size(1)
        logger.debug(f"Prompt ids: {prompt_ids.shape} dtype={prompt_ids.dtype}")

        with torch.no_grad():
            prompt_embs = self.llm_model.get_input_embeddings()(prompt_ids)
            prompt_embs = prompt_embs.to(device=device, dtype=dtype)
            logger.debug(f"Prompt embeddings: {prompt_embs.shape} dtype={prompt_embs.dtype}")

        # --------------------------------------------------------
        # 4) TARGET EMBEDDINGS (FROZEN)
        # --------------------------------------------------------
        target_ids = target_ids.to(device)
        L_labels = target_ids.size(1)
        logger.debug(f"Target ids: {target_ids.shape} dtype={target_ids.dtype}")

        with torch.no_grad():
            target_embs = self.llm_model.get_input_embeddings()(target_ids)
            target_embs = target_embs.to(device=device, dtype=dtype)
            logger.debug(f"Target embeddings: {target_embs.shape} dtype={target_embs.dtype}")

        # --------------------------------------------------------
        # 5) LENGTHS
        # --------------------------------------------------------
        audio_lens  = proj_mask.sum(dim=1)                                   # [B]
        prompt_lens = (prompt_ids != self.tokenizer.pad_token_id).sum(dim=1) # [B]
        target_lens = (target_ids != self.tokenizer.pad_token_id).sum(dim=1) # [B]

        total_lens = audio_lens + prompt_lens + target_lens
        max_len = total_lens.max().item()
        logger.debug(f"Lengths: audio_lens={audio_lens} prompt_lens={prompt_lens} target_lens={target_lens} total_lens={total_lens} max_len={max_len}")

        # --------------------------------------------------------
        # 6) Allocate final tensors
        # --------------------------------------------------------
        inputs_embeds = torch.zeros((B, max_len, D), device=device, dtype=dtype)
        attention_mask = ( torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1) < total_lens.unsqueeze(1) ).long()
        logger.debug(f"Allocate final inputs_embeds: {inputs_embeds.shape} dtype={inputs_embeds.dtype} attention_mask={attention_mask.shape}")

        ignore_index = -100
        labels = torch.full((B, max_len), ignore_index, device=device, dtype=torch.long)
        logger.debug(f"Allocate final labels: {labels.shape} dtype={labels.dtype} ignore_index={ignore_index}")

        # --------------------------------------------------------
        # 7) CONCATENATION
        # --------------------------------------------------------
        # Concatenate audio embeddings + prompt embeddings (right-padded)
        # and return final padded inputs_embeds and labels.

        # For instance,
        # Input EMBEDDINGS should be:
        # [ a a a a p p p 0 0 0 0 0 0]
        # [ a a p p 0 0 0 0 0 0 0 0 0]
        # [ a a a p p p 0 0 0 0 0 0 0]
        # (a means audio embedding, p means prompt embedding 0 means pad embedding)

        # While LABELS should be:
        # [ -100 -100 -100 -100 -100 -100 -100 -100 t t t    t -100]
        # [ -100 -100 -100 -100 -100 -100 -100 -100 t t t -100 -100]
        # [ -100 -100 -100 -100 -100 -100 -100 -100 t t t    t    t]
        # (t means label token)

        batch_arange = torch.arange(B, device=device)

        # AUDIO
        audio_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        audio_valid = audio_idx < audio_lens.unsqueeze(1)
        inputs_embeds[ batch_arange.unsqueeze(1).expand_as(audio_idx)[audio_valid], audio_idx[audio_valid] ] = proj_embs[audio_valid]

        # PROMPT
        prompt_idx = torch.arange(T_prompt, device=device).unsqueeze(0).expand(B, T_prompt)
        prompt_valid = prompt_idx < prompt_lens.unsqueeze(1)
        dest_prompt_pos = audio_lens.unsqueeze(1) + prompt_idx
        batch_idx_prompt = batch_arange.unsqueeze(1).expand_as(prompt_idx)
        inputs_embeds[ batch_idx_prompt[prompt_valid], dest_prompt_pos[prompt_valid] ] = prompt_embs[prompt_valid]

        # TARGET
        target_idx = torch.arange(L_labels, device=device).unsqueeze(0).expand(B, L_labels)
        target_valid = target_idx < target_lens.unsqueeze(1)
        dest_target_pos = audio_lens.unsqueeze(1) + prompt_lens.unsqueeze(1) + target_idx
        batch_idx_target = batch_arange.unsqueeze(1).expand_as(target_idx)
        inputs_embeds[ batch_idx_target[target_valid], dest_target_pos[target_valid] ] = target_embs[target_valid]        

        # LABELS 
        labels[ batch_idx_target[target_valid], dest_target_pos[target_valid] ] = target_ids[target_valid]

        # --------------------------------------------------------
        # 8) LLM FORWARD
        # --------------------------------------------------------
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        logger.debug(f"LLM outputs: loss={outputs.loss} logits={outputs.logits.shape} dtype={outputs.logits.dtype}")

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "labels": labels,
            "attention_mask": attention_mask,
        }


