# AudioToLLM.py

import torch
import logging

from Embedder import Embedder
from Projector import Projector
from Backbone import Backbone

logger = logging.getLogger("AudioToLLM")

class AudioToLLM(torch.nn.Module):
    """
    Wrapper combining Embedder -> Projector -> LLM
    """
    def __init__(self, config, device, dtype, is_infer=False):
        super().__init__()

        # ====================================================
        # 1. Modules
        # ====================================================
        self.audio_embedder = Embedder(config['audio'])
        self.backbone = Backbone(config['llm'], config['lora'])
        self.llm_model = self.backbone.llm_model
        self.tokenizer = self.backbone.tokenizer

        self.projector = Projector(
            config['projector'], 
            audio_embedding_dim=self.audio_embedder.embedding_dim, 
            llm_embedding_dim=self.llm_model.config.hidden_size
        )

        # ====================================================
        # 2. Device & dtype
        # ====================================================
        self.audio_embedder.to(device=device, dtype=torch.float32)
        self.projector.to(device=device, dtype=torch.float32)
        self.llm_model.to(device=device, dtype=dtype)

        # ====================================================
        # 3. Freeze / train settings
        # ====================================================
        # Freeze audio embedder
        self.audio_embedder.eval()
        for p in self.audio_embedder.parameters():
            p.requires_grad = False

        if is_infer:
            # Freeze LLM and projector
            self.llm_model.eval()
            self.projector.eval()
            for p in self.llm_model.parameters():
                p.requires_grad = False
            for p in self.projector.parameters():
                p.requires_grad = False
        else:
            # Training: LLM base frozen, LoRA trainable
            self.llm_model.train()
            for n, p in self.llm_model.named_parameters():
                p.requires_grad = "lora" in n.lower()
            # Projector trainable
            self.projector.train()
            for p in self.projector.parameters():
                p.requires_grad = True
            self.llm_model.print_trainable_parameters()

        logger.info(f"Audio embedder: {next(self.audio_embedder.parameters()).dtype} on {next(self.audio_embedder.parameters()).device}")
        logger.info(f"Projector: {next(self.projector.parameters()).dtype} on {next(self.projector.parameters()).device}")
        logger.info(f"LLM: {next(self.llm_model.parameters()).dtype} on {next(self.llm_model.parameters()).device}")

    # ========================================================
    # Forward (training)
    # ========================================================
    def forward(self, audio_paths, prompt_ids, target_ids):
        """
        Forward pass for training: audio + prompt + target → LLM
        """
        device = self.llm_model.device
        llm_dtype = next(self.llm_model.parameters()).dtype
        batch_arange = torch.arange(len(audio_paths), device=device)

        # ----------------------------
        # 1) Audio embeddings (frozen)
        # ----------------------------
        with torch.no_grad():
            embs, embs_mask = self.audio_embedder(audio_paths)
            embs = embs.to(device=device)
            embs_mask = embs_mask.bool().to(device)
            # logging.info(f"embs.shape = {embs.shape}")
            # logging.info(f"embs_mask.shape = {embs_mask.shape}")

        # ----------------------------
        # 2) Projector (trainable)
        # ----------------------------
        proj_embs, proj_mask = self.projector(embs, embs_mask)
        proj_mask = proj_mask.bool()
        B, S, D = proj_embs.shape
        # logging.info(f"proj_embs.shape = {proj_embs.shape}")
        # logging.info(f"proj_mask.shape = {proj_mask.shape}")

        # ----------------------------
        # 3) Prompt embeddings (frozen)
        # ----------------------------
        prompt_ids = prompt_ids.to(device)
        T_prompt = prompt_ids.size(1)
        with torch.no_grad():
            prompt_embs = self.llm_model.get_input_embeddings()(prompt_ids)
            # logging.info(f"prompt_embs.shape = {prompt_embs.shape}")

        # ----------------------------
        # 4) Target embeddings (frozen)
        # ----------------------------
        target_ids = target_ids.to(device)
        L_labels = target_ids.size(1)
        with torch.no_grad():
            target_embs = self.llm_model.get_input_embeddings()(target_ids)
            # logging.info(f"target_embs.shape = {target_embs.shape}")

        # ----------------------------
        # 5) Lengths
        # ----------------------------
        audio_lens  = proj_mask.sum(dim=1) #[B]
        prompt_lens = (prompt_ids != self.tokenizer.pad_token_id).sum(dim=1) #[B]
        target_lens = (target_ids != self.tokenizer.pad_token_id).sum(dim=1) #[B]
        total_lens = audio_lens + prompt_lens + target_lens #[B]
        max_len = total_lens.max().item()
        # logging.info(f"max_len={max_len}")

        # ----------------------------
        # 6) Allocate tensors
        # ----------------------------
        inputs_embeds = torch.zeros((B, max_len, D), device=device, dtype=llm_dtype) #[B, max_len, D]
        attention_mask = (torch.arange(max_len, device=device).unsqueeze(0) < total_lens.unsqueeze(1)).long()
        labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)
        # logging.info(f"inputs_embeds.shape = {inputs_embeds.shape}")
        # logging.info(f"attention_mask.shape = {attention_mask.shape}")
        # logging.info(f"labels.shape = {labels.shape}")

        # ----------------------------
        # 7) Copy audio embeddings
        # ----------------------------
        audio_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        audio_valid = audio_idx < audio_lens.unsqueeze(1)
        inputs_embeds[batch_arange.unsqueeze(1).expand_as(audio_idx)[audio_valid], audio_idx[audio_valid]] = proj_embs[audio_valid].to(llm_dtype)

        # ----------------------------
        # 8) Copy prompt embeddings
        # ----------------------------
        prompt_idx = torch.arange(T_prompt, device=device).unsqueeze(0).expand(B, T_prompt)
        prompt_valid = prompt_idx < prompt_lens.unsqueeze(1)
        dest_prompt_pos = audio_lens.unsqueeze(1) + prompt_idx
        batch_idx_prompt = batch_arange.unsqueeze(1).expand_as(prompt_idx)
        inputs_embeds[batch_idx_prompt[prompt_valid], dest_prompt_pos[prompt_valid]] = prompt_embs[prompt_valid].to(llm_dtype)
        assert (dest_prompt_pos[prompt_valid] < inputs_embeds.size(1)).all()

        # ----------------------------
        # 9) Copy target embeddings and labels
        # ----------------------------
        target_idx = torch.arange(L_labels, device=device).unsqueeze(0).expand(B, L_labels)
        target_valid = target_idx < target_lens.unsqueeze(1)
        dest_target_pos = audio_lens.unsqueeze(1) + prompt_lens.unsqueeze(1) + target_idx
        inputs_embeds[batch_arange.unsqueeze(1).expand_as(target_idx)[target_valid], dest_target_pos[target_valid]] = target_embs[target_valid].to(llm_dtype)
        labels[batch_arange.unsqueeze(1).expand_as(target_idx)[target_valid], dest_target_pos[target_valid]] = target_ids[target_valid]

        # ----------------------------
        # 10) Positional IDs
        # ----------------------------
        position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)

        # ----------------------------
        # 11) LLM forward
        # ----------------------------
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    # ========================================================
    # Generate (inference)
    # ========================================================
    @torch.no_grad()
    def generate(self, audio_files: list[str], prompt: str,
                 max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95):
        """
        Generate text conditioned on audio + prompt.
        """
        device = self.llm_model.device
        dtype = next(self.projector.parameters()).dtype
        B = len(audio_files)

        # ----------------------------
        # 1) Audio → Projector embeddings
        # ----------------------------
        audio_embs, audio_mask = self.audio_embedder(audio_files)
        audio_embs = audio_embs.to(device=device, dtype=dtype)
        audio_mask = audio_mask.to(device=device).bool()

        proj_embs, proj_mask = self.projector(audio_embs, audio_mask)
        proj_embs = proj_embs.to(device=device, dtype=dtype)
        proj_mask = proj_mask.bool()

        audio_lens = proj_mask.sum(dim=1)
        D = proj_embs.size(-1)

        # ----------------------------
        # 2) Prompt → embeddings
        # ----------------------------
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False).input_ids.to(device)
        T_prompt = prompt_ids.size(1)
        prompt_embs = self.llm_model.get_input_embeddings()(prompt_ids).expand(B, -1, -1)


        # correct norm of proj_embs compared to prompt_embs
        target_norm = prompt_embs.norm(dim=-1).mean().detach()
        proj_embs = proj_embs / proj_embs.norm(dim=-1, keepdim=True) * target_norm

        logger.debug(
            f"proj norm={proj_embs.norm(dim=-1).mean():.2f}, "
            f"text norm={prompt_embs.norm(dim=-1).mean():.2f}"
        )

        llm_dtype = next(self.llm_model.parameters()).dtype

        # ----------------------------
        # 3) Allocate input embeddings & attention mask
        # ----------------------------
        total_lens = audio_lens + T_prompt
        max_len = total_lens.max().item()
        inputs_embeds = torch.zeros((B, max_len, D), device=device, dtype=llm_dtype)
        attention_mask = (torch.arange(max_len, device=device).unsqueeze(0) < total_lens.unsqueeze(1)).long()

        # ----------------------------
        # 4) Copy audio embeddings
        # ----------------------------
        batch_idx = torch.arange(B, device=device)
        T_audio = proj_embs.size(1)
        audio_idx = torch.arange(T_audio, device=device).unsqueeze(0).expand(B, T_audio)
        audio_valid = audio_idx < audio_lens.unsqueeze(1)
        inputs_embeds[batch_idx.unsqueeze(1).expand_as(audio_idx)[audio_valid], audio_idx[audio_valid]] = proj_embs[audio_valid].to(llm_dtype)

        # ----------------------------
        # 5) Copy prompt embeddings
        # ----------------------------
        dest_prompt_pos = audio_lens.unsqueeze(1) + torch.arange(T_prompt, device=device).unsqueeze(0)
        inputs_embeds[batch_idx.unsqueeze(1).expand_as(dest_prompt_pos), dest_prompt_pos] = prompt_embs.to(llm_dtype)

        # ----------------------------
        # 6) Positional IDs
        # ----------------------------
        position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)

        # ----------------------------
        # 7) Generate
        # ----------------------------
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        # ----------------------------
        # 7) Slice generated tokens
        # ----------------------------
        gen_tokens = outputs[:, max_len:]  # exclude prefix
        texts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        logger.info(f"Generated text: {texts[0] if len(texts) > 0 else ''}")
        return texts
