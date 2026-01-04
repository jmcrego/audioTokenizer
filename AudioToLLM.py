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

        self.audio_token = config["llm"]["audio_token"]
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        assert self.audio_token_id is not None, "audio_token_id is None"
        assert isinstance(self.audio_token_id, int), type(self.audio_token_id)
        logger.info(f"Audio token: '{self.audio_token}' -> ID: {self.audio_token_id}")

        self.projector = Projector(
            config['projector'], 
            audio_embedding_dim=self.audio_embedder.embedding_dim, 
            llm_embedding_dim=self.llm_model.config.hidden_size
        )

        # ====================================================
        # 2. Device & dtype
        # ====================================================
        logger.info(f"Moving models to device={device}, dtype={dtype}")
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
        logger.info("Audio embedder frozen (eval mode)")

        if is_infer:
            logger.info("Inference mode: freezing all models")
            # Freeze LLM and projector
            self.llm_model.eval()
            self.projector.eval()
            for p in self.llm_model.parameters():
                p.requires_grad = False
            for p in self.projector.parameters():
                p.requires_grad = False
        else:
            logger.info("Training mode: LLM base frozen, LoRA + Projector trainable")
            # Training: LLM base frozen, LoRA trainable
            self.llm_model.train()
            for n, p in self.llm_model.named_parameters():
                p.requires_grad = "lora" in n.lower()
            # Projector trainable
            self.projector.train()
            for p in self.projector.parameters():
                p.requires_grad = True

            # Print LoRA info
            if hasattr(self.llm_model, 'print_trainable_parameters'):
                self.llm_model.print_trainable_parameters()


        logger.info(f"Audio embedder: {next(self.audio_embedder.parameters()).dtype} on {next(self.audio_embedder.parameters()).device}")
        logger.info(f"Projector: {next(self.projector.parameters()).dtype} on {next(self.projector.parameters()).device}")
        logger.info(f"LLM: {next(self.llm_model.parameters()).dtype} on {next(self.llm_model.parameters()).device}")

        self.summary()

    def summary(self):
        """Log AudioToLLM model parameter summary"""      
          
        # Embedder
        embedder_total = sum(p.numel() for p in self.audio_embedder.parameters())
        embedder_trainable = sum(p.numel() for p in self.audio_embedder.parameters() if p.requires_grad)
        
        # Projector
        projector_total = sum(p.numel() for p in self.projector.parameters())
        projector_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        
        # LLM
        llm_total = sum(p.numel() for p in self.llm_model.parameters())
        llm_trainable = sum(p.numel() for p in self.llm_model.parameters() if p.requires_grad)
        
        # Total
        total = embedder_total + projector_total + llm_total
        trainable = embedder_trainable + projector_trainable + llm_trainable
        frozen = total - trainable
        
        logger.info("-" * 80)
        logger.info("AudioToLLM model parameter summary")
        logger.info("-" * 80)
        logger.info(f"Audio Embedder : {embedder_total:>12,} total | {embedder_trainable:>12,} trainable | {embedder_total - embedder_trainable:>12,} frozen")
        logger.info(f"Projector      : {projector_total:>12,} total | {projector_trainable:>12,} trainable | {projector_total - projector_trainable:>12,} frozen")
        logger.info(f"LLM (+ LoRA)   : {llm_total:>12,} total | {llm_trainable:>12,} trainable | {llm_total - llm_trainable:>12,} frozen")
        logger.info("-" * 80)
        logger.info(f"TOTAL          : {total:>12,} total | {trainable:>12,} trainable | {frozen:>12,} frozen")
        logger.info(f"Trainable %    : {100 * trainable / total:.2f}%")
        logger.info("-" * 80)

    # ========================================================
    # Forward (training)
    # ========================================================
    def forward(self, audio_paths, prompt_ids, target_ids):
        """
        Fully vectorized forward pass:
        audio_paths + prompt_ids (with <extra_id_0> token) + target_ids -> LLM
        """
        assert prompt_ids.dtype == torch.long
        assert prompt_ids.dim() == 2
        assert target_ids.dtype == torch.long
        assert target_ids.dim() == 2

        device = self.llm_model.device
        llm_dtype = next(self.llm_model.parameters()).dtype
        B = len(audio_paths)
        
        # ----------------------------
        # 1) Audio embeddings (frozen) + projection
        # ----------------------------
        with torch.no_grad():
            embs, embs_mask = self.audio_embedder(audio_paths)
            embs = embs.to(device=device)
            embs_mask = embs_mask.bool().to(device)

        proj_embs, proj_mask = self.projector(embs, embs_mask)
        proj_mask = proj_mask.bool()
        B, S, D = proj_embs.shape
        audio_lens = proj_mask.sum(dim=1)  # [B]

        # ----------------------------
        # 2) Prompt embeddings (frozen)
        # ----------------------------
        prompt_ids = prompt_ids.to(device)
        with torch.no_grad():
            prompt_embs = self.llm_model.get_input_embeddings()(prompt_ids)  # [B, T_prompt, D]

        audio_token_mask = (prompt_ids == self.audio_token_id)
        assert (audio_token_mask.sum(dim=1) == 1).all(), "Each prompt must have exactly one <extra_id_0> token"
        audio_pos = audio_token_mask.float().argmax(dim=1)  # [B]

        prompt_mask = (prompt_ids != self.tokenizer.pad_token_id)
        prompt_lens = prompt_mask.sum(dim=1)

        # ----------------------------
        # 3) Target embeddings (frozen)
        # ----------------------------
        target_ids = target_ids.to(device)
        with torch.no_grad():
            target_embs = self.llm_model.get_input_embeddings()(target_ids)

        target_mask = (target_ids != self.tokenizer.pad_token_id)
        target_lens = target_mask.sum(dim=1)

        # ----------------------------
        # 4) Total lengths
        # ----------------------------
        total_lens = audio_lens + prompt_lens + target_lens
        max_len = total_lens.max().item()
        inputs_embeds = torch.zeros((B, max_len, D), device=device, dtype=llm_dtype)
        labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)

        # ----------------------------
        # 5) Positional IDs
        # ----------------------------
        position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)

        # ----------------------------
        # 6) Insert prompt embeddings
        # ----------------------------
        T_prompt = prompt_ids.size(1)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T_prompt)
        inputs_embeds[:, :T_prompt] = prompt_embs

        # ----------------------------
        # 7) Safe vectorized audio insertion
        # ----------------------------
        B, S_max, D = proj_embs.shape
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, S_max)  # [B, S_max]
        range_S = torch.arange(S_max, device=device).unsqueeze(0).expand(B, S_max)

        # Only keep valid audio positions
        valid_audio = range_S < audio_lens.unsqueeze(1)          # [B, S_max]
        dest_pos_audio = audio_pos.unsqueeze(1) + range_S        # [B, S_max]

        # Clip to max_len (safe)
        dest_pos_audio = torch.clamp(dest_pos_audio, max=max_len-1)

        # Flatten for advanced indexing
        flat_batch = batch_idx[valid_audio]
        flat_pos   = dest_pos_audio[valid_audio]
        flat_embs  = proj_embs[valid_audio]

        inputs_embeds[flat_batch, flat_pos] = flat_embs

        # Inputs Embeds (B × L)
        # Batch 0: [ P, A, A, A, P, P, P, P, T, T, T, 0, 0, 0 ]
        # Batch 1: [ P, A, A, A, A, P, P, P, T, T, T, T, T, T ]
        # P = prompt embedding
        # A = audio embedding
        # 0 = padding embedding (zero vector)

        # Labels (B × L)
        # Batch 0: [ -, -, -, -, -, -, -, -, T, T, T, -, -, - ]
        # Batch 1: [ -, -, -, -, -, -, -, -, T, T, T, T, T, T ]
        # T = target embedding
        # - = label ignore (-100)

        # ----------------------------
        # 8) Safe vectorized target insertion
        # ----------------------------
        B, L_target, D = target_embs.shape
        batch_idx_t = torch.arange(B, device=device).unsqueeze(1).expand(B, L_target)
        range_L = torch.arange(L_target, device=device).unsqueeze(0).expand(B, L_target)

        valid_target = range_L < target_lens.unsqueeze(1)
        dest_pos_target = audio_lens.unsqueeze(1) + prompt_lens.unsqueeze(1) + range_L
        dest_pos_target = torch.clamp(dest_pos_target, max=max_len-1)

        flat_batch_t = batch_idx_t[valid_target]
        flat_pos_t   = dest_pos_target[valid_target]
        flat_embs_t  = target_embs[valid_target]
        flat_ids_t   = target_ids[valid_target]

        inputs_embeds[flat_batch_t, flat_pos_t] = flat_embs_t
        labels[flat_batch_t, flat_pos_t] = flat_ids_t

        # ----------------------------
        # 9) Safe attention mask
        # ----------------------------
        attention_mask = torch.zeros((B, max_len), device=device, dtype=torch.long)

        # Audio
        attention_mask[flat_batch, flat_pos] = 1

        # Prompt (excluding <extra_id_0>)
        prompt_no_audio = prompt_mask & ~audio_token_mask
        batch_idx_prompt = torch.arange(B, device=device).unsqueeze(1).expand(B, T_prompt)
        range_prompt = torch.arange(T_prompt, device=device).unsqueeze(0).expand(B, T_prompt)
        valid_prompt = prompt_no_audio
        flat_batch_p = batch_idx_prompt[valid_prompt]
        flat_pos_p   = range_prompt[valid_prompt]
        attention_mask[flat_batch_p, flat_pos_p] = 1

        # Target
        flat_batch_t = batch_idx_t[valid_target]
        flat_pos_t   = dest_pos_target[valid_target]
        attention_mask[flat_batch_t, flat_pos_t] = 1

        # ----------------------------
        # 10) Compute norms safely
        # ----------------------------
        audio_norm = flat_embs.norm(dim=-1).mean() if flat_embs.numel() > 0 else torch.tensor(0.0, device=device)
        text_norm  = prompt_embs[prompt_mask].norm(dim=-1).mean()

        # ----------------------------
        # 11) LLM forward
        # ----------------------------
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio_norm": audio_norm,
            "text_norm": text_norm,
        }


    # ========================================================
    # Generate (inference)
    # ========================================================
    @torch.no_grad()
    def generate(self, audio_files, prompt_ids, max_new_tokens=256, temperature=0.7, top_p=0.95):
        """
        Batched generation with per-sample prompts containing exactly one <extra_id_0> token.
        """
        device = self.llm_model.device
        dtype = next(self.projector.parameters()).dtype
        B = len(audio_files)
        assert prompt_ids.size(0) == B, f"audio_files length ({B}) and prompt_ids batch size ({prompt_ids.size(0)}) must match"

        # ----------------------------
        # 1) Audio → Embedding → Projector
        # ----------------------------
        audio_embs, audio_mask = self.audio_embedder(audio_files)
        audio_embs = audio_embs.to(device=device, dtype=dtype)
        audio_mask = audio_mask.bool().to(device)

        proj_embs, proj_mask = self.projector(audio_embs, audio_mask)
        proj_embs = proj_embs.to(device=device, dtype=dtype)
        proj_mask = proj_mask.bool()

        audio_lens = proj_mask.sum(dim=1)          # [B]
        S_max = proj_embs.size(1)
        D = proj_embs.size(-1)

        # ----------------------------
        # 2) Prompt → Embedding
        # ----------------------------

        prompt_ids = prompt_ids.to(device)      # [B, T]
        prompt_mask = prompt_ids != self.tokenizer.pad_token_id
        prompt_lens = prompt_mask.sum(dim=1)        # [B]
        T_prompt = prompt_ids.size(1)

        prompt_embs = self.llm_model.get_input_embeddings()(prompt_ids)  # [B, T, D]

        # ----------------------------
        # 3) Locate <extra_id_0> token
        # ----------------------------
        audio_token_mask = (prompt_ids == self.audio_token_id)
        assert (audio_token_mask.sum(dim=1) == 1).all(), f"Each prompt must contain exactly one <extra_id_0> token"

        audio_pos = audio_token_mask.float().argmax(dim=1)  # [B]

        # ----------------------------
        # 4) Allocate final sequence
        # ----------------------------
        total_lens = prompt_lens + audio_lens
        max_len = total_lens.max().item()

        inputs_embeds = torch.zeros((B, max_len, D), device=device, dtype=dtype)
        attention_mask = torch.zeros((B, max_len), device=device, dtype=torch.long)

        batch_idx = torch.arange(B, device=device)

        # ----------------------------
        # 5) Insert audio embeddings
        # ----------------------------
        audio_range = torch.arange(S_max, device=device).unsqueeze(0)     # [1, S]
        audio_valid = audio_range < audio_lens.unsqueeze(1)               # [B, S]
        dest_audio_pos = audio_pos.unsqueeze(1) + audio_range             # [B, S]

        b_audio = batch_idx.unsqueeze(1).expand_as(dest_audio_pos)[audio_valid]
        p_audio = dest_audio_pos[audio_valid]

        inputs_embeds[b_audio, p_audio] = proj_embs[audio_valid]
        attention_mask[b_audio, p_audio] = 1

        # ----------------------------
        # 6) Insert prompt embeddings
        # ----------------------------
        prompt_range = torch.arange(T_prompt, device=device).unsqueeze(0).expand(B, -1)

        # a) before <extra_id_0>
        before = prompt_range < audio_pos.unsqueeze(1)
        b_before = batch_idx.unsqueeze(1).expand_as(prompt_range)[before]
        p_before = prompt_range[before]

        inputs_embeds[b_before, p_before] = prompt_embs[before]
        attention_mask[b_before, p_before] = 1

        # b) after <extra_id_0>
        after = prompt_range > audio_pos.unsqueeze(1)
        b_after = batch_idx.unsqueeze(1).expand_as(prompt_range)[after]

        p_after = (
            prompt_range[after]
            + audio_lens.unsqueeze(1).expand_as(prompt_range)[after]
        )

        inputs_embeds[b_after, p_after] = prompt_embs[after]
        attention_mask[b_after, p_after] = 1

        # ----------------------------
        # 7) Position IDs
        # ----------------------------
        position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)

        # ----------------------------
        # 8) Generate
        # ----------------------------
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        # ----------------------------
        # 9) Decode
        # ----------------------------
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.info(f"Generated text: {texts[0] if len(texts) > 0 else ''}")
        return texts
    
