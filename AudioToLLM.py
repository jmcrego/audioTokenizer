# AudioToLLM.py

import torch
import logging
from transformers import StoppingCriteriaList
from transformers import StoppingCriteria

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
        embedder_trainable_names = [n for n, p in self.audio_embedder.named_parameters() if p.requires_grad]
        
        # Projector
        projector_total = sum(p.numel() for p in self.projector.parameters())
        projector_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        projector_trainable_names = [n for n, p in self.projector.named_parameters() if p.requires_grad]
        
        # LLM
        llm_total = sum(p.numel() for p in self.llm_model.parameters())
        llm_trainable = sum(p.numel() for p in self.llm_model.parameters() if p.requires_grad)
        llm_trainable_names = [n for n, p in self.llm_model.named_parameters() if p.requires_grad]
        
        # Total
        total = embedder_total + projector_total + llm_total
        trainable = embedder_trainable + projector_trainable + llm_trainable
        frozen = total - trainable
        
        logger.info("=" * 100)
        logger.info("AudioToLLM MODEL PARAMETER SUMMARY")
        logger.info("=" * 100)
        logger.info(f"Audio Embedder : {embedder_total:>15,} total | {embedder_trainable:>15,} trainable | {embedder_total - embedder_trainable:>15,} frozen")
        logger.info(f"Projector      : {projector_total:>15,} total | {projector_trainable:>15,} trainable | {projector_total - projector_trainable:>15,} frozen")
        logger.info(f"LLM (+ LoRA)   : {llm_total:>15,} total | {llm_trainable:>15,} trainable | {llm_total - llm_trainable:>15,} frozen")
        logger.info("-" * 100)
        logger.info(f"TOTAL          : {total:>15,} total | {trainable:>15,} trainable | {frozen:>15,} frozen")
        logger.info(f"Trainable %    : {100 * trainable / total:.2f}%")
        logger.info("=" * 100)
        
        # Show trainable parameter names for each component
        logger.info("TRAINABLE PARAMETERS:")
        logger.info("-" * 100)
        
        # Audio Embedder
        if embedder_trainable_names:
            logger.info(f"Audio Embedder ({len(embedder_trainable_names)} params):")
            for name in embedder_trainable_names[:20]:  # Show first 20
                logger.info(f"  - {name}")
            if len(embedder_trainable_names) > 20:
                logger.info(f"  ... and {len(embedder_trainable_names) - 20} more")
        else:
            logger.info("Audio Embedder: (none - all frozen)")
        
        # Projector
        if projector_trainable_names:
            logger.info(f"Projector ({len(projector_trainable_names)} params):")
            for name in projector_trainable_names:
                logger.info(f"  - {name}")
        else:
            logger.info("Projector: (none - all frozen)")
        
        # LLM
        if llm_trainable_names:
            logger.info(f"LLM ({len(llm_trainable_names)} params):")
            for name in llm_trainable_names[:20]:  # Show first 20
                logger.info(f"  - {name}")
            if len(llm_trainable_names) > 20:
                logger.info(f"  ... and {len(llm_trainable_names) - 20} more")
        else:
            logger.info("LLM: (none - all frozen)")
        
        logger.info("=" * 100)


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
        logger.info(f"Batch with {B} samples")

        # ----------------------------
        # 1) Audio → Embedding → Projector
        # ----------------------------
        audio_embs, audio_mask = self.audio_embedder(audio_files)
        audio_embs = audio_embs.to(device=device, dtype=dtype)
        audio_mask = audio_mask.bool().to(device)
        logger.info(f"audio_embs.shape = {audio_embs.shape}")
        logger.info(f"audio_mask.shape = {audio_mask.shape}")

        proj_embs, proj_mask = self.projector(audio_embs, audio_mask)
        proj_embs = proj_embs.to(device=device, dtype=dtype)
        proj_mask = proj_mask.bool()
        logger.info(f"proj_embs.shape = {proj_embs.shape}")
        logger.info(f"proj_mask.shape = {proj_mask.shape}")

        audio_lens = proj_mask.sum(dim=1)          # [B]
        S_max = proj_embs.size(1)
        D = proj_embs.size(-1)
        logger.info(f"audio_lens = {audio_lens}")
        logger.info(f"S_max = {S_max}")
        logger.info(f"D = {D}")

        # ----------------------------
        # 2) Prompt → Embedding
        # ----------------------------
        prompt_ids = prompt_ids.to(device)      # [B, T]
        prompt_mask = prompt_ids != self.tokenizer.pad_token_id
        prompt_lens = prompt_mask.sum(dim=1)        # [B]
        T_prompt = prompt_ids.size(1)
        logger.info(f"prompt_ids.shape = {prompt_ids.shape}")
        logger.info(f"prompt_mask.shape = {prompt_mask.shape}")
        logger.info(f"prompt_lens = {prompt_lens}")
        logger.info(f"T_prompt = {T_prompt}")

        prompt_embs = self.llm_model.get_input_embeddings()(prompt_ids)  # [B, T, D]
        logger.info(f"prompt_embs.shape = {prompt_embs.shape}")

        # ----------------------------
        # 3) Locate <extra_id_0> token
        # ----------------------------
        audio_token_mask = (prompt_ids == self.audio_token_id)
        assert (audio_token_mask.sum(dim=1) == 1).all(), f"Each prompt must contain exactly one <extra_id_0> token"

        audio_pos = audio_token_mask.float().argmax(dim=1)  # [B]
        logger.info(f"audio_token_mask.shape = {audio_token_mask.shape}")
        logger.info(f"audio_pos = {audio_pos}")

        # ----------------------------
        # 4) Allocate final sequence
        # ----------------------------
        total_lens = prompt_lens - 1 + audio_lens    # each prompt loses <extra_id_0>
        max_len = total_lens.max().item()
        logger.info(f"total_lens = {total_lens}")
        logger.info(f"max_len = {max_len}")

        inputs_embeds = torch.zeros((B, max_len, D), device=device, dtype=dtype)
        attention_mask = torch.zeros((B, max_len), device=device, dtype=torch.long)

        logger.info(f"inputs_embeds.shape = {inputs_embeds.shape}")
        logger.info(f"attention_mask.shape = {attention_mask.shape}")

        attn_sum = attention_mask.sum(dim=1)
        if not torch.all(attn_sum == total_lens):
            raise RuntimeError(
                f"Attention mismatch:\n"
                f"attn_sum={attn_sum}\n"
                f"total_lens={total_lens}"
            )

        # ----------------------------
        # 5) Compute insert indices
        # ----------------------------
        # lengths before/after audio
        before_len = audio_pos                                  # [B]
        after_len  = prompt_lens - audio_pos - 1                # [B]
        logger.info(f"before_len = {before_len}")
        logger.info(f"after_len = {after_len}")
        # batch indices
        batch_idx = torch.arange(B, device=device)

        # ----------------------------
        # 6) Insert prompt embeddings BEFORE audio
        # ----------------------------
        # [B, max_before] boolean mask
        prompt_before_mask = torch.arange(T_prompt, device=device).unsqueeze(0) < before_len.unsqueeze(1)
        logger.info(f"prompt_before_mask.shape = {prompt_before_mask.shape}")
        b_idx, t_idx = torch.nonzero(prompt_before_mask, as_tuple=True)
        inputs_embeds[b_idx, t_idx] = prompt_embs[b_idx, t_idx]
        attention_mask[b_idx, t_idx] = 1

        # ----------------------------
        # 7) Insert audio embeddings
        # ----------------------------
        # audio positions in final sequence
        audio_range = torch.arange(S_max, device=device).unsqueeze(0)       # [1, S_max]
        valid_audio = audio_range < audio_lens.unsqueeze(1)                 # [B, S_max]
        dest_audio_pos = before_len.unsqueeze(1) + audio_range              # [B, S_max]
        b_audio, pos_audio = torch.nonzero(valid_audio, as_tuple=True)
        logger.info(f"b_audio.shape = {b_audio.shape}")
        logger.info(f"pos_audio.shape = {pos_audio.shape}")

        inputs_embeds[b_audio, dest_audio_pos[b_audio, pos_audio]] = proj_embs[b_audio, pos_audio]
        attention_mask[b_audio, dest_audio_pos[b_audio, pos_audio]] = 1

        # ----------------------------
        # 8) Insert prompt embeddings AFTER audio
        # ----------------------------
        # compute source indices in prompt
        prompt_after_mask = torch.arange(T_prompt, device=device).unsqueeze(0) > audio_pos.unsqueeze(1)
        b_after, t_after = torch.nonzero(prompt_after_mask, as_tuple=True)
        # target positions in final sequence
        after_shift = audio_lens[b_after] + before_len[b_after]
        target_pos = t_after - (audio_pos[b_after] + 1) + after_shift
        inputs_embeds[b_after, target_pos] = prompt_embs[b_after, t_after]
        attention_mask[b_after, target_pos] = 1

        # ----------------------------
        # 9) Position IDs
        # ----------------------------
        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        logger.info(f"position_ids.shape = {position_ids.shape}")

        # ----------------------------
        # 10) Generate
        # ----------------------------
        stopping_criteria = StoppingCriteriaList([StopOnEOSFirst(self.tokenizer.eos_token_id)])

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=self.tokenizer.eos_token_id,
            # pad_token_id=self.tokenizer.pad_token_id,
            # eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        logger.info(f"outputs.shape = {outputs.shape}")


        # ----------------------------
        # 11) Decode
        # ----------------------------
        prompt_len = max_len
        generated = outputs[:, prompt_len:]
        
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        logger.info(f"Generated text: {texts[0] if len(texts) > 0 else ''}")
        return texts
    
class StopOnEOSFirst(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # first sequence in batch emits eos
        return input_ids[0, -1] == self.eos_token_id 
    
class StopOnEOSAll(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id
        self.finished = None

    def __call__(self, input_ids, scores, **kwargs):
        # all sequences in batch have emitted eos
        if self.finished is None:
            self.finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        self.finished |= (input_ids[:, -1] == self.eos_token_id)
        return self.finished.all() 

