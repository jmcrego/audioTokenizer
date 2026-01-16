# AudioToLLM.py

import os
import torch
import logging
from collections import OrderedDict
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

        self.buffer_size = 2  # max number of cache buckets kept in memory

        # ====================================================
        # 1. Modules
        # ====================================================
        self.audio_embedder = Embedder(config['audio'])
        self.backbone = Backbone(config['llm'], config['lora'], config['embeddings'])

        self.audio_token = config["llm"]["audio_token"]
        self.audio_token_id = self.backbone.tokenizer.convert_tokens_to_ids(self.audio_token)
        assert self.audio_token_id is not None, "audio_token_id is None"
        assert isinstance(self.audio_token_id, int), type(self.audio_token_id)
        logger.info(f"Audio token: '{self.audio_token}' -> ID: {self.audio_token_id}")

        self.projector = Projector(
            config['projector'], 
            audio_embedding_dim=self.audio_embedder.embedding_dim, 
            llm_embedding_dim=self.backbone.llm_model.config.hidden_size
        )

        # ====================================================
        # 2. Device & dtype
        # ====================================================
        logger.info(f"Moving models to device={device}, dtype={dtype}")
        self.audio_embedder.to(device=device, dtype=dtype) #use float32 if numerical issues
        self.projector.to(device=device, dtype=dtype)      #use float32 to ensure stability during early training of projector
        self.backbone.llm_model.to(device=device, dtype=dtype)      #float16/bfloat16 is for memory efficiency

        self.audio_embedder.freeze()

        if is_infer:
            self.projector.freeze()
            self.backbone.freeze()
        else:
            self.projector.unfreeze()
            self.backbone.unfreeze()      

        logger.info(f"Audio embedder: {next(self.audio_embedder.parameters()).dtype} on {next(self.audio_embedder.parameters()).device}")
        logger.info(f"Projector: {next(self.projector.parameters()).dtype} on {next(self.projector.parameters()).device}")
        logger.info(f"LLM: {next(self.backbone.llm_model.parameters()).dtype} on {next(self.backbone.llm_model.parameters()).device}")

        self.summary()

    def save(self, ckpt_path):
        self.projector.save(ckpt_path)
        self.backbone.save(ckpt_path)


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
        llm_total = sum(p.numel() for p in self.backbone.llm_model.parameters())
        llm_trainable = sum(p.numel() for p in self.backbone.llm_model.parameters() if p.requires_grad)
        llm_trainable_names = [n for n, p in self.backbone.llm_model.named_parameters() if p.requires_grad]
        
        # Total
        total = embedder_total + projector_total + llm_total
        trainable = embedder_trainable + projector_trainable + llm_trainable
        frozen = total - trainable
        
        logger.info("=" * 100)
        logger.info("AudioToLLM MODEL PARAMETER SUMMARY")
        logger.info("=" * 100)
        logger.info(f"Audio Embedder : {embedder_total:>15,} total | {embedder_trainable:>15,} trainable | {embedder_total - embedder_trainable:>15,} frozen")
        logger.info(f"Projector      : {projector_total:>15,} total | {projector_trainable:>15,} trainable | {projector_total - projector_trainable:>15,} frozen")
        logger.info(f"LLM (LoRA/Emb) : {llm_total:>15,} total | {llm_trainable:>15,} trainable | {llm_total - llm_trainable:>15,} frozen")
        logger.info("-" * 100)
        logger.info(f"TOTAL          : {total:>15,} total | {trainable:>15,} trainable | {frozen:>15,} frozen")
        logger.info(f"Trainable %    : {100 * trainable / total:.2f}%")
        logger.info("=" * 100)
        
        # Show trainable parameter names for each component
        logger.info("-" * 100)
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
        
        logger.info("-" * 100)


    def format_batch(self, audio_paths, prompt_ids, target_ids=None, pt_paths=None, offsets=None):
        """
        Formats a batch by combining prompt, audio, and (optionally) target embeddings.

        Args:
            audio_paths: list of audio file paths
            prompt_ids: [B, T_prompt] input token IDs
            target_ids: [B, T_target] target token IDs (optional)
            pt_paths / offsets: for cached audio embeddings

        Returns:
            dict with:
                inputs_embeds: [B, L_max, D] final embeddings
                attention_mask: [B, L_max] attention mask
                labels: [B, L_max] (if target_ids provided)
                audio_norm, text_norm, target_norm: for logging
        """
        device = self.backbone.llm_model.device
        llm_dtype = next(self.backbone.llm_model.parameters()).dtype
        B = prompt_ids.size(0)

        # 1) Embed audio
        if pt_paths is None and offsets is None:
            with torch.no_grad():
                audio_embs, _ = self.audio_embedder(audio_paths)  # [B, T_audio, D_audio]
        else:
            audio_embs = self.read_cache_embs(pt_paths, offsets)

        audio_embs = audio_embs.to(device)

        # 2) Project audio embeddings
        proj_embs, proj_mask = self.projector(audio_embs)      # [B, S_max, D_llm], [B, S_max]
        proj_mask = proj_mask.bool()
        B, S_max, D = proj_embs.shape
        audio_lens = proj_mask.sum(dim=1)                      # [B]

        # 3) Embed prompt
        prompt_ids = prompt_ids.to(device)
        prompt_embs = self.backbone.llm_model.get_input_embeddings()(prompt_ids)  # [B, T_prompt, D]
        prompt_mask = prompt_ids != self.backbone.tokenizer.pad_token_id
        prompt_lens = prompt_mask.sum(dim=1)                   # [B]
        T_prompt = prompt_ids.size(1)

        # 3bis) Embed target if provided
        if target_ids is not None:
            target_ids = target_ids.to(device)
            target_embs = self.backbone.llm_model.get_input_embeddings()(target_ids)  # [B, T_target, D]
            target_mask = target_ids != self.backbone.tokenizer.pad_token_id
            target_lens = target_mask.sum(dim=1)          # [B]
            T_target = target_ids.size(1)

        # 4) Locate <extra_id_0> in prompt
        audio_token_mask = prompt_ids == self.audio_token_id
        assert (audio_token_mask.sum(dim=1) == 1).all(), "Each sample must have exactly one <extra_id_0>"
        audio_pos = audio_token_mask.float().argmax(dim=1)  # [B]

        # 5) Allocate final batch
        if target_ids is not None:
            total_lens = (prompt_lens - 1) + audio_lens + target_lens  # [B]
        else:
            total_lens = (prompt_lens - 1) + audio_lens

        max_len = total_lens.max().item()
        inputs_embeds = torch.zeros((B, max_len, D), device=device, dtype=llm_dtype)
        attention_mask = torch.zeros((B, max_len), device=device, dtype=torch.long)
        if target_ids is not None:
            labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)

        # 6) Insert prompt tokens before <extra_id_0>
        range_T = torch.arange(T_prompt, device=device).unsqueeze(0)  # [1, T_prompt]
        before_mask = range_T < audio_pos.unsqueeze(1)                # [B, T_prompt]
        b_idx, t_idx = torch.nonzero(before_mask, as_tuple=True)
        inputs_embeds[b_idx, t_idx] = prompt_embs[b_idx, t_idx]
        attention_mask[b_idx, t_idx] = 1

        # 7) Insert audio embeddings
        range_S = torch.arange(S_max, device=device).unsqueeze(0)     # [1, S_max]
        valid_audio = range_S < audio_lens.unsqueeze(1)               # [B, S_max]
        audio_dest = audio_pos.unsqueeze(1) + range_S                 # [B, S_max]
        b_a, s_a = torch.nonzero(valid_audio, as_tuple=True)
        inputs_embeds[b_a, audio_dest[b_a, s_a]] = proj_embs[b_a, s_a]
        attention_mask[b_a, audio_dest[b_a, s_a]] = 1

        # 8) Insert prompt tokens after <extra_id_0>
        after_mask = range_T > audio_pos.unsqueeze(1)                 # [B, T_prompt]
        b_p, t_p = torch.nonzero(after_mask, as_tuple=True)
        after_offset = audio_lens[b_p] + audio_pos[b_p]
        dest_pos = after_offset + (t_p - (audio_pos[b_p] + 1))
        inputs_embeds[b_p, dest_pos] = prompt_embs[b_p, t_p]
        attention_mask[b_p, dest_pos] = 1

        # 9) Insert target if provided
        if target_ids is not None:
            range_L = torch.arange(T_target, device=device).unsqueeze(0)   # [1, T_target]
            valid_target = range_L < target_lens.unsqueeze(1)               # [B, T_target]
            target_offset = (prompt_lens - 1 + audio_lens).unsqueeze(1)    # [B,1]
            target_dest = target_offset + range_L                           # [B, T_target]
            b_t, l_t = torch.nonzero(valid_target, as_tuple=True)
            inputs_embeds[b_t, target_dest[b_t, l_t]] = target_embs[b_t, l_t]
            labels[b_t, target_dest[b_t, l_t]] = target_ids[b_t, l_t]
            attention_mask[b_t, target_dest[b_t, l_t]] = 1

        attn_sum = attention_mask.sum(dim=1)
        if not torch.all(attn_sum == total_lens):
            raise RuntimeError(f"Attention mismatch:\nattn_sum={attn_sum}\ntotal_lens={total_lens}")

        # 10) Norms for logging
        output = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

        if target_ids is None:
            return output
        
        flat_audio_embs = proj_embs[valid_audio]
        audio_norm = flat_audio_embs.norm(dim=-1).mean() if flat_audio_embs.numel() > 0 else torch.tensor(0.0, device=device)

        prompt_no_audio_mask = prompt_mask & ~audio_token_mask
        flat_prompt_embs = prompt_embs[prompt_no_audio_mask]
        text_norm = flat_prompt_embs.norm(dim=-1).mean() if flat_prompt_embs.numel() > 0 else torch.tensor(0.0, device=device)

        flat_target_embs = target_embs[valid_target]
        target_norm = flat_target_embs.norm(dim=-1).mean() if flat_target_embs.numel() > 0 else torch.tensor(0.0, device=device)

        output.update({
            "labels": labels,
            "audio_norm": audio_norm,
            "text_norm": text_norm,
            "target_norm": target_norm
        })

        return output


    # ========================================================
    # Forward (training)
    # ========================================================
    def forward(self, audio_paths, prompt_ids, target_ids, pt_paths, offsets):

        formatted_batch = self.format_batch(audio_paths, prompt_ids, target_ids=target_ids, pt_paths=pt_paths, offsets=offsets)

        inputs_embeds=formatted_batch["inputs_embeds"]
        attention_mask=formatted_batch["attention_mask"]
        labels=formatted_batch["labels"]
        audio_norm=formatted_batch["audio_norm"]
        text_norm=formatted_batch["text_norm"]
        target_norm=formatted_batch["target_norm"]

        # Inputs Embeds (B × L)

        # Batch 0: [ P, A, A, A, P, P, P, P, T, T, T, 0, 0, 0 ]
        # Batch 1: [ P, A, A, A, A, P, P, P, T, T, T, T, T, T ]

        # Labels (B × L)
        # Batch 0: [ -, -, -, -, -, -, -, -, T, T, T, -, -, - ]
        # Batch 1: [ -, -, -, -, -, -, -, -, T, T, T, T, T, T ]

        # P = prompt embedding
        # A = audio embedding
        # T = target embedding
        # 0 = padding embedding (zero vector)
        # - = label ignore (-100)

        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        outputs = self.backbone.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio_norm": audio_norm,
            "text_norm": text_norm,
            "target_norm": target_norm,
        }

    # ========================================================
    # Generate (inference)
    # ========================================================
    @torch.no_grad()
    def generate(
        self, 
        audio_paths, 
        prompt_ids, 
        max_new_tokens=256, 
        temperature=0.7, 
        top_p=0.95,
        no_repeat_ngram_size = 0, #dangerous for ASR/STT, speech allow repetitions
        repetition_penalty = 1.1, #good for ASR/STT, but bad for QA
    ):

        formatted_batch = self.format_batch(audio_paths, prompt_ids)

        inputs_embeds=formatted_batch["inputs_embeds"]
        attention_mask=formatted_batch["attention_mask"]

        # Inputs Embeds (B × L)

        # Batch 0: [ P, A, A, A, P, P, P, P ]
        # Batch 1: [ P, A, A, A, A, P, P, 0 ]

        # P = prompt embedding
        # A = audio embedding
        # 0 = padding embedding (zero vector)

        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        stopping_criteria = StoppingCriteriaList([StopOnEOSFirst(self.backbone.tokenizer.eos_token_id)])

        outputs = self.backbone.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            no_repeat_ngram_size = no_repeat_ngram_size, 
            repetition_penalty = repetition_penalty,
            pad_token_id = self.backbone.tokenizer.eos_token_id,
            eos_token_id = self.backbone.tokenizer.eos_token_id,
            use_cache=True,
        )

        return self.backbone.tokenizer.batch_decode(outputs, skip_special_tokens=False) #skip_special_tokens should be set to True when working



    def read_cache_embs(self, pt_paths, offsets):
        """
        Reads the batch embeddings cached in disk as indicated by pt_paths and offsets
        Args:
            pt_paths (List[str]): bucket filenames
            offsets (List[int] or Tensor): index inside each bucket

        Returns:
            audio_embs: Tensor [B, T, D] (on CPU)
        """

        if not hasattr(self, "_bucket_cache"):
            self._bucket_cache = OrderedDict()

        if isinstance(offsets, torch.Tensor):
            offsets = offsets.tolist()

        assert len(pt_paths) == len(offsets)

        # Group batch positions by pt_path
        path_to_items = {}
        for batch_idx, (pt_path, offset) in enumerate(zip(pt_paths, offsets)):
            path_to_items.setdefault(pt_path, []).append((batch_idx, offset))

        batch_embs = [None] * len(pt_paths)

        for pt_path, items in path_to_items.items():
            # ---- Load or reuse bucket ----
            if pt_path in self._bucket_cache:
                bucket = self._bucket_cache.pop(pt_path)  # mark as recently used
            else:
                bucket = torch.load(pt_path, map_location="cpu")
                # Enforce buffer size (LRU eviction)
                if len(self._bucket_cache) >= self.buffer_size:
                    self._bucket_cache.popitem(last=False)

            self._bucket_cache[pt_path] = bucket
            bucket_embs = bucket["audio_embs"]  # [B_bucket, T, D]

            # ---- Extract needed embeddings ----
            for batch_idx, offset in items:
                batch_embs[batch_idx] = bucket_embs[offset]

        # Stack in original batch order
        audio_embs = torch.stack(batch_embs, dim=0)

        return audio_embs


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

