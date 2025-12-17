
# AudioToLLM.py

import torch
import json
import logging
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel, LoraConfig

from Embedder import Embedder
from Projector import Projector

logger = logging.getLogger("AudioToLLM")

class AudioToLLM(torch.nn.Module):
    """
    Wrapper combining Embedder -> Projector -> LLM
    """
    def __init__(self, config, device, dtype, is_infer=False):
        super().__init__()

        self.config = config

        ###### Embedder (frozen) ####################################
        self.audio_embedder = Embedder(config['audio'])
        self.audio_embedding_dim = self.audio_embedder.embedding_dim

        ###### Tokenizer ##################################################
        llm_path = config['llm']['path']
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        logger.info(f"Loaded Tokenizer from {llm_path}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        ###### LLM (frozen) + LoRa (trainable) ############################
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True)
        self.llm_embedding_dim = self.llm_model.config.hidden_size
        logger.info(f"Loaded LLM model from {llm_path}")

        # Load LoRA adapters
        if config['lora']['path'] is not None:
            self.llm_model = PeftModel.from_pretrained(self.llm_model, config['lora']['path'], is_trainable=not is_infer)
            logger.info(f"Loaded LoRa adapters from {config['lora']['path']}")
        else:
            # Initialize LoRa in LLM
            lora_cfg = LoraConfig(
                r=config["lora"]["config"]["lora_r"],
                lora_alpha=config["lora"]["config"]["lora_alpha"],
                target_modules=config["lora"]["config"]["target_modules"],
                lora_dropout=config["lora"]["config"]["lora_dropout"],
                bias=config["lora"]["config"]["bias"],
                task_type=config["lora"]["config"]["task_type"],
            )
            self.llm_model = get_peft_model(self.llm_model, lora_cfg)
            logger.info(f"Initialized LoRa adapters")


        ###### Projector (trainable) ######################################
        self.projector = Projector(config['projector'], audio_embedding_dim=self.audio_embedding_dim, llm_embedding_dim=self.llm_embedding_dim)

        ### set to correct device/dtype
        self.audio_embedder.to(device=device, dtype=dtype)
        self.projector.to(device=device, dtype=dtype)
        self.llm_model.to(device, dtype=dtype)

        ### freeze/unfreeze parameters and set eval mode if needed
        if is_infer:
            # Freeze base llm_model, Freeze LoRA
            self.llm_model.eval()
            for n, p in self.llm_model.named_parameters():
                p.requires_grad = False
            # Freeze projector
            self.projector.eval()
            for p in self.projector.parameters():
                p.requires_grad = False
        else: # is training
            # Freeze base llm_model, keep LoRA trainable
            for n, p in self.llm_model.named_parameters():
                p.requires_grad = ("lora" in n.lower())
            # Keep projector trainable
            for p in self.projector.parameters():
                p.requires_grad = True

        # Freeze audio_embedder
        self.audio_embedder.eval()
        for p in self.audio_embedder.parameters():
            p.requires_grad = False


    def save(self, path):
        # Save projector to path.proj.pt
        torch.save(self.projector.state_dict(), path + ".proj.pt")
        logger.info(f"Saved Projector to {path}.proj.pt")
        # Save LoRa adapters (PEFT) to path.lora/{adapter_model.bin,adapter_config.json}
        self.llm_model.save_pretrained(path + ".lora")
        logger.info(f"Saved LoRa adapters to {path}.lora")
        # Save config to path.config.json}
        self.config['lora']['path'] = path + ".lora"
        with open(f"{path}.config.json", "w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=4)
        logger.info(f"Saved config to {path}.config.json")



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



    @torch.no_grad()
    def generate(
        self, 
        audio_files: str, 
        prompt: str,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
    ):
        """
        audio_files: list[str]
        prompt: text instruction ("transcribe and translate", etc.)
        """
        device = self.llm_model.device
        dtype  = next(self.projector.parameters()).dtype

        # --------------------------------------------------
        # 1) Audio → embeddings → LLM projected embeddings 
        # --------------------------------------------------
        audio_embs, audio_mask = self.audio_embedder(audio_files)
        audio_embs = audio_embs.to(device, dtype)
        audio_mask = audio_mask.to(device)

        proj_embs, proj_mask = self.projector(audio_embs, audio_mask)
        proj_embs = proj_embs.to(device, dtype)
        proj_mask = proj_mask.to(device)
        logger.info(f"proj_embs size = {proj_embs.shape}")
        logger.info(f"proj_mask size = {proj_mask.shape}")

        B, S, D = proj_embs.shape

        # --------------------------------------------------
        # 2) Prompt embeddings
        # --------------------------------------------------
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
#            add_special_tokens=False,
        ).input_ids.to(device)

        logger.info(f"prompt: {prompt}")

        prompt_embs = self.llm_model.get_input_embeddings()(prompt_ids)
        prompt_embs = prompt_embs.expand(B, -1, -1)
        logger.info(f"prompt_embs size = {prompt_embs.shape}")

        # --------------------------------------------------
        # 3) Concatenate embeddings
        # --------------------------------------------------
        inputs_embeds = torch.cat([proj_embs, prompt_embs], dim=1)

        attention_mask = torch.cat(
            [
                proj_mask,
                torch.ones(
                    (B, prompt_embs.size(1)),
                    device=device,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )

        logger.info(f"inputs_embeds size = {inputs_embeds.shape}")
        logger.info(f"attention_mask size = {attention_mask.shape}")

        # --------------------------------------------------
        # 4) Generate
        # --------------------------------------------------
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=None,#self.tokenizer.eos_token_id,
        )
        logger.info(f"outputs size = {outputs.shape}")

        # --------------------------------------------------
        # 6) Decode ONLY generated tokens
        # --------------------------------------------------
        gen_tokens = outputs#[:, inputs_embeds.size(1):]
        texts = self.tokenizer.batch_decode(
            gen_tokens,
            skip_special_tokens=False,
        )
        logger.info(f"texts = {texts[0]}")

        return texts