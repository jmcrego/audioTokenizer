# BackboneLLM.py

import torch
import logging
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel, LoraConfig

logger = logging.getLogger("Backbone")

class Backbone(torch.nn.Module):
    """
    Wrapper for the base LLM with LoRA adapters
    """
    def __init__(self, config, config_lora):
        super().__init__()

        llm_path = config["path"]

        ###### Tokenizer ##################################################
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        logger.info(f"Loaded Tokenizer from {llm_path}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        ###### LLM (frozen) + LoRa (trainable) ############################
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True)
        logger.info(f"Loaded LLM model from {llm_path}")

        ###### new tokens/embeddings (trainable) ##########################
        self.add_new_tokens(config['add_tokens'])

        # LoRA adapters
        if config_lora is not None:
            lora_path = config_lora["path"]
            if lora_path is not None:
                # load preexisting lora adapters
                self.llm_model = PeftModel.from_pretrained(self.llm_model, lora_path)
                logger.info(f"Loaded LoRa adapters from {lora_path}")
            else:
                # create new lora adapters
                lora_cfg = LoraConfig(
                    r=config_lora["r"],
                    lora_alpha=config_lora["lora_alpha"],
                    target_modules=config_lora["target_modules"],
                    lora_dropout=config_lora["lora_dropout"],
                    bias=config_lora["bias"],
                    task_type=config_lora["task_type"],
                )
                self.llm_model = get_peft_model(self.llm_model, lora_cfg)
                logger.info(f"Initialized LoRa adapters {lora_cfg}")

        assert self.llm_model.get_input_embeddings().weight.shape[0] == len(self.tokenizer)


    def add_new_tokens(self, new_tokens): 
        """
        Add new special tokens to the tokenizer
        Note: HF will assign new IDs at the end of the vocab.
        """
        self.asr_start_token=new_tokens["asr_start_token"]
        self.asr_end_token=new_tokens["asr_end_token"]
        self.stt_start_token=new_tokens["stt_start_token"]
        self.stt_end_token=new_tokens["stt_end_token"]
        self.audio_token=new_tokens["audio_token"]
        path=new_tokens["path"]

        # Extract new tokens
        new_tokens = [self.asr_start_token, self.asr_end_token, self.stt_start_token, self.stt_end_token, self.audio_token]
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if added > 0:
            # Resize model embeddings to accommodate new tokens
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

        # Store new token IDs for later convenience
        self.special_token_ids = {tok: self.tokenizer.convert_tokens_to_ids(tok) for tok in new_tokens}
        self.asr_start_token_id = self.special_token_ids.get("<asr>", None)
        self.asr_end_token_id = self.special_token_ids.get("</asr>", None)
        self.stt_start_token_id = self.special_token_ids.get("<stt>", None)
        self.stt_end_token_id = self.special_token_ids.get("</stt>", None)
        self.audio_token_id = self.special_token_ids.get("<audio>", None)

        logger.info(f"Tokenizer patched with special tokens: {self.special_token_ids}")

        if path is None:
            logger.info(f"Special token embeddings initialized from scratch")
            return
        
        # Load embeddings for previously added special tokens.
        payload = torch.load(path, map_location="cpu")
        saved_ids = payload["token_ids"]
        saved_embeddings = payload["embeddings"]
        emb_layer = self.llm_model.get_input_embeddings()

        # Safety checks
        assert len(saved_ids) == saved_embeddings.size(0)
        assert saved_embeddings.size(1) == emb_layer.weight.size(1)

        for i, tid in enumerate(saved_ids):
            emb_layer.weight.data[tid] = saved_embeddings[i].to(emb_layer.weight.device)

        logger.info(f"Loaded {len(saved_ids)} special token embeddings from {path}")

