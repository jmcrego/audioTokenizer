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

        self.patch_vocab(config['token_map'])

        ###### LLM (frozen) + LoRa (trainable) ############################
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True)
        logger.info(f"Loaded LLM model from {llm_path}")

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

    def patch_vocab(self, token_map):
        """
        token_map = {
            "<asr>": 5,
            "</asr>": 6,
            "<stt>": 7,
            "</stt>": 8,
            "<[audio]>": 9
        }
        """
        vocab = self.tokenizer["model"]["vocab"]
        token_map_ids = set(token_map.values())

        # (1) Remove old tokens that currently use token_map IDs (5, 6, 7, 8, 9)
        old_tokens = [tok for tok, tid in vocab.items() if tid in token_map_ids]
        for tok in old_tokens:
            del vocab[tok]

        # (2) Insert new tokens with fixed IDs
        for tok, tid in token_map.items():
            vocab[tok] = tid

        # (3) Patch added_tokens (important for fast tokenizer)
        if "added_tokens" in self.tokenizer:
            for entry in self.tokenizer["added_tokens"]:
                tid = entry.get("id")
                if tid in token_map_ids:
                    # find corresponding token string
                    for tok, tok_id in token_map.items():
                        if tok_id == tid:
                            entry["content"] = tok
                            entry["special"] = True
                            break

        # assert no duplicate tokens
        ids = list(vocab.values())
        assert len(ids) == len(set(ids)), "Duplicate token IDs detected after patch_vocab"

        # assert token_map tokens correctly mapped
        for tok, tid in token_map.items():
            assert vocab.get(tok) == tid, f"Token {tok} not mapped to ID {tid}"

        logger.info(f"Vocabulary patched token_map = {token_map}")