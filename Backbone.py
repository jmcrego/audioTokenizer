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

        self.patch_vocab(config['token_map'])

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
        # Add special tokens to tokenizer
        new_tokens = list(token_map.keys())
        self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

        # Resize LLM embeddings if necessary
        self.llm_model.resize_token_embeddings(len(self.tokenizer))

        logger.info(f"Tokenizer patched with tokens {new_tokens}")


