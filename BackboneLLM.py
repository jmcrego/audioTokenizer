# BackboneLLM.py

import torch
import logging
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel, LoraConfig

logger = logging.getLogger("BackboneLLM")

class BackboneLLM(torch.nn.Module):
    """
    Wrapper for the base LLM
    """
    def __init__(self, config, config_lora):
        super().__init__()

        self.config = config

        llm_path = config['path']

        ###### Tokenizer ##################################################
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        logger.info(f"Loaded Tokenizer from {llm_path}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        ###### LLM (frozen) + LoRa (trainable) ############################
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True)
        logger.info(f"Loaded LLM model from {llm_path}")

        # LoRA adapters
        if config_lora["path"] is not None:
            # load preexisting lora adapters
            self.llm_model = PeftModel.from_pretrained(self.llm_model, config_lora["path"])
            logger.info(f"Loaded LoRa adapters from {lora_cfg}")
        else:
            lora_cfg = LoraConfig(
                r=config_lora["config"]["lora_r"],
                lora_alpha=config_lora["config"]["lora_alpha"],
                target_modules=config_lora["config"]["target_modules"],
                lora_dropout=config_lora["config"]["lora_dropout"],
                bias=config_lora["config"]["bias"],
                task_type=config_lora["config"]["task_type"],
            )
            # create new lora adapters
            self.llm_model = get_peft_model(self.llm_model, lora_cfg)
            logger.info(f"Initialized LoRa adapters")

