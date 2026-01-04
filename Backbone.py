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
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")

        ###### LLM (frozen) + LoRa (trainable) ############################
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True)
        logger.info(f"Loaded LLM model from {llm_path}")

        # # Freeze base model parameters
        # for param in self.llm_model.parameters():
        #     param.requires_grad = False

        # LoRA adapters
        if config_lora is not None:
            lora_path = config_lora.get("path", None)
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
                logger.info(f"Initialized LoRA adapters with config: {lora_cfg}")
        else:
            logger.warning("No LoRA config provided - all LLM parameters are frozen!")

        # Verify tokenizer and embedding alignment
        vocab_size = len(self.tokenizer)
        embed_size = self.llm_model.get_input_embeddings().weight.shape[0]
        assert embed_size == vocab_size, f"Embedding size mismatch: {embed_size} != {vocab_size}"
        logger.info(f"Tokenizer vocabulary size: {vocab_size:,}")

        #self.summary()

    def lora_parameters(self):
        """Returns only LoRA parameters (useful for separate optimizer)"""
        return [p for n, p in self.llm_model.named_parameters() if "lora" in n.lower() and p.requires_grad]

    def summary(self):
        """Log parameter counts and trainable parameter names"""
        total_params = 0
        trainable_params = 0
        trainable_names = []

        for name, param in self.llm_model.named_parameters():  # Use llm_model, not self
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
                trainable_names.append(name)

        frozen_params = total_params - trainable_params
        
        logger.info(f"Backbone LLM - Total: {total_params:,} | Trainable: {trainable_params:,} | Frozen: {frozen_params:,}")        
        if len(trainable_names):
            logger.info(f"Trainable parameters ({len(trainable_names)}): {trainable_names}")
        else:
            logger.info("No trainable parameters in LLM!")
