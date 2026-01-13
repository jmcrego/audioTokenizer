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
    def __init__(self, config, config_lora, config_embeddings):
        super().__init__()

        llm_path = config["path"]

        #self.llm_model.tie_weights()

        ###### Tokenizer ##################################################
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        self.original_vocab_size = len(self.tokenizer)
        logger.info(f"Loaded Tokenizer from {llm_path} with size={self.original_vocab_size}")
        logger.info(f"eos_token is: {self.tokenizer.eos_token}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")

        ###### LLM  ############################
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True)
        self.tied_embeddings = self.llm_model.get_input_embeddings().weight.data_ptr() == self.llm_model.get_output_embeddings().weight.data_ptr()

        original_input_embeddings_size = self.llm_model.get_input_embeddings().weight.shape[0]
        original_output_embeddings_size = self.llm_model.get_output_embeddings().weight.shape[0]
        logger.info(f"Loaded LLM model from {llm_path} with embeddings size={original_input_embeddings_size}/{original_output_embeddings_size}")
        assert self.original_vocab_size == original_input_embeddings_size

        # special embeddings
        if config_embeddings is not None:
            self.special_tokens = config_embeddings.get("special_tokens", None)

            if self.special_tokens is not None:
                self.tokenizer.add_tokens(self.special_tokens)
                self.llm_model.resize_token_embeddings(len(self.tokenizer))
                self.new_vocab_size = len(self.tokenizer)

                embeddings_path = config_embeddings.get("path", None)
                if embeddings_path is not None:
                    ckpt = torch.load(embeddings_path, map_location="cpu")
                    new_tokens = ckpt["special_tokens"]
                    new_input_emb = ckpt["input_embeddings"]
                    new_output_emb = ckpt["output_embeddings"]
                    assert new_tokens == self.special_tokens
                    assert len(new_tokens) == self.new_vocab_size - self.original_vocab_size
                    ### insert tokens
                    with torch.no_grad():
                        self.llm_model.get_input_embeddings().weight[self.original_vocab_size : self.new_vocab_size].copy_(new_input_emb)
                        self.llm_model.get_output_embeddings().weight[self.original_vocab_size : self.new_vocab_size].copy_(new_output_emb)
                    logger.info(f"Loaded special_tokens embeddings with config: {config_embeddings}")
                else:
                    logger.info(f"Initialized special_tokens embeddings with config: {config_embeddings}")
        else:
            self.new_vocab_size = None
            logger.warning("No embeddings config provided - all LLM embedding parameters are frozen!")


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


    def freeze(self):
        self.llm_model.eval()
        for p in self.llm_model.parameters():
            p.requires_grad = False
        logger.info("llm_model frozen (eval mode)")


    def unfreeze(self):
        self.llm_model.train()
        
        def freeze_old_embeddings(grad):
            grad[:self.original_vocab_size] = 0
            return grad
        
        for n, p in self.llm_model.named_parameters():
            # LoRA trainable
            if "lora" in n.lower():
                p.requires_grad = True
            # Embeddings: enable grads globally (old rows frozen below)
            elif n in ["model.embed_tokens.weight", "lm_head.weight"]:
                p.requires_grad = True
                # Register hook immediately after enabling gradients
                if n == "model.embed_tokens.weight" and not hasattr(self, "_embedding_hook_registered"):
                    p.register_hook(freeze_old_embeddings)
                    self._embedding_hook_registered = True
            # Everything else → frozen
            else:
                p.requires_grad = False

        logger.info("llm_model (LoRA, special_tokens i/o embeddings) unfrozen (train mode)")


    def lora_parameters(self):
        """Returns only LoRA parameters (useful for separate optimizer)"""
        return [p for n, p in self.llm_model.named_parameters() if "lora" in n.lower() and p.requires_grad]


    def save(self, ckpt_path):
        self.llm_model.save_pretrained(ckpt_path + ".lora")
        logger.info(f"Saved LoRA adapters to {ckpt_path}.lora")

        input_embs = self.llm_model.get_input_embeddings().weight[self.llm_model.original_vocab_size : ].detach().cpu().clone()
        output_embs = self.llm_model.get_output_embeddings().weight[self.llm_model.original_vocab_size : ].detach().cpu().clone()
        torch.save({"special_tokens": self.special_tokens, "input_embeddings": input_embs, "output_embeddings": output_embs}, ckpt_path + ".embs.pt")
        logger.info(f"Saved special_tokens embeddings to {ckpt_path}.embs.pt")
