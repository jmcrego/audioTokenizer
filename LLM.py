# LLM.py

import torch
import logging
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel, LoraConfig

logger = logging.getLogger("LLM")

class LLM(torch.nn.Module):
    """
    Wrapper for the base LLM with LoRA adapters
    """
    def __init__(self, config, config_lora, config_embeddings):
        super().__init__()

        llm_path = config["path"]

        ###### Tokenizer ##################################################
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        self.original_vocab_size = len(self.tokenizer)
        logger.info(f"Loaded Tokenizer from {llm_path} with size={self.original_vocab_size}")
        logger.info(f"bos_token = {self.tokenizer.bos_token} {self.tokenizer.bos_token_id}")
        logger.info(f"eos_token = {self.tokenizer.eos_token} {self.tokenizer.eos_token_id}")
        logger.info(f"pad_token = {self.tokenizer.pad_token} {self.tokenizer.pad_token_id}")
        # Set PAD token (id is automatically inferred)
        self.tokenizer.pad_token = config['pad_token']
        logger.info(f"pad_token = {self.tokenizer.pad_token} {self.tokenizer.pad_token_id}")
        kk
        ### ADD SPECIAL TOKENS
        self.special_tokens = config_embeddings.get("special_tokens", [])
        additional_tokens = {"additional_special_tokens": self.special_tokens}
        num_added = self.tokenizer.add_special_tokens(additional_tokens)
        self.new_vocab_size = len(self.tokenizer)
        logger.info(f"Added {num_added} special tokens, new vocab size is {self.new_vocab_size}")

        ### VERIFY TOKENS
        tokens_to_verify = self.special_tokens + [config['pad_token'], config['audio_token']]
        for token in tokens_to_verify:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            assert isinstance(token_id, int), f"Token '{token}' does not map to a single token_id: {token_id}"
            assert token_id != self.tokenizer.unk_token_id, f"Token '{token}' is mapped to <unk>"
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            assert len(ids) == 1, f"Token '{token}' does not map to a single token_id when encoded: {ids}"
            assert ids[0] == token_id, f"Token '{token}' does not map to same token_id when encoded '{ids[0]}' than token_to_id '{token_id}'"
            logger.info(f"{token}: {token_id}")


        ###### LLM  ############################
        self.model = AutoModelForCausalLM.from_pretrained(llm_path, low_cpu_mem_usage=True)

        original_input_embeddings_size = self.model.get_input_embeddings().weight.shape[0]
        original_output_embeddings_size = self.model.get_output_embeddings().weight.shape[0]
        logger.info(f"Loaded LLM model from {llm_path} with embeddings size={original_input_embeddings_size}/{original_output_embeddings_size}")
        assert self.original_vocab_size == original_input_embeddings_size
        assert self.original_vocab_size == original_output_embeddings_size

        # special embeddings
        if config_embeddings is not None:
            if len(self.special_tokens):
                embeddings_path = config_embeddings.get("path", None)
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Extended LLM embeddings ({self.new_vocab_size}) accordingly")

                if embeddings_path is not None:
                    ckpt = torch.load(embeddings_path, map_location="cpu")
                    new_tokens = ckpt["special_tokens"]
                    new_input_emb = ckpt["input_embeddings"]
                    new_output_emb = ckpt["output_embeddings"]
                    assert new_tokens == self.special_tokens
                    assert len(new_tokens) == self.new_vocab_size - self.original_vocab_size
                    ### insert tokens
                    with torch.no_grad():
                        self.model.get_input_embeddings().weight[self.original_vocab_size : self.new_vocab_size].copy_(new_input_emb)
                        self.model.get_output_embeddings().weight[self.original_vocab_size : self.new_vocab_size].copy_(new_output_emb)
                    logger.info(f"Loaded special_tokens embeddings with config: {config_embeddings}")
                    assert self.tokenizer.convert_tokens_to_ids(self.special_tokens[0]) == self.original_vocab_size

                else:
                    with torch.no_grad():
                        self.model.get_input_embeddings().weight[self.original_vocab_size:].normal_(mean=0.0, std=0.02)
                    logger.info(f"Initialized special_tokens embeddings with config: {config_embeddings}")

        else:
            self.new_vocab_size = None
            logger.warning("No embeddings config provided - all LLM embedding parameters are frozen!")


        # LoRA adapters
        if config_lora is not None:
            lora_path = config_lora.get("path", None)
            if lora_path is not None:
                # load preexisting lora adapters
                self.model = PeftModel.from_pretrained(self.model, lora_path)
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
                self.model = get_peft_model(self.model, lora_cfg)
                logger.info(f"Initialized LoRA adapters with config: {lora_cfg}")
        else:
            logger.warning("No LoRA config provided - all LLM parameters are frozen!")

        # Verify tokenizer and embedding alignment
        vocab_size = len(self.tokenizer)
        embed_size = self.model.get_input_embeddings().weight.shape[0]
        assert embed_size == vocab_size, f"Embedding size mismatch: {embed_size} != {vocab_size}"


    def freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        logger.info("llm_model frozen (eval mode)")


    def unfreeze(self):
        """
        Unfreezes model parameters for training:
        - LoRA adapters → fully trainable
        - Input embeddings → only new rows trainable
        - Output embeddings → only new rows trainable
        - All other backbone parameters → frozen
        Registers a gradient hook on input embeddings to freeze old vocab rows.
        """
        self.model.train()

        # -----------------------------
        # Gradient hook: freeze old vocab rows in input/output embeddings
        # -----------------------------
        def freeze_old_embeddings_hook(grad):
            # Freeze original rows, allow gradients only for new embeddings
            grad[:self.original_vocab_size] = 0
            return grad

        # -----------------------------
        # Iterate all parameters
        # -----------------------------
        for name, param in self.model.named_parameters():

            # LoRA adapters → trainable
            if "lora" in name.lower():
                param.requires_grad = True

            # Input embeddings
            elif "embed_tokens" in name:
                param.requires_grad = True
                # Hook to freeze old rows, only once
                if not hasattr(self, "_embedding_hook_registered"):
                    param.register_hook(freeze_old_embeddings_hook)
                    self._embedding_hook_registered = True

            # Output embeddings
            elif "lm_head" in name:
                param.requires_grad = True
                if not hasattr(self, "_lm_head_hook_registered"):
                    param.register_hook(freeze_old_embeddings_hook)
                    self._lm_head_hook_registered = True

            # Everything else → frozen
            else:
                param.requires_grad = False


        logger.info("llm_model unfrozen: LoRA adapters + special-token input/output embeddings trainable")


    def lora_parameters(self):
        """Returns only LoRA parameters (useful for separate optimizer)"""
        return [p for n, p in self.model.named_parameters() if "lora" in n.lower() and p.requires_grad]


    def save(self, ckpt_path):
        self.model.save_pretrained(ckpt_path + ".lora")
        logger.info(f"Saved LoRA adapters to {ckpt_path}.lora")

        input_embs = self.model.get_input_embeddings().weight[self.original_vocab_size : ].detach().cpu().clone()
        output_embs = self.model.get_output_embeddings().weight[self.original_vocab_size : ].detach().cpu().clone()
        torch.save({"special_tokens": self.special_tokens, "input_embeddings": input_embs, "output_embeddings": output_embs}, ckpt_path + ".embs.pt")
        logger.info(f"Saved special_tokens embeddings to {ckpt_path}.embs.pt")


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    parser = argparse.ArgumentParser(description="Instantiate LLM backbone")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    # Load JSON config
    with open(args.config, "r") as f:
        config = json.load(f)

    llm = LLM(config['llm'], config['lora'], config['embeddings'])
    logger.info("LLM successfully initialized")
