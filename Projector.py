# Projector.py

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Projector")

class Projector(nn.Module):
    """
    Projects audio embeddings into LLM embedding space using superframe stacking
    and a low-rank MLP with RMSNorm.
    """
    def __init__(self, config, audio_embedding_dim, llm_embedding_dim):
        super().__init__()
        logger.info(f"Initializing Projector {config}, audio_embedding_dim={audio_embedding_dim}, llm_embedding_dim={llm_embedding_dim}")

        self.config = config
        path = config.get('path', None)
        rmsnorm_pre = config.get('rmsnorm_pre', True)
        rmsnorm_mid = config.get('rmsnorm_mid', False)
        rmsnorm_pos = config.get('rmsnorm_pos', True)
        middle_dim = config.get('middle_dim', llm_embedding_dim)

        self.stack_size = config.get('stack_size', 8)
        self.stacked_dim = audio_embedding_dim * self.stack_size
        self.llm_embedding_dim = llm_embedding_dim


        # --- Pre RMSNorm ---
        self.ln_pre = nn.RMSNorm(self.stacked_dim) if rmsnorm_pre else nn.Identity()

        # --- Projector MLP ---
        self.linear1 = nn.Linear(self.stacked_dim, middle_dim, bias=False)
        self.act = nn.SiLU()

        # --- Mid RMSNorm ---
        self.ln_mid = nn.RMSNorm(middle_dim) if rmsnorm_mid else nn.Identity()

        # --- Output projector ---
        self.linear2 = nn.Linear(middle_dim, self.llm_embedding_dim, bias=False)

        # --- Post RMSNorm ---
        self.ln_pos = nn.RMSNorm(self.llm_embedding_dim) if rmsnorm_pos else nn.Identity()

        # --- Load projector if path is provided ---
        if path is not None:
            state_dict = torch.load(path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded Projector from {path}")
        else:
            logger.info("Initialized Projector with random weights")




    def forward(self, x, mask=None):
        """
        Args:
            x: audio frame embeddings, shape [B, T, D_audio]
            mask: optional boolean mask, shape [B, T]
        Returns:
            proj_embs: projected embeddings, shape [B, T_new, D_llm]
            proj_mask: mask for superframes, shape [B, T_new]
        """
        B, T, D_audio = x.shape
        S = self.stack_size

        # --- Frame stacking ---
        T_trim = (T // S) * S
        x = x[:, :T_trim, :]
        if mask is not None:
            mask = mask[:, :T_trim]

        x = x.view(B, T_trim // S, D_audio * S) # [B, T_trim, D_audio] -> [B, T_trim//S, D_audio*S]
        if mask is not None:
            mask = mask.view(B, T_trim // S, S)
            proj_mask = mask.any(dim=-1)
        else:
            proj_mask = torch.ones(B, T_trim // S, dtype=torch.bool, device=x.device)

        # --- Pre RMSNorm ---
        x = self.ln_pre(x)
        # --- Linear1 + SiLU ---
        x = self.linear1(x)
        x = self.act(x)
        # --- Mid RMSNorm ---
        x = self.ln_mid(x)
        # --- Linear2 ---
        x = self.linear2(x)
        # --- Post RMSNorm ---
        x = self.ln_pos(x)

        return x, proj_mask


if __name__ == "__main__":
    import json
    import argparse
    from Embedder import Embedder

    parser = argparse.ArgumentParser(description="Test Projector using an Embedder.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--audio_files", type=str, help="Comma separated list of audio files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    embedder = Embedder(config=config['audio'])
    projector = Projector(config=config['projector'], audio_embedding_dim=embedder.embedding_dim, llm_embedding_dim=2048)

    embed, masks = embedder(args.audio_files.split(","))  # embeddings: [B, T, D], masks: [B, T]
    print("Embeddings shape:", embed.shape)
    print("Masks shape:", masks.shape)

    proj_embed, proj_mask = projector(embed, masks)

    print("Projected LLM embeddings shape:", proj_embed.shape)
    print("Superframe mask shape:", proj_mask.shape)
    print("Superframe mask:", proj_mask)