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
        path = config['path']
        stack_size = config['stack_size']
        stacked_dim = audio_embedding_dim * stack_size
        rank_dim = config['rank_dim']

        # --- Low-Rank MLP ---
        self.proj = nn.Sequential(
            nn.Linear(stacked_dim, rank_dim),
            nn.SiLU(),
            nn.Linear(rank_dim, llm_embedding_dim),
            nn.RMSNorm(llm_embedding_dim),
        )

        # --- Load projector if given ---
        if path is not None:
            state_dict = torch.load(path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded Projector from {path}")
        else:
            logger.info("Initialized Projector with random weights")            


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x        : [B, T, D]
            mask     : [B, T] or None
        Returns:
            out      : [B, N, llm_dim]
            sf_mask  : [B, N]
        """
        B, T, D = x.shape
        S = self.config['stack_size']
        logger.debug(f"input.shape={x.shape}")

        # ---- pad to full superframe ----
        pad_len = (S - (T % S)) % S
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)

        T2 = x.shape[1] 
        assert T2 % S == 0
        N = T2 // S  # number of superframes after padding

        # ----´Stack frames into superframes ----
        x = x.view(B, N, S * D)  # stack frames into superframes [B, N, S*D]
        logger.debug(f"stacked superframes.shape={x.shape}")

        # ---- low-rank projection into llm space ----
        x = x.to(dtype=next(self.proj.parameters()).dtype)
        x = self.proj(x)  # [B, N, llm_dim]
        logger.debug(f"proj output.shape={x.shape}")

        # ---- superframe mask ----
        # A superframe is valid only if ALL its S frames are valid. If any frame is padded → entire superframe is masked out        
        sf_mask = None if mask is None else mask[:, :T2].view(B, N, S).all(dim=-1) # [B, N]
        logger.debug(f"proj mask.shape={sf_mask.shape}")

        logger.debug(f"proj mean={x.mean()} std={x.std()} norm={x.norm(dim=-1).mean()}")
        return x, sf_mask 
    


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