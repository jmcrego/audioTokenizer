# Projector.py

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Projector")


class Projector(nn.Module):
    """
    Projects audio embeddings into LLM embedding space using superframe stacking,
    a low-rank MLP.
    """

    def __init__(self, config, audio_embedding_dim, llm_embedding_dim):
        """
        Args:
            config contains:
            audio_embedding_dim: Original audio frame dimension (e.g., 768 for mHuBERT)
            llm_dimension: Target LLM embedding size (e.g., 2048)
        """
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
            nn.GELU(),
            nn.Linear(rank_dim, llm_embedding_dim),
            nn.LayerNorm(llm_embedding_dim),
        )

        # scale output to match the llm embeddings norm
        self.scale = nn.Parameter(torch.tensor(0.03))  # ~1 / sqrt(2048) => 1 / 45 ≈ 0.022

        if path is not None:
            state_dict = torch.load(path, map_location="cpu")
            missing, unexpected = self.load_state_dict(state_dict, strict=False) # will load everythin matching, if something new  will leave the new model just created  ###jmcc this wont be needed in future (use the commented code below)
            logger.info(f"Missing keys: {missing}, unexpected keys: {unexpected}")
        else:
            logger.info("Initialized Projector with random weights")            

        logger.info(f"Projector scale = {self.scale.item()}")

        # load projector if given
        # if path is not None:
        #     state_dict = torch.load(path, map_location="cpu")
        #     self.load_state_dict(state_dict, strict=True)
        #     logger.info(f"Loaded Projector from {path}")
        # else:
        #     logger.info("Initialized Projector with random weights")            


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

        # every sequence of audio embeddings (T) in batch must be merged into superframes (S embeddings -> 1 superframe)
        # this may introduce pad embeddings at the end (to fit superframe size S)
        # Superframes with any frame/embedding consisting of pad will then be discarded (the entire superframe is masked)

        # ---- pad to full superframe ----
        pad_len = (S - (T % S)) % S
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)


        T2 = x.shape[1] 
        assert T2 % S == 0

        # ----´Stack frames into superframes ----

        N = T2 // S  # number of superframes after padding
        x = x.view(B, N, S * D)  # stack frames into superframes [B, N, S*D]

        # ---- low-rank projection into llm space ----

        x = x.to(dtype=next(self.proj.parameters()).dtype)
        x = self.proj(x)  # [B, N, llm_dim]

        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6) # ensures L2 normalization per vector
        x = x * torch.clamp(self.scale, 0.01, 0.2) #scales to LLM-compatible (similar embeddings) magnitude, safely bounded 

        # ---- superframe mask ----

        # A superframe is valid only if ALL its S frames are valid
        # If any frame is padded → entire superframe is masked out        
        if mask is None:
            sf_mask = None
        else:
            sf_mask = mask[:, :T2].view(B, N, S).all(dim=-1) # [B, N]

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