# Projector.py

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Projector")

def build_rope_freqs(n_positions: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    Build rotary positional embedding frequencies.

    Args:
        n_positions: maximum number of positions
        dim: embedding dimension (should be even)
        base: RoPE base (default 10000)

    Returns:
        freqs: [n_positions, dim//2] tensor
    """
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even.")
    half = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, 1).float() / half))
    positions = torch.arange(n_positions).float()[:, None]
    return positions * freqs[None, :]  # shape [n_positions, half]

def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embedding to input tensor.

    Args:
        x: [B, N, D] input embeddings
        freqs: [N, D//2] RoPE frequencies

    Returns:
        x_rot: [B, N, D] embeddings with RoPE applied
    """
    B, N, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    cos = freqs.cos().unsqueeze(0)  # [1, N, D//2]
    sin = freqs.sin().unsqueeze(0)

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    return torch.cat([x1_rot, x2_rot], dim=-1)



class Projector(nn.Module):
    """
    Projects audio embeddings into LLM embedding space using superframe stacking,
    a low-rank MLP, and RoPE positional encoding.
    """

    def __init__(self, config, audio_embedding_dim, llm_embedding_dim):
        """
        Args:
            config contains:
            audio_embedding_dim: Original audio frame dimension (e.g., 768 for mHuBERT)
            llm_dimension: Target LLM embedding size (e.g., 2048)
        """
        super().__init__()
        logger.info(f"Initializing Projector {config} audio_embedding_dim={audio_embedding_dim} llm_embedding_dim={llm_embedding_dim}")

        self.config = config
        path = config['path']
        stack_size = config['stack_size']
        stacked_dim = audio_embedding_dim * stack_size
        rank_dim = config['rank_dim']
        max_seq_len = config['max_seq_len']

        # --- Low-Rank MLP ---
        self.proj = nn.Sequential(
            nn.Linear(stacked_dim, rank_dim),
            nn.GELU(),
            nn.Linear(rank_dim, llm_embedding_dim),
            nn.LayerNorm(llm_embedding_dim),
        )

        # precompute the RoPE frequencies
        rope_freqs = build_rope_freqs(max_seq_len, llm_embedding_dim)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # load projector if given
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

        # every sequence of audio embeddings (T) in batch must be merged into superframes (S embeddings -> 1 superframe)
        # this may introduce pad embeddings at the end (to fit superframe size S)
        # Superframes with any frame/embedding consisting of pad will then be discarded (the entire superframe is masked)

        # ---- pad to full superframe ----
        pad_len = (S - (T % S)) % S
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)

        T2 = x.shape[1] # after padding
        N = T2 // S  # number of superframes

        x = x.view(B, N, S * D)  # stack frames into superframes [B, N, S*D]

        # ---- low-rank projection ----
        x = self.proj(x)  # [B, N, llm_dim]

        # Apply RoPE (scale positions by stack_size for superframes)
        rope_freqs = self.rope_freqs[:N].to(x.device, x.dtype) * S # [N, llm_dim//2]
        x = apply_rope(x, rope_freqs)

        # ---- build superframe mask ----
        # padded frames contaminate superframes... all superframes become masked if any of their frames are masked
        if mask is None:
            sf_mask = None
        else:
            sf_mask = mask[:, :T2].view(B, N, S).all(dim=-1)

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