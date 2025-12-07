import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioToLLMProjector(nn.Module):
    """
    Efficient Audio → LLM projector.
    Reduces computation by projecting audio → low rank → LLM dimension.
    """

    def __init__(
        self,
        audio_embedding_dim,
        stack_size,
        llm_dimension=768,
        rank=256,          # low-rank bottleneck
        max_seq_len=4096,
    ):
        """
        Args:
            audio_embedding_dim: Original audio frame dimension (e.g., 768 for mHuBERT)
            stack_size: Frames per superframe (e.g., 10)
            llm_dimension: Target LLM embedding size (e.g., 2048)
            rank: Low-rank internal dimension (default 256)
        """
        super().__init__()

        self.audio_embedding_dim = audio_embedding_dim
        self.stack_size = stack_size
        self.llm_dimension = llm_dimension
        self.stacked_dim = audio_embedding_dim * stack_size
        self.rank = rank
        self.max_seq_len = max_seq_len

        # Positional encoding for superframes
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, llm_dimension) * 0.01)

        # --- Low-Rank MLP ---
        # Equivalent to Linear(stacked_dim → rank → llm_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.stacked_dim, rank),
            nn.GELU(),
            nn.Linear(rank, llm_dimension),
            nn.LayerNorm(llm_dimension),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x        : [B, T, D]
            mask     : [B, T] or None

        Returns:
            out      : [B, N, llm_dim]
            sf_mask  : [B, N]
        """

        B, T, D = x.shape
        S = self.stack_size

        # ---- pad to full superframe ----
        pad_len = (S - (T % S)) % S
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)

        T2 = x.shape[1] # after padding
        N = T2 // S  # number of superframes

        x = x.view(B, N, S * D)  # merge frames (stack)

        # ---- low-rank projection ----
        x = self.proj(x)  # [B, N, llm_dim]

        # ---- positional encoding ----
        x = x + self.pos_embed[:, :N]

        # ---- build superframe mask ----
        if mask is None:
            sf_mask = None
        else:
            sf_mask = mask[:, :T2].view(B, N, S).any(dim=-1)

        return x, sf_mask
