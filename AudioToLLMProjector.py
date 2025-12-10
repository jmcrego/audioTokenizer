import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AudioToLLMProjector(nn.Module):
    """
    Projects audio embeddings into LLM embedding space using superframe stacking,
    a low-rank MLP, and RoPE positional encoding.
    """

    def __init__(
        self,
        audio_embedding_dim: int,
        stack_size: int,
        llm_dimension: int=768,
        rank_dim: int=256,          # low-rank bottleneck
        max_seq_len: int=4096,
    ):
        """
        Args:
            audio_embedding_dim: Original audio frame dimension (e.g., 768 for mHuBERT)
            stack_size: Frames per superframe (e.g., 10)
            llm_dimension: Target LLM embedding size (e.g., 2048)
            rank_dim: Low-rank internal dimension (default 256)
        """
        super().__init__()

        self.audio_embedding_dim = audio_embedding_dim
        self.stack_size = stack_size
        self.llm_dimension = llm_dimension
        self.stacked_dim = audio_embedding_dim * stack_size
        self.rank_dim = rank_dim
        self.max_seq_len = max_seq_len

        # --- Low-Rank MLP ---
        # Equivalent to Linear(stacked_dim → rank_dim → llm_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.stacked_dim, rank_dim),
            nn.GELU(),
            nn.Linear(rank_dim, llm_dimension),
            nn.LayerNorm(llm_dimension),
        )

        # precompute the RoPE frequencies
        self.rope_freqs = build_rope_freqs(max_seq_len, llm_dimension)

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
        S = self.stack_size

        # every sequence of audio embeddings (T) in batch must be merged into superframes (S embeddings -> 1 superframes)
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
        rope_freqs = self.rope_freqs[:N] * self.stack_size  # [N, llm_dim//2]
        x = apply_rope(x, rope_freqs.to(x.device))


        # ---- build superframe mask ----
        # padded frames contaminate superframes so all superframes become masked
        if mask is None:
            sf_mask = None
        else:
            sf_mask = mask[:, :T2].view(B, N, S).all(dim=-1)

        return x, sf_mask

if __name__ == "__main__":
    import sys
    from AudioEmbedder import AudioEmbedder

    audio_files = [sys.argv[1]]
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    embedder = AudioEmbedder(model="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147", device=device)
    proj = AudioToLLMProjector(audio_embedding_dim=embedder.D, stack_size=4, llm_dimension=4096, rank_dim=256, max_seq_len=100).to(device)

    embeddings, masks = embedder(audio_files)  # embeddings: [B, T, D], masks: [B, T]
    print("Embeddings shape:", embeddings.shape)
    print("Masks shape:", masks.shape)

    embeddings = embeddings.to(device)
    masks = masks.to(device)

    llm_embeddings, sf_mask = proj(embeddings, masks)

    print("Projected LLM embeddings shape:", llm_embeddings.shape)
    print("Superframe mask shape:", sf_mask.shape)
    print("Superframe mask:", sf_mask)