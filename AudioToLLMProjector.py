import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("AudioToLLMProjector")

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
        proj_path: str = None,
        audio_embedding_dim: int,
        stack_size: int,
        llm_dimension: int=768,
        rank_dim: int=256,          # low-rank bottleneck
        max_seq_len: int=4096,
        device: str='cpu',
        dtype: torch.dtype=torch.float32,
    ):
        """
        Args:
            audio_embedding_dim: Original audio frame dimension (e.g., 768 for mHuBERT)
            stack_size: Frames per superframe (e.g., 10)
            llm_dimension: Target LLM embedding size (e.g., 2048)
            rank_dim: Low-rank internal dimension (default 256)
        """
        meta = {k: v for k, v in locals().items() if k != "self"}
        logger.info(f"Initializing {meta}")
        super().__init__()

        self.audio_embedding_dim = audio_embedding_dim
        self.stack_size = stack_size
        self.llm_dimension = llm_dimension
        self.stacked_dim = audio_embedding_dim * stack_size
        self.rank_dim = rank_dim
        self.max_seq_len = max_seq_len

        # --- Low-Rank MLP ---
        self.proj = nn.Sequential(
            nn.Linear(self.stacked_dim, rank_dim),
            nn.GELU(),
            nn.Linear(rank_dim, llm_dimension),
            nn.LayerNorm(llm_dimension),
        )

        # load projector if given
        if proj_path is not None:
            self.projector.load(proj_path, device=device)

        self.proj = self.proj.to(device, dtype=dtype)

        # precompute the RoPE frequencies
        self.rope_freqs = build_rope_freqs(max_seq_len, llm_dimension)
        logger.info(f"Read projector {proj_path}")


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
        rope_freqs = self.rope_freqs[:N].to(x.device, x.dtype) * self.stack_size # [N, llm_dim//2]
        x = apply_rope(x, rope_freqs)

        # ---- build superframe mask ----
        # padded frames contaminate superframes... all superframes become masked if any of their frames are masked
        if mask is None:
            sf_mask = None
        else:
            sf_mask = mask[:, :T2].view(B, N, S).all(dim=-1)

        return x, sf_mask
    
    def load(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)
        print(f"Loaded AudioToLLMProjector from {path}")

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Saved AudioToLLMProjector to {path}")


if __name__ == "__main__":
    import argparse
    from AudioEmbedder import AudioEmbedder

    parser = argparse.ArgumentParser(description="Test Projector using an Audio Embedder.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147", help="Path or HuggingFace model name (i.e. openai/whisper-small, utter-project/mhubert-147, facebook/wav2vec2-xlsr-53 models)")
    parser.add_argument("--wav", type=str, help="Comma separated list of paths to audio files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])

    audio_files = args.wav.split(",")
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    embedder = AudioEmbedder(model="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147", device=device)
    proj = AudioToLLMProjector(audio_embedding_dim=embedder.D, stack_size=8, llm_dimension=4096, rank_dim=256, max_seq_len=100).to(device)

    embeddings, masks = embedder(audio_files)  # embeddings: [B, T, D], masks: [B, T]
    print("Embeddings shape:", embeddings.shape)
    print("Masks shape:", masks.shape)

    embeddings = embeddings.to(device)
    masks = masks.to(device)

    llm_embeddings, sf_mask = proj(embeddings, masks)

    print("Projected LLM embeddings shape:", llm_embeddings.shape)
    print("Superframe mask shape:", sf_mask.shape)
    print("Superframe mask:", sf_mask)