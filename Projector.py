# Projector.py

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Projector")

# class Projector(nn.Module):
#     """
#     Projects audio embeddings into LLM embedding space using superframe stacking
#     and a low-rank MLP with RMSNorm.
#     """
#     def __init__(self, config, audio_embedding_dim, llm_embedding_dim):
#         super().__init__()
#         logger.info(f"Initializing Projector {config}, audio_embedding_dim={audio_embedding_dim}, llm_embedding_dim={llm_embedding_dim}")

#         self.config = config
#         path = config.get('path', None)
#         self.stack_size = config.get('stack_size', 8)
#         rmsnorm_pre = config.get('rmsnorm_pre', True)
#         rmsnorm_mid = config.get('rmsnorm_mid', False)
#         rmsnorm_pos = config.get('rmsnorm_pos', True)
#         scale = config.get('scale', 0.0)
#         use_bias = config.get('use_bias', False)
#         middle_dim = config.get('middle_dim', llm_embedding_dim)

#         self.stacked_dim = audio_embedding_dim * self.stack_size
#         self.llm_embedding_dim = llm_embedding_dim

#         # --- Pre RMSNorm ---
#         self.ln_pre = nn.RMSNorm(self.stacked_dim) if rmsnorm_pre else nn.Identity()

#         # --- Projector MLP ---
#         self.linear1 = nn.Linear(self.stacked_dim, middle_dim, bias=False) # [256x6k] => [256x2k]
#         self.act = nn.SiLU()

#         # --- Mid RMSNorm ---
#         self.ln_mid = nn.RMSNorm(middle_dim) if rmsnorm_mid else nn.Identity()

#         # --- Output projector ---
#         self.linear2 = nn.Linear(middle_dim, self.llm_embedding_dim, bias=False)

#         # --- Post RMSNorm ---
#         self.ln_pos = nn.RMSNorm(self.llm_embedding_dim) if rmsnorm_pos else nn.Identity()

#         # --- Add a learnable scale to the projector ---
#         if scale > 0.:
#             self.register_parameter('scale', nn.Parameter(torch.tensor(scale)))
#         else:
#             self.scale = None

#         # --- Add learnable bias for better alignment ---
#         if use_bias:
#             self.register_parameter('bias', nn.Parameter(torch.zeros(self.llm_embedding_dim)))
#         else:
#             self.bias = None

#         # --- Load projector if path is provided ---
#         if path is not None:
#             state_dict = torch.load(path, map_location="cpu")
#             self.load_state_dict(state_dict, strict=True)
#             logger.info(f"Loaded Projector from {path}")
#         else:
#             # initialize with with Xavier uniform, rest are random
#             nn.init.xavier_uniform_(self.linear1.weight)
#             nn.init.xavier_uniform_(self.linear2.weight)
#             logger.info("Initialized Projector with random weights")



#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: audio frame embeddings, shape [B, T, D_audio]
#             mask: optional boolean mask, shape [B, T]
#         Returns:
#             proj_embs: projected embeddings, shape [B, T_new, D_llm]
#             proj_mask: mask for superframes, shape [B, T_new]
#         """
#         B, T, D_audio = x.shape
#         S = self.stack_size

#         # --- Frame stacking ---
#         T_trim = (T // S) * S
#         x = x[:, :T_trim, :]
#         if mask is not None:
#             mask = mask[:, :T_trim]

#         x = x.view(B, T_trim // S, D_audio * S) # [B, T_trim, D_audio] -> [B, T_trim//S, D_audio*S]
#         if mask is not None:
#             mask = mask.view(B, T_trim // S, S)
#             proj_mask = mask.any(dim=-1)
#         else:
#             proj_mask = torch.ones(B, T_trim // S, dtype=torch.bool, device=x.device)

#         # --- Pre RMSNorm ---
#         x = self.ln_pre(x)
#         # --- Linear1 + SiLU ---
#         x = self.linear1(x)
#         x = self.act(x)
#         # --- Mid RMSNorm ---
#         x = self.ln_mid(x)
#         # --- Linear2 ---
#         x = self.linear2(x)
#         # --- Post RMSNorm ---
#         x = self.ln_pos(x)
#         # --- HARD norm constraint ---
#         # x = F.normalize(x, dim=-1)

#         # --- Apply scale ---
#         # Option 1: Unconstrained scale (current)
#         if self.scale is not None:
#             x = self.scale * x        
#         # Option 2: Constrained scale (uncomment if needed)
#         # x = torch.clamp(self.scale, max=1.0) * x

#         # Add bias for better alignment
#         if self.bias is not None:
#             x = x + self.bias

#         return x, proj_mask



class Projector(nn.Module):
    def __init__(self, config, audio_embedding_dim, llm_embedding_dim):
        super().__init__()

        self.config = config
        self.audio_embedding_dim = audio_embedding_dim
        self.llm_embedding_dim = llm_embedding_dim

        path = config.get('path', None)
        conv_kernel = config.get('conv_kernel', 30)
        conv_stride = config.get('conv_stride', 30)
        rmsnorm_pre = config.get('rmsnorm_pre', True)
        act = (config.get('act', None) or '').lower()
        rmsnorm_pos = config.get('rmsnorm_pos', True)
        scale = config.get('scale', 0.0)
        use_bias = config.get('use_bias', False)

        assert 1500 % conv_stride == 0, f"conv_stride={conv_stride} must divide audio frames (1500) or frames will be dropped"

       # --- Pre RMSNorm ---
        self.ln_pre = nn.RMSNorm(audio_embedding_dim) if rmsnorm_pre else nn.Identity()

        # Depthwise Conv1d
        self.dw_conv = nn.Conv1d(in_channels=audio_embedding_dim, out_channels=audio_embedding_dim,
            kernel_size=conv_kernel,
            stride=conv_stride,
            groups=audio_embedding_dim,  # depthwise
            bias=False
        )

        # Pointwise conv to mix channels
        self.pw_conv = nn.Conv1d(in_channels=audio_embedding_dim, out_channels=audio_embedding_dim,
            kernel_size=1,
            bias=False
        )

        # Linear projection to LLM embedding
        self.linear = nn.Linear(audio_embedding_dim, llm_embedding_dim, bias=False)

        # Activation
        self.act = nn.SiLU() if act == 'silu' else nn.GELU() if act == 'gelu' else nn.ReLU() if act == 'relu' else nn.Identity()

        # Post-linear RMSNorm
        self.ln_post = nn.RMSNorm(llm_embedding_dim) if rmsnorm_pos else nn.Identity()

        # Optional learnable scale and bias
        if scale > 0:
            self.register_parameter('scale', nn.Parameter(torch.tensor(scale)))
        else:
            self.scale = None

        if use_bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(llm_embedding_dim)))
        else:
            self.bias = None

        # --- Load projector if path is provided ---
        if path is not None:
            state_dict = torch.load(path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded Projector from {path}")
        else:
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.xavier_uniform_(self.pw_conv.weight)
            nn.init.xavier_uniform_(self.dw_conv.weight)            
            logger.info("Initialized Projector with xavier_uniform")

        # Log parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Initialized AudioProjector with {total_params/1e6:.2f}M params")


    def forward(self, x):
        """
        x: [B, T, D_audio]
        returns:
            x_proj: [B, T_out, D_llm]
            mask: [B, T_out] boolean mask
        """
        B, T, D = x.shape
        assert D == self.audio_embedding_dim, f"Expected D={self.audio_embedding_dim}, got {D}"

        # --- Pre RMSNorm ---
        x = self.ln_pre(x)  # [B, T, D_audio]

        # --- Depthwise Conv1d ---
        x = x.transpose(1, 2)  # [B, D_audio, T]
        x = self.dw_conv(x)  # [B, D_audio, T_out]
        x = self.pw_conv(x)  # [B, D_audio, T_out]
        x = x.transpose(1, 2)  # [B, T_out, D_audio]
        T_out = x.size(1)

        # --- Linear + Activation ---
        x = self.linear(x)  # [B, T_out, D_llm]
        x = self.act(x)

        # --- Post RMSNorm ---
        x = self.ln_post(x)

        # --- Optional scale + bias ---
        if self.scale is not None:
            x = x * self.scale            
        if self.bias is not None:
            x = x + self.bias

        # Mask: all frames valid
        mask = torch.ones(B, T_out, dtype=torch.bool, device=x.device)

        return x, mask


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