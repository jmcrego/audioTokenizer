# Projector.py

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Projector")


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


    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        logger.info("Projector frozen (eval mode)")


    def unfreeze(self):
        self.train()
        for p in self.parameters():
            p.requires_grad = True
        logger.info("Projector unfrozen (train mode)")


    def save(self, ckpt_path):
        torch.save(self.state_dict(), ckpt_path + ".proj.pt")
        logger.info(f"Saved Projector to {ckpt_path}.proj.pt")


    def forward(self, x):
        """
        x: [B, T, D_audio]
        returns:
            x_proj: [B, T_out, D_llm]
            mask: [B, T_out] boolean mask
        """
        B, T, D = x.shape
        assert D == self.audio_embedding_dim, f"Expected D={self.audio_embedding_dim}, got {D}"

        # --- Convolutions ---
        x = self.ln_pre(x)    # [B, T, D_audio]
        x = x.transpose(1, 2) # [B, D_audio, T]
        x = self.dw_conv(x)   # [B, D_audio, T_out]
        x = self.pw_conv(x)   # [B, D_audio, T_out]
        x = x.transpose(1, 2) # [B, T_out, D_audio]
        # --- Linear + Activation ---
        x = self.linear(x)    # [B, T_out, D_llm]
        x = self.act(x)
        # --- Post RMSNorm ---
        x = self.ln_post(x)
        # --- Optional scale + bias ---
        if self.scale is not None:
            x = x * self.scale            
        if self.bias is not None:
            x = x + self.bias
        T_out = x.size(1)
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