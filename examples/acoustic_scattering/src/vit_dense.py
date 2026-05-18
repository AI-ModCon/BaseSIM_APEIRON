"""Dense Vision Transformer for pixel-level regression (next-frame prediction)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn, Tensor


class PatchEmbed(nn.Module):
    """Convert (B, C_in, H, W) → (B, N, embed_dim) via non-overlapping Conv2d."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        gh, gw = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x, gh, gw


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block (LayerNorm → MHSA → LayerNorm → FFN)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class DenseViT(nn.Module):
    """Vision Transformer with a dense (per-patch) decoder head.

    Input:  (B, in_channels, H, W)
    Output: (B, 1, H, W)

    The encoder tokenises the image into non-overlapping patches, processes
    them through standard Transformer blocks, then a linear head maps each
    token back to patch_size^2 pixels which are rearranged (un-patchified)
    into the output spatial grid.
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.pos_embed: nn.Parameter  # set lazily or via fixed init

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Dense decoder: project each token → patch_size^2 pixel values
        self.head = nn.Linear(embed_dim, patch_size * patch_size)

        # Positional embedding is allocated for up to 1024 tokens; we slice
        # at forward time so the same weights work for varying resolutions.
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                nn.init.trunc_normal_(m.weight, std=math.sqrt(2.0 / fan_in))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _unpatchify(self, x: Tensor, gh: int, gw: int) -> Tensor:
        """Rearrange (B, N, P*P) → (B, 1, H, W)."""
        B = x.shape[0]
        P = self.patch_size
        x = x.reshape(B, gh, gw, P, P)
        # (B, gh, gw, P, P) → (B, 1, gh*P, gw*P)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, 1, gh * P, gw * P)
        return x

    def forward(self, x: Tensor) -> Tensor:
        tokens, gh, gw = self.patch_embed(x)
        n_tokens = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :n_tokens, :]

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)
        pixels = self.head(tokens)  # (B, N, P*P)
        return self._unpatchify(pixels, gh, gw)


def vit_dense_base(**kwargs) -> DenseViT:
    """DenseViT-Base: 768d, 12 layers, 12 heads (~86 M params)."""
    defaults = dict(embed_dim=768, depth=12, num_heads=12, patch_size=16, in_channels=4)
    defaults.update(kwargs)
    return DenseViT(**defaults)


def vit_dense_large(**kwargs) -> DenseViT:
    """DenseViT-Large: 1024d, 24 layers, 16 heads (~304 M params)."""
    defaults = dict(
        embed_dim=1024, depth=24, num_heads=16, patch_size=16, in_channels=4
    )
    defaults.update(kwargs)
    return DenseViT(**defaults)


def vit_dense_small(**kwargs) -> DenseViT:
    """DenseViT-Small: 384d, 6 layers, 6 heads (~22 M params)."""
    defaults = dict(embed_dim=384, depth=6, num_heads=6, patch_size=16, in_channels=4)
    defaults.update(kwargs)
    return DenseViT(**defaults)
