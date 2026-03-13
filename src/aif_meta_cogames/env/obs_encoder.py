"""Observation encoders: token obs (200, 3) uint8 -> latent vector."""

import torch
import torch.nn as nn


class FlatEncoder(nn.Module):
    """Flatten and MLP encode token observations.

    Input: (batch, 200, 3) uint8
    Output: (batch, z_dim) float
    """

    def __init__(self, z_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(200 * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.reshape(obs.shape[0], -1).float() / 255.0
        return self.net(x)


class TokenTransformerEncoder(nn.Module):
    """Transformer encoder treating 200 tokens as a set.

    Each token (location, feature_id, value) is embedded and pooled.
    Input: (batch, 200, 3) uint8
    Output: (batch, z_dim) float
    """

    def __init__(self, z_dim: int = 64, embed_dim: int = 32, n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        self.location_embed = nn.Embedding(256, embed_dim)
        self.feature_embed = nn.Embedding(256, embed_dim)
        self.value_proj = nn.Linear(1, embed_dim)

        self.combine = nn.Linear(3 * embed_dim, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = nn.Linear(embed_dim, z_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        loc = self.location_embed(obs[:, :, 0].long())
        feat = self.feature_embed(obs[:, :, 1].long())
        val = self.value_proj(obs[:, :, 2:3].float() / 255.0)

        tokens = self.combine(torch.cat([loc, feat, val], dim=-1))

        # Mask empty tokens (location == 0xFF)
        mask = obs[:, :, 0] == 255

        out = self.transformer(tokens, src_key_padding_mask=mask)
        # Mean pool over non-empty tokens
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        pooled = out.sum(dim=1) / counts.squeeze(-1)

        return self.pool(pooled)
