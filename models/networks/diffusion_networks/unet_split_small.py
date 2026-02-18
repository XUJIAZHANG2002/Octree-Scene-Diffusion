import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(1, dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t):
        # Normalize and expand dims
        t = t.float().unsqueeze(-1) / 1000
        emb = self.linear1(t)
        emb = self.act(emb)
        return self.linear2(emb)

class UNet3DModelSmall(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels):
        super().__init__()
        self.time_embed = SimpleTimeEmbedding(model_channels)

        self.input_proj = nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)

        # Simple residual block (no dims arg)
        self.resblock = nn.Sequential(
            nn.Conv3d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(model_channels),
            nn.SiLU(),
            nn.Conv3d(model_channels, model_channels, kernel_size=3, padding=1),
        )

        self.out = nn.Sequential(
            nn.BatchNorm3d(model_channels),
            nn.SiLU(),
            nn.Conv3d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps):
        emb = self.time_embed(timesteps)  # [B, C]
        emb = emb[:, :, None, None, None]  # expand to [B, C, 1, 1, 1]

        h = self.input_proj(x)
        h = h + emb  # add timestep embedding
        h_res = self.resblock(h)
        h = h + h_res  # residual connection
        return self.out(h)