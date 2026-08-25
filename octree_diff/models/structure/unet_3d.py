import torch
import torch.nn as nn
from octree_diff.models.semantic.graph_unet_hr import timestep_embedding

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Sinusoidal timestep embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=device) * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# --- Residual Block with FiLM time conditioning ---
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_ch, affine=True)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch)
            )
        else:
            self.time_mlp = None

    def forward(self, x, t_emb=None):
        h = self.conv1(self.act(self.norm1(x)))
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb).view(t_emb.size(0), -1, 1, 1, 1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

# --- UNet3D for diffusion ---
class StructureUNet(nn.Module):
    def __init__(self, in_ch, base_ch=64, time_emb_dim=128):
        super().__init__()
        # timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim*4),
            nn.SiLU(),
            nn.Linear(time_emb_dim*4, time_emb_dim)
        )

        # Encoder
        self.enc1 = ResBlock3D(in_ch, base_ch, time_emb_dim)
        self.enc2 = ResBlock3D(base_ch, base_ch*2, time_emb_dim)
        self.enc3 = ResBlock3D(base_ch*2, base_ch*4, time_emb_dim)
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.mid = ResBlock3D(base_ch*4, base_ch*8, time_emb_dim)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ResBlock3D(base_ch*8, base_ch*4, time_emb_dim)

        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ResBlock3D(base_ch*4, base_ch*2, time_emb_dim)

        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = ResBlock3D(base_ch*2, base_ch, time_emb_dim)

        # Output
        self.out_norm = nn.InstanceNorm3d(base_ch, affine=True)
        self.out_conv = nn.Conv3d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        # embed timestep
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)             # [B,64,32,32,32]
        e2 = self.enc2(self.pool(e1), t_emb) # [B,128,16,16,16]
        e3 = self.enc3(self.pool(e2), t_emb) # [B,256,8,8,8]

        # Bottleneck
        m = self.mid(self.pool(e3), t_emb)   # [B,512,4,4,4]

        # Decoder
        d3 = self.up3(m)                     # [B,256,8,8,8]
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb)

        d2 = self.up2(d3)                    # [B,128,16,16,16]
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)

        d1 = self.up1(d2)                    # [B,64,32,32,32]
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)

        out = self.out_conv(F.silu(self.out_norm(d1)))
        return out
