import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x):
        return self.block(x)

# -----------------------------
# Spatial-latent 16×16×16 VAE
# -----------------------------
class VoxelVAE(nn.Module):
    """
    Keeps spatial size (D,H,W) = (16,16,16).
    Encodes to latent with z_channels channels: z ~ N(mu, sigma) at each voxel.
    """
    def __init__(self, z_channels=4, base=32, in_ch=1):
        super().__init__()
        self.z_channels = z_channels

        # Encoder: [1,16,16,16] -> [base] -> [2*z_channels] (mu & logvar)
        self.enc_stem = ConvBlock(in_ch, base)         # [B, base, 16,16,16]
        self.enc_mid  = ConvBlock(base, base)          # [B, base, 16,16,16]
        self.to_mu    = nn.Conv3d(base, z_channels, kernel_size=3, padding=1)
        self.to_logv  = nn.Conv3d(base, z_channels, kernel_size=3, padding=1)

        # Decoder: z -> [base] -> [1]
        self.dec_in   = nn.Conv3d(z_channels, base, kernel_size=3, padding=1)
        self.dec_mid  = ConvBlock(base, base)
        self.dec_out  = nn.Conv3d(base, 1, kernel_size=3, padding=1)  # logits

    def encode(self, x):
        h = self.enc_stem(x)
        h = self.enc_mid(h)
        mu    = self.to_mu(h)
        logvar = self.to_logv(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_in(z)
        h = self.dec_mid(h)
        logits = self.dec_out(h)  # logits for BCE
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, z
