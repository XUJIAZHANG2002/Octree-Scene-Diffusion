import torch
import tqdm
import os
import warnings

# Ignore the FutureWarning for torch.load as requested previously
warnings.filterwarnings("ignore", category=FutureWarning)

from models.structure_networks.structure_vae import VoxelVAE
from models.structure_networks.unet_3d import StructureUNet # Ensure your provided code is in this file
from dataset.split_dataset import get_split_dataloader
from utils.config_loader import load_config
from utils.util_sample_stuff import log_snr_schedule_cosine, log_snr_to_alpha_sigma

def train():
    # Load configurations
    full_vae_cfg = load_config("configs/structure_vae_config.yaml")
    vae_cfg = full_vae_cfg["model"]
    
    full_d_cfg = load_config("configs/structure_diffusion_config.yaml")
    d_cfg = full_d_cfg["unet"]
    t_cfg = full_d_cfg["training"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Frozen Structural VAE
    vae = VoxelVAE(
        z_channels=vae_cfg["z_channels"], 
        base=vae_cfg["base_channels"]
    ).to(device)
    vae.load_state_dict(torch.load(full_d_cfg["vae_ref"]["checkpoint"], weights_only=True))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # 2. Initialize your specific StructureUNet
    model = StructureUNet(
        in_ch=d_cfg["in_ch"],
        base_ch=d_cfg["base_ch"],
        time_emb_dim=d_cfg["time_emb_dim"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"])
    loader = get_split_dataloader(t_cfg["data_dir"], batch_size=t_cfg["batch_size"])

    print(f"Starting Structure Diffusion training on {device}...")

    # 3. Training Loop
    for epoch in range(t_cfg["t_max"]):
        model.train()
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        
        for vox, _ in pbar:
            vox = vox.to(device) # Expected shape [B, 1, 32, 32, 32]
            
            # Get Latents from VAE
            with torch.no_grad():
                mu, _ = vae.encode(vox)
                z_clean = mu # Shape [B, in_ch, 32, 32, 32]
            
            # Diffusion Process
            # t must be a 1D tensor of shape [B] for your SinusoidalPosEmb
            t = torch.rand(vox.size(0), device=device)
            
            logsnr_t = log_snr_schedule_cosine(t)
            alpha_t, sigma_t = log_snr_to_alpha_sigma(logsnr_t)
            
            # Reshape scalars for 5D broadcasting: [B, 1, 1, 1, 1]
            alpha_t = alpha_t.view(-1, 1, 1, 1, 1)
            sigma_t = sigma_t.view(-1, 1, 1, 1, 1)
            
            noise = torch.randn_like(z_clean)
            z_noisy = alpha_t * z_clean + sigma_t * noise
            
            # Forward Pass through your UNet
            # Note: your forward(x, t) handles the time embedding internally
            pred_x0 = model(z_noisy, t)
            
            loss = torch.nn.functional.mse_loss(pred_x0, z_clean)
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"mse": f"{loss.item():.4f}"})

        # Save checkpoint periodically
        if epoch % 5 == 0 or epoch == t_cfg["t_max"] - 1:
            os.makedirs(os.path.dirname(t_cfg["save_path"]), exist_ok=True)
            torch.save(model.state_dict(), t_cfg["save_path"])

if __name__ == "__main__":
    train()