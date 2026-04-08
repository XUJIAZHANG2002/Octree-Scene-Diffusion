import torch
import torch.nn.functional as F
import tqdm
import os
from models.structure_networks.structure_vae import VoxelVAE
from dataset.split_dataset import get_split_dataloader
from utils.config_loader import load_config

def train():
    # Load configuration
    config = load_config("configs/structure_config.yaml")
    m_cfg = config["model"]
    t_cfg = config["training"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Setup Data
    loader = get_split_dataloader(t_cfg["data_dir"], batch_size=t_cfg["batch_size"])

    # 2. Initialize Model
    model = VoxelVAE(
        z_channels=m_cfg["z_channels"],
        base=m_cfg["base_channels"],
        in_ch=m_cfg["in_channels"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"])
    
    print(f"Starting Structure VAE Training on {device}...")

    # 3. Training Loop
    for epoch in range(t_cfg["epochs"]):
        model.train()
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for vox, _ in pbar:
            vox = vox.to(device) # Shape: [B, 1, 32, 32, 32] or [B, 1, 16, 16, 16]
            
            # Forward Pass
            logits, mu, logvar, z = model(vox)
            
            # Loss Calculation
            # 1. Reconstruction: Binary Cross Entropy with Logits
            # vox is treated as targets (0 or 1)
            recon_loss = F.binary_cross_entropy_with_logits(logits, vox, reduction='sum')
            
            # 2. KL Divergence for VAE Bottleneck
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Scale loss by batch size
            total_loss = (recon_loss + kl_loss) / vox.size(0)

            # Optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_recon_loss += recon_loss.item() / vox.size(0)
            epoch_kl_loss += kl_loss.item() / vox.size(0)
            pbar.set_postfix({
                "recon": f"{recon_loss.item()/vox.size(0):.2f}", 
                "kl": f"{kl_loss.item()/vox.size(0):.2f}"
            })

        # Save checkpoint
        if epoch % 10 == 0:
            os.makedirs(os.path.dirname(t_cfg["checkpoint_path"]), exist_ok=True)
            torch.save(model.state_dict(), t_cfg["checkpoint_path"])
            print(f" Saved checkpoint to {t_cfg['checkpoint_path']}")

if __name__ == "__main__":
    train()