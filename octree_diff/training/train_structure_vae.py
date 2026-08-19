import torch
import torch.nn.functional as F
import tqdm
import os
from octree_diff.models.factory import build_structure_vae
from octree_diff.data.split_dataset import get_split_dataloader
from octree_diff.config import load_stage


def train(config_dir=None, device=None):
    # Load configuration
    config, _ = load_stage("str_vae", config_dir)
    m_cfg = config["model"]
    t_cfg = config["training"]

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. Setup Data
    loader = get_split_dataloader(t_cfg["data_dir"], batch_size=t_cfg["batch_size"])

    # 2. Initialize Model
    model = build_structure_vae(m_cfg, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"])
    
    print(f"Starting Structure VAE Training on {device}...")

    # 3. Training Loop
    for epoch in range(t_cfg["epochs"]):
        model.train()
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for vox, _ in pbar:
            vox = vox.to(device) # Shape: [B, 1, 32, 32, 32] or [B, 1, 16, 16, 16]

            logits, mu, logvar, z = model(vox)
            
            # Loss Calculation
            # 1. Reconstruction: Binary Cross Entropy with Logits
            # vox is treated as targets (0 or 1)
            recon_loss = F.mse_loss(logits, vox, reduction='sum') # Average per batch
            
            # 2. KL Divergence for VAE Bottleneck
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Scale loss by batch size
            total_loss = (recon_loss + 0.01*kl_loss) 

            # Optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.set_postfix({
                "recon": f"{recon_loss.item():.2f}", 
                "kl": f"{kl_loss.item():.2f}"
            })

        # Save checkpoint
        os.makedirs(os.path.dirname(t_cfg["checkpoint_path"]), exist_ok=True)
        torch.save(model.state_dict(), t_cfg["checkpoint_path"])
        print(f" Saved checkpoint to {t_cfg['checkpoint_path']}")
if __name__ == "__main__":
    train()