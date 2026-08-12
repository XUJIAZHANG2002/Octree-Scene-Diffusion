import torch
import torch.nn.functional as F
import tqdm
import os
from models.structure_networks.structure_vae import VoxelVAE
from dataset.split_dataset import get_split_dataloader
from utils.config_loader import load_config
from utils.util_visual_stuff import *
def train():
    # Load configuration
    config = load_config("configs/structure_vae_config.yaml")
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
@torch.no_grad()
def inference():
    # 1. Load configuration
    config = load_config("configs/structure_vae_config.yaml")
    m_cfg = config["model"]
    t_cfg = config["training"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Initialize Model and Load Weights immediately
    model = VoxelVAE(
        z_channels=m_cfg["z_channels"],
        base=m_cfg["base_channels"],
        in_ch=m_cfg["in_channels"]
    ).to(device)

    checkpoint_path = t_cfg["checkpoint_path"]
    if os.path.exists(checkpoint_path):
        print(f"📦 Loading weights from {checkpoint_path}...")
        # weights_only=True silences the FutureWarning
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    else:
        print(f"⚠️ Warning: No checkpoint found at {checkpoint_path}. Running with random weights.")

    model.eval()

    # 3. Setup Data - get exactly one sample
    # batch_size=1 so we only process one "room/scene"
    loader = get_split_dataloader(t_cfg["data_dir"], batch_size=1)
    
    # Get the first sample from the loader
    vox, _ = next(iter(loader))
    vox = vox.to(device)

    print(f"📥 Input shape: {vox.shape}")

    # 4. Forward Pass
    # We only care about logits for reconstruction/visualization
    logits, mu, logvar, z = model(vox)
    print(z.mean(),z.std())
    logits_vis = logits.clone()
    logits_vis[logits_vis < 0] = 0.0
    logits_vis[logits_vis > 0] = 1.0
    from utils.util_visual_stuff import visualize_kitti_instance
    visualize_kitti_instance(logits_vis[0][0])
    print(f"📤 Output Logits shape: {logits.shape}")
    print(f"📊 Logits Range: min={logits.min().item():.4f}, max={logits.max().item():.4f}")

    # Optional: convert to occupancy (probability)
    probs = torch.sigmoid(logits)
    occupancy = (probs > 0.5).float()
    print(f"✅ Occupancy count: {torch.sum(occupancy).item()} non-empty voxels.")

    return logits
if __name__ == "__main__":
    train()