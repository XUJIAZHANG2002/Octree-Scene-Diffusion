import torch
import tqdm
import os
from models.networks.dualoctree_networks.graph_sem_vae import GraphVAE
from dataset.voxel_dataset import get_dataloader
from utils.util_octree_stuff import (
    get_non_empty_mask, 
    voxel_grid_to_points, 
    points2octree, 
    voxel_to_patch, 
    assign_octree_patch_features
)
from loss.octree_losses import compute_semantic_loss, compute_octree_loss
from utils.config_loader import load_config

def train():
    # Load and split config for cleaner access
    full_config = load_config("configs/sem_vae_config.yaml")
    m_cfg = full_config["model"]
    t_cfg = full_config["training"]

    # 1. Setup Data
    loader = get_dataloader(t_cfg["data_dir"], batch_size=t_cfg["batch_size"])

    # 2. Initialize Model using m_cfg
    vae = GraphVAE(
        depth=m_cfg["depth_in"],
        channel_in=m_cfg["channel_in"],
        nout=m_cfg["nout"],
        full_depth=m_cfg["full_depth"],
        depth_stop=m_cfg["depth_stop"],
        depth_out=m_cfg["depth_in"],
        latent_dim=m_cfg["latent_dim"],
        num_classes=m_cfg["total_classes"],
        resblk_num=m_cfg["resblk_num"],
    ).cuda()
    patch_size = m_cfg["patch_size"]
    optimizer = torch.optim.Adam(vae.parameters(), lr=t_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"])

    # 3. Training Loop
    for epoch in range(t_cfg["epochs"]):
        vae.train()
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        
        for vox, _ in pbar:
            vox = vox.cuda().long()
            
            # Preprocessing
            non_empty_mask = get_non_empty_mask(vox[0], patch_size)
            points = voxel_grid_to_points(non_empty_mask)
            octree = points2octree(points, depth=m_cfg["depth_in"], full_depth=m_cfg["full_depth"]).cuda()
            patch_feat = voxel_to_patch(vox, patch_size)
            assign_octree_patch_features(patch_feat[0], octree, m_cfg["depth_in"])
            
            # Forward Pass
            output = vae(octree, octree_out=octree, update_octree=False)
            
            # Loss Calculation
            sem_loss_dict = compute_semantic_loss(
                output['sem_voxs'], output['octree_out'], vox, 
                torch.ones(m_cfg["total_classes"]).cuda()
            )
            oct_loss_dict = compute_octree_loss(output['logits'], octree)
            
            total_loss = output['kl_loss'] + sem_loss_dict["sem_loss_6"]
            total_loss += sum(v for k, v in oct_loss_dict.items() if 'loss_' in k)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{total_loss.item():.3f}", 
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "acc": f"{sem_loss_dict.get('sem_accu_6', 0.0):.3f}"
            })

        scheduler.step()
        
        # Save checkpoint
        os.makedirs(os.path.dirname(t_cfg["save_path"]), exist_ok=True)
        torch.save(vae.state_dict(), t_cfg["save_path"])

if __name__ == "__main__":
    train()