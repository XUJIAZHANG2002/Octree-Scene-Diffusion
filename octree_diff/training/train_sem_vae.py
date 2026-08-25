import torch
import tqdm
import os
from octree_diff.octree.build import build_semantic_octree
from octree_diff.models.factory import build_semantic_vae
from octree_diff.data.voxel_dataset import get_dataloader
from octree_diff.training.losses import compute_semantic_loss, compute_octree_loss
from octree_diff.config import load_stage
from octree_diff.training.class_weights import load_class_weights

def train(config_dir=None, device=None):
    # Load and split config for cleaner access
    full_config, _ = load_stage("sem_vae", config_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m_cfg = full_config["model"]
    t_cfg = full_config["training"]

    # 1. Setup Data
    loader = get_dataloader(t_cfg["data_dir"], batch_size=t_cfg["batch_size"])

    # 2. Initialize Model using m_cfg
    vae = build_semantic_vae(m_cfg, device)
    patch_size = m_cfg["patch_size"]
    optimizer = torch.optim.Adam(vae.parameters(), lr=t_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"])

    class_weights = load_class_weights(
        t_cfg.get("class_weights", "none"),
        t_cfg.get("class_weights_file", "data/replica_class_counts.pt"),
        m_cfg["total_classes"],
    ).to(device)

    # 3. Training Loop
    for epoch in range(t_cfg["epochs"]):
        vae.train()
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        
        for vox, _ in pbar:
            vox = vox.to(device).long()
            
            # Preprocessing
            octree = build_semantic_octree(vox, patch_size, m_cfg["depth_in"],
                                           m_cfg["full_depth"], device)
            
            # Forward Pass
            output = vae(octree, octree_out=octree, update_octree=False)
            
            # Loss Calculation
            sem_loss_dict = compute_semantic_loss(
                output['sem_voxs'], output['octree_out'], vox,
                class_weights
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
        os.makedirs(os.path.dirname(t_cfg["checkpoint_path"]), exist_ok=True)
        torch.save(vae.state_dict(), t_cfg["checkpoint_path"])

if __name__ == "__main__":
    train()