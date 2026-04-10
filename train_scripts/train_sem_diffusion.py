import torch
import tqdm
import os
from models.networks.diffusion_networks.graph_unet_hr import UNet3DModel
from models.networks.dualoctree_networks.graph_sem_vae import GraphVAE
from utils.util_sample_stuff import log_snr_schedule_cosine, log_snr_to_alpha_sigma
from dataset.voxel_dataset import get_dataloader
from utils.util_octree_stuff import (
    get_non_empty_mask, voxel_grid_to_points, points2octree, 
    voxel_to_patch, assign_octree_patch_features
)
from utils.config_loader import load_config

def train():
    # Load both configs
    v_cfg = load_config("configs/vae_config.yaml")["model"]
    d_cfg = load_config("configs/sem_diffusion_config.yaml")

    # 1. Load Pre-trained VAE (Eval Mode)
    vae = GraphVAE(
        depth=v_cfg["depth_in"],
        channel_in=v_cfg["channel_in"],
        nout=v_cfg["nout"],
        full_depth=v_cfg["full_depth"],
        latent_dim=v_cfg["latent_dim"],
        num_classes=v_cfg["total_classes"],
        resblk_num=v_cfg["resblk_num"]
    ).cuda()
    vae.load_state_dict(torch.load(d_cfg["vae_ref"]["checkpoint"]))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # 2. Initialize UNet Diffusion Model
    unet_model = UNet3DModel(
        image_size=d_cfg["unet"]["image_size"],
        input_depth=v_cfg["depth_in"],
        full_depth=v_cfg["full_depth"],
        in_channels=v_cfg["latent_dim"],
        model_channels=d_cfg["unet"]["model_channels"],
        lr_model_channels=d_cfg["unet"]["lr_model_channels"],
        out_channels=v_cfg["latent_dim"],
        num_res_blocks=d_cfg["unet"]["num_res_blocks"],
        channel_mult=d_cfg["unet"]["channel_mult"],
    ).cuda()

    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=d_cfg["training"]["lr"])
    loader = get_dataloader(d_cfg["training"]["data_dir"], batch_size=1)

    # 3. Training Loop
    for epoch in range(d_cfg["training"]["t_max"]):
        unet_model.train()
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")

        for vox, _ in pbar:
            vox = vox.cuda().long()
            
            # Use VAE to get latents
            with torch.no_grad():
                non_empty_mask = get_non_empty_mask(vox[0], 2)
                points = voxel_grid_to_points(non_empty_mask)
                octree = points2octree(points, depth=v_cfg["depth_in"], full_depth=v_cfg["full_depth"]).cuda()
                patch_feat = voxel_to_patch(vox, patch_size=2)
                assign_octree_patch_features(patch_feat[0], octree, v_cfg["depth_in"])
                
                _, mu, doctree = vae.extract_code(octree)
                z_clean = mu.cuda()

            # Diffusion Step
            t_cont = torch.rand(1, device=z_clean.device)
            logsnr_t = log_snr_schedule_cosine(t_cont)
            alpha_t, sigma_t = log_snr_to_alpha_sigma(logsnr_t)
            
            noise = torch.randn_like(z_clean)
            z_noisy = alpha_t * z_clean + sigma_t * noise

            pred_x0 = unet_model(z_noisy, doctree=doctree, timesteps=t_cont)
            loss = torch.nn.functional.mse_loss(pred_x0, z_clean)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet_model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if epoch % 10 == 0:
            os.makedirs(os.path.dirname(d_cfg["training"]["save_path"]), exist_ok=True)
            torch.save(unet_model.state_dict(), d_cfg["training"]["save_path"])

if __name__ == "__main__":
    train()