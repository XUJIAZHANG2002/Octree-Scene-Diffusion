import torch
import tqdm
import os
from octree_diff.models.semantic.graph_unet_hr import UNet3DModel
from octree_diff.models.semantic.graph_sem_vae import GraphVAE
from octree_diff.diffusion.util_sample_stuff import log_snr_schedule_cosine, log_snr_to_alpha_sigma
from octree_diff.data.voxel_dataset import get_dataloader
from octree_diff.training.ema import EMA, lr_at
from octree_diff.training.latent_scale import resolve_latent_scale
from octree_diff.octree.util_octree_stuff import (
    get_non_empty_mask, voxel_grid_to_points, points2octree,
    voxel_to_patch, assign_octree_patch_features
)
from octree_diff.config import load_config

def train():
    # Load both configs
    v_cfg = load_config("configs/sem_vae_config.yaml")["model"]
    d_cfg = load_config("configs/sem_diffusion_config.yaml")
    t_cfg = d_cfg["training"]

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

    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=t_cfg["lr"])
    loader = get_dataloader(t_cfg["data_dir"], batch_size=1)
    patch_size = v_cfg["patch_size"]

    def encode(vox):
        """Voxel labels -> (octree latents, dual octree)."""
        non_empty_mask = get_non_empty_mask(vox[0], patch_size)
        points = voxel_grid_to_points(non_empty_mask)
        octree = points2octree(points, depth=v_cfg["depth_in"],
                               full_depth=v_cfg["full_depth"]).cuda()
        patch_feat = voxel_to_patch(vox, patch_size)
        assign_octree_patch_features(patch_feat[0], octree, v_cfg["depth_in"])
        _, mu, doctree = vae.extract_code(octree)
        return mu.cuda(), doctree

    # The noise schedule assumes ~unit-variance x0, so rescale the VAE latents.
    @torch.no_grad()
    def measure_std():
        chunks = []
        for i, (vox, _) in enumerate(loader):
            mu, _ = encode(vox.cuda().long())
            chunks.append(mu.flatten().cpu())
            if i >= 40:
                break
        return torch.cat(chunks).std().item()

    latent_scale = resolve_latent_scale(
        t_cfg.get("latent_scale", 1.0), measure_std,
        os.path.splitext(t_cfg["save_path"])[0] + "_meta.yaml", "semantic")

    ema = EMA(unet_model, decay=t_cfg["ema_decay"])
    accum = t_cfg["accum_steps"]
    total_steps = t_cfg["t_max"] * len(loader) // accum
    step = 0

    print(f"{len(loader)} samples/epoch x {t_cfg['t_max']} epochs, "
          f"accum {accum} -> {total_steps} optimiser steps")

    # 3. Training Loop
    optimizer.zero_grad()
    for epoch in range(t_cfg["t_max"]):
        unet_model.train()
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")

        for i, (vox, _) in enumerate(pbar):
            vox = vox.cuda().long()

            # Use VAE to get latents
            with torch.no_grad():
                mu, doctree = encode(vox)
                z_clean = mu * latent_scale

            # Diffusion Step
            t_cont = torch.rand(1, device=z_clean.device)
            logsnr_t = log_snr_schedule_cosine(t_cont)
            alpha_t, sigma_t = log_snr_to_alpha_sigma(logsnr_t)

            noise = torch.randn_like(z_clean)
            z_noisy = alpha_t * z_clean + sigma_t * noise

            pred_x0 = unet_model(z_noisy, doctree=doctree, timesteps=t_cont)
            loss = torch.nn.functional.mse_loss(pred_x0, z_clean)

            (loss / accum).backward()

            # The octree build forces batch 1, so accumulate to get a usable
            # effective batch before stepping.
            if (i + 1) % accum == 0:
                for g in optimizer.param_groups:
                    g["lr"] = lr_at(step, t_cfg["lr"], total_steps, t_cfg["warmup_steps"])
                torch.nn.utils.clip_grad_norm_(unet_model.parameters(), t_cfg["grad_clip"])
                optimizer.step()
                optimizer.zero_grad()
                ema.update(unet_model)
                step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        # EMA weights are the ones to sample from; raw kept for inspection.
        os.makedirs(os.path.dirname(t_cfg["save_path"]), exist_ok=True)
        torch.save(ema.state_dict(), t_cfg["save_path"])
        torch.save(unet_model.state_dict(), t_cfg["save_path"].replace(".pt", "_raw.pt"))

if __name__ == "__main__":
    train()
