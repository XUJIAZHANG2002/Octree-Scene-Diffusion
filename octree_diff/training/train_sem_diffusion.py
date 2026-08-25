import torch
import tqdm
import os
from octree_diff.octree.build import build_semantic_octree
from octree_diff.models.factory import build_semantic_vae, build_semantic_unet
from octree_diff.diffusion.util_sample_stuff import log_snr_schedule_cosine, log_snr_to_alpha_sigma
from octree_diff.data.voxel_dataset import get_dataloader
from octree_diff.training.ema import EMA, lr_at
from octree_diff.training.latent_scale import resolve_latent_scale
from octree_diff.config import load_stage

def train(config_dir=None, device=None):
    # Load both configs
    d_cfg, full_v_cfg = load_stage("sem_diff", config_dir)
    v_cfg = full_v_cfg["model"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t_cfg = d_cfg["training"]

    # 1. Load Pre-trained VAE (Eval Mode)
    vae = build_semantic_vae(v_cfg, device)
    vae.load_state_dict(torch.load(d_cfg["vae_ref"]["checkpoint"]))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # 2. Initialize UNet Diffusion Model
    unet_model = build_semantic_unet(v_cfg, d_cfg["unet"], device)

    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=t_cfg["lr"])
    loader = get_dataloader(t_cfg["data_dir"], batch_size=1)
    patch_size = v_cfg["patch_size"]

    def encode(vox):
        """Voxel labels -> (octree latents, dual octree)."""
        octree = build_semantic_octree(vox, patch_size, v_cfg["depth_in"],
                                       v_cfg["full_depth"], device)
        _, mu, doctree = vae.extract_code(octree)
        return mu.to(device), doctree

    # The noise schedule assumes ~unit-variance x0, so rescale the VAE latents.
    @torch.no_grad()
    def measure_std():
        chunks = []
        for i, (vox, _) in enumerate(loader):
            mu, _ = encode(vox.to(device).long())
            chunks.append(mu.flatten().cpu())
            if i >= 40:
                break
        return torch.cat(chunks).std().item()

    latent_scale = resolve_latent_scale(
        t_cfg.get("latent_scale", 1.0), measure_std,
        os.path.splitext(t_cfg["checkpoint_path"])[0] + "_meta.yaml", "semantic")

    ema = EMA(unet_model, decay=t_cfg["ema_decay"])
    accum = t_cfg["accum_steps"]
    total_steps = t_cfg["epochs"] * len(loader) // accum
    step = 0

    print(f"{len(loader)} samples/epoch x {t_cfg['epochs']} epochs, "
          f"accum {accum} -> {total_steps} optimiser steps")

    # 3. Training Loop
    optimizer.zero_grad()
    for epoch in range(t_cfg["epochs"]):
        unet_model.train()
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")

        for i, (vox, _) in enumerate(pbar):
            vox = vox.to(device).long()

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
        os.makedirs(os.path.dirname(t_cfg["checkpoint_path"]), exist_ok=True)
        torch.save(ema.state_dict(), t_cfg["checkpoint_path"])
        torch.save(unet_model.state_dict(), t_cfg["checkpoint_path"].replace(".pt", "_raw.pt"))

if __name__ == "__main__":
    train()
