import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# Model Imports
from models.structure_networks.structure_vae import VoxelVAE
from models.structure_networks.unet_3d import StructureUNet
from models.networks.diffusion_networks.graph_unet_hr import UNet3DModel
from models.networks.dualoctree_networks.graph_sem_vae import GraphVAE

# Utility Imports
from utils.config_loader import load_config
from utils.util_sample_stuff import log_snr_schedule_cosine, log_snr_to_alpha_sigma
from utils.util_octree_stuff import (
    get_non_empty_mask, voxel_grid_to_points, points2octree, 
    voxel_to_patch, assign_octree_patch_features
)
from utils.util_visual_stuff import *

@torch.no_grad()
def run_full_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. LOAD CONFIGS
    str_vae_cfg = load_config("configs/structure_vae_config.yaml")
    str_diff_cfg = load_config("configs/structure_diffusion_config.yaml")
    sem_vae_cfg = load_config("configs/sem_vae_config.yaml")["model"]
    sem_diff_cfg = load_config("configs/sem_diffusion_config.yaml")

    # 2. INITIALIZE & LOAD MODELS
    print("📦 Loading models...")
    
    # --- Structure Stage ---
    str_vae = VoxelVAE(z_channels=str_vae_cfg["model"]["z_channels"], base=str_vae_cfg["model"]["base_channels"]).to(device).eval()
    str_vae.load_state_dict(torch.load(str_vae_cfg["training"]["checkpoint_path"], map_location=device, weights_only=True))

    str_unet = StructureUNet(in_ch=str_diff_cfg["unet"]["in_ch"], base_ch=str_diff_cfg["unet"]["base_ch"],time_emb_dim=str_diff_cfg['unet']['time_emb_dim']).to(device).eval()
    str_unet.load_state_dict(torch.load(str_diff_cfg["training"]["save_path"], map_location=device, weights_only=True))

    # --- Semantic Stage ---
    sem_vae = GraphVAE(
        depth=sem_vae_cfg["depth_in"], channel_in=sem_vae_cfg["channel_in"],
        nout=sem_vae_cfg["nout"], full_depth=sem_vae_cfg["full_depth"],
        latent_dim=sem_vae_cfg["latent_dim"], num_classes=sem_vae_cfg["total_classes"],
        resblk_num=sem_vae_cfg["resblk_num"]
    ).to(device).eval()
    sem_vae.load_state_dict(torch.load(sem_diff_cfg["vae_ref"]["checkpoint"], map_location=device, weights_only=True))

    sem_unet = UNet3DModel(
        image_size=sem_diff_cfg["unet"]["image_size"], input_depth=sem_vae_cfg["depth_in"],
        full_depth=sem_vae_cfg["full_depth"], in_channels=sem_vae_cfg["latent_dim"],
        model_channels=sem_diff_cfg["unet"]["model_channels"], out_channels=sem_vae_cfg["latent_dim"],
        lr_model_channels=sem_diff_cfg["unet"]["lr_model_channels"],
        num_res_blocks=sem_diff_cfg["unet"]["num_res_blocks"], channel_mult=sem_diff_cfg["unet"]["channel_mult"]
    ).to(device).eval()
    sem_unet.load_state_dict(torch.load(sem_diff_cfg["training"]["save_path"], map_location=device, weights_only=True))

    steps = 1000
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)

    # 3. STAGE 1: GENERATE DENSE STRUCTURE
    print("🏗️  Step 1: Denoising Structure Latent...")
    z_str = torch.randn((1, str_diff_cfg["unet"]["in_ch"], 32, 32, 32), device=device)
    for i in range(steps):
        t = times[i].expand(1)
        pred_x0 = str_unet(z_str, t)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_schedule_cosine(t))
        eps = (z_str - alpha.view(-1,1,1,1,1) * pred_x0) / sigma.view(-1,1,1,1,1)
        z_str = alpha.view(-1,1,1,1,1) * pred_x0 + sigma.view(-1,1,1,1,1) * eps

    # Decode latents to dense voxel occupancy
    str_logits = str_vae.decode(z_str)
    str_logits[str_logits < 0] = 0
    str_logits[str_logits > 0] = 1
    print(str_logits.shape)      
    visualize_kitti_instance(str_logits[0][0])
    vox_occupancy = (torch.sigmoid(str_logits) > 0.5).long() # [1, 1, 32, 32, 32]

    # 4. CONVERT VOXEL TO OCTREE (Bridge between pipelines)
    print("🌿 Step 2: Converting Voxel Grid to Octree...")

    non_empty_mask = get_non_empty_mask(vox_occupancy[0][0], 1)
    print(non_empty_mask.shape)
    
    points = voxel_grid_to_points(non_empty_mask)
    # The Semantic UNet needs the 'doctree' (dual octree) structure
    octree = points2octree(points, depth=sem_vae_cfg["depth_in"], full_depth=sem_vae_cfg["full_depth"]).to(device)
    print(octree.nnum)
    # We need to extract the structure-specific doctree using the VAE
    # This prepares the graph nodes for the Semantic UNet
    octree = sem_vae.create_child_octree(octree)
    print(octree.nnum)
    _, _, doctree = sem_vae.extract_code(octree)

    # 5. STAGE 2: GENERATE SEMANTICS ON OCTREE
    print("🎨 Step 3: Denoising Semantic Graph...")
    z_sem = torch.randn((1, sem_vae_cfg["latent_dim"]), device=device) # Shapes vary by node count
    # Note: In Octree Diffusion, z_sem size corresponds to the number of nodes in 'doctree'
    z_sem = torch.randn((doctree[sem_vae_cfg["depth_in"]].npt, sem_vae_cfg["latent_dim"]), device=device)

    for i in range(steps):
        t = times[i].expand(1)
        # Note: Semantic UNet uses the 'doctree' as the spatial skeleton
        pred_x0_sem = sem_unet(z_sem, doctree=doctree, timesteps=t)
        
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_schedule_cosine(t))
        eps = (z_sem - alpha * pred_x0_sem) / sigma
        z_sem = alpha * pred_x0_sem + sigma * eps

    # 6. DECODE OCTREE TO SEMANTIC VOXELS
    print("💾 Step 4: Final Decoding...")
    # This part depends on your GraphVAE's decoding implementation
    # Usually, it maps graph node latents back to voxel patches
    output = sem_vae.decode_code(z_sem, doctree) 
    # 'sem_voxs' contains the final labels
    final_labels = torch.argmax(output['sem_voxs'], dim=1) 

    print("✅ Full Inference Complete!")
    # Visualization logic here (Matplotlib or saving .npy)

if __name__ == "__main__":
    run_full_inference()