"""Single source of truth for building the four models from config.

Every model was previously constructed inline at each call site, and the sites had
drifted: `train_sem_diffusion` omitted `depth_stop`/`depth_out`, silently taking
GraphVAE's defaults (6/8) where training used 6/6, and `train_structure_diffusion`
omitted `in_ch` on VoxelVAE. Both happened to be harmless -- the state dicts came
out identical -- but only by luck, and nothing would have caught it if they had not.

Build models here so a config change reaches training and inference together.
"""

from octree_diff.models.semantic.graph_sem_vae import GraphVAE
from octree_diff.models.semantic.graph_unet_hr import UNet3DModel
from octree_diff.models.structure.structure_vae import VoxelVAE
from octree_diff.models.structure.unet_3d import StructureUNet


def build_structure_vae(m, device=None):
    """m: the `model:` block of structure_vae_config.yaml"""
    vae = VoxelVAE(z_channels=m["z_channels"], base=m["base_channels"],
                   in_ch=m["in_channels"])
    return vae.to(device) if device else vae


def build_structure_unet(u, device=None):
    """u: the `unet:` block of structure_diffusion_config.yaml"""
    net = StructureUNet(in_ch=u["in_channels"], base_ch=u["base_ch"],
                        time_emb_dim=u["time_emb_dim"])
    return net.to(device) if device else net


def build_semantic_vae(m, device=None):
    """m: the `model:` block of sem_vae_config.yaml

    `depth_out` is deliberately `depth_in`, not GraphVAE's default of 8 -- the
    shipped weights were trained with the decoder stopping at the input depth.
    """
    vae = GraphVAE(
        depth=m["depth_in"],
        channel_in=m["in_channels"],
        nout=m["nout"],
        full_depth=m["full_depth"],
        depth_stop=m["depth_stop"],
        depth_out=m["depth_in"],
        latent_dim=m["latent_dim"],
        num_classes=m["total_classes"],
        resblk_num=m["resblk_num"],
    )
    return vae.to(device) if device else vae


def build_semantic_unet(m, u, device=None):
    """m: `model:` block of sem_vae_config.yaml; u: `unet:` block of sem_diffusion_config.yaml

    Width is taken from the VAE's latent_dim on both ends, so the UNet cannot be
    configured to a size the VAE does not produce.
    """
    net = UNet3DModel(
        image_size=u["image_size"],
        input_depth=m["depth_in"],
        full_depth=m["full_depth"],
        in_channels=m["latent_dim"],
        out_channels=m["latent_dim"],
        model_channels=u["model_channels"],
        lr_model_channels=u["lr_model_channels"],
        num_res_blocks=u["num_res_blocks"],
        channel_mult=u["channel_mult"],
    )
    return net.to(device) if device else net
