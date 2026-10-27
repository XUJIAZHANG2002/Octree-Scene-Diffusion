"""Export SemanticKITTI completions to the web viewer's binary format.

The outdoor counterpart to export_web_scenes.py. Same encoding, same manifest
shape, so the viewer needs no changes -- it reads dims, palette, class names and
the variant list out of the manifest.

One difference from indoor, forced by the data:

**The observation is the scan itself**, not a mask over a complete scene. A
single sweep fills about 0.6% of the volume. The known region is recovered by
tracing back from each hit to the sensor origin, marking everything on the way
as observed-empty -- so "unknown" means genuinely occluded, not just empty.

The `truth` panel comes from SemanticKITTI's own SSC voxel labels under
`<data>/dataset/sequences/*/voxels/`. They are co-aligned with the voxelisation
here, not merely assumed to be: on 00:000202 and 08:000100, 99% of the scan's
occupied voxels fall on voxels the ground truth also calls occupied, with 98%
label agreement, and shifting the ground truth by two voxels along any axis
drops that to 0.20-0.85. Voxels the `.invalid` mask marks as outside the
sensor's traversal have no defined label -- a mean of 29% of each volume, from 4%
to 62% -- and are rendered as empty, so the truth panel is itself sparse in
exactly the regions the completion has most to say about.

Needs the `octfusion` environment: open3d for voxelisation, numba for the trace.

    python scripts/export_web_scenes_kitti.py --frames 00:000202 00:001100 \
        --data /path/to/dataset --weights /path/to/weights --out out/web_kitti
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_web_scenes import check_roundtrip, encode  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = (256, 256, 32)
PATCH = 4
DEPTH = 6
FULL_DEPTH = 3
LATENT_DIM = 8
SENSOR_ORIGIN = (0, 128, 9)   # the Velodyne sits at the near edge, mid-width


# ---------------------------------------------------------------- sensor trace
# Lifted from notebooks/kitti/01_completion_ddim_logsnr.ipynb. Kept here rather
# than in the package so numba stays out of octree_diff's dependency surface --
# this is export tooling, not library code.

from numba import njit, prange  # noqa: E402


@njit(cache=True)
def _dda_trace_mark_free(occ, ox, oy, oz, hx, hy, hz):
    """March from a hit voxel back to the origin, turning unknown (-1) into free (0)."""
    x, y, z = hx, hy, hz
    if x == ox and y == oy and z == oz:
        return
    dx, dy, dz = ox - x, oy - y, oz - z
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    step_z = 0 if dz == 0 else (1 if dz > 0 else -1)
    inv_x = 1.0 / abs(float(dx)) if dx != 0 else 1e30
    inv_y = 1.0 / abs(float(dy)) if dy != 0 else 1e30
    inv_z = 1.0 / abs(float(dz)) if dz != 0 else 1e30
    tx = 0.5 * inv_x if dx != 0 else 1e30
    ty = 0.5 * inv_y if dy != 0 else 1e30
    tz = 0.5 * inv_z if dz != 0 else 1e30
    H, W, D = occ.shape
    while not (x == ox and y == oy and z == oz):
        if 0 <= x < H and 0 <= y < W and 0 <= z < D:
            if occ[x, y, z] == -1:
                occ[x, y, z] = 0
            # No early exit on an already-carved voxel. It looks safe -- another
            # ray has been here, so the rest of the path is done -- but that
            # only holds if the other ray continued along *this* ray's remaining
            # path, and rays from the same origin in different directions do
            # not. It under-carved, and because `_carve` runs the rays in
            # `prange`, which ray reached a voxel first depended on thread
            # scheduling, so the whole observation was nondeterministic:
            # 35,244 voxels differed between two calls in one process. Writing
            # only -1 -> 0 and never reading back makes the pass idempotent and
            # order-independent, so it is now reproducible as well as correct.
        else:
            break
        if tx <= ty and tx <= tz:
            x += step_x; tx += inv_x
        elif ty <= tx and ty <= tz:
            y += step_y; ty += inv_y
        else:
            z += step_z; tz += inv_z


@njit(parallel=True, cache=True)
def _carve(occ, origin, hits):
    ox, oy, oz = origin
    for i in prange(hits.shape[0]):
        _dda_trace_mark_free(occ, ox, oy, oz, hits[i, 0], hits[i, 1], hits[i, 2])


@njit(cache=True)
def _fill_between_ones_1d(arr):
    n = arr.shape[0]
    i = 0
    while i < n:
        while i < n and arr[i] != 1:
            i += 1
        if i >= n:
            break
        j = i + 1
        while j < n and arr[j] != 1:
            j += 1
        if j < n:
            for k in range(i + 1, j):
                if arr[k] == 0:
                    arr[k] = 1
        i = j


@njit(parallel=True, cache=True)
def _ground_fill(occ, z_levels):
    H, W, D = occ.shape
    for zi in range(z_levels.shape[0]):
        z = z_levels[zi]
        if z < 0 or z >= D:
            continue
        for y in prange(W):
            _fill_between_ones_1d(occ[:, y, z])
        for x in prange(H):
            _fill_between_ones_1d(occ[x, :, z])


def observed_state(labels, origin=SENSOR_ORIGIN, z_levels=(0, 1, 2, 3)):
    """Tri-state occupancy: -1 unknown (occluded), 0 observed empty, 1 observed hit."""
    occ = np.full(labels.shape, -1, dtype=np.int8)
    hits = np.argwhere(labels > 0).astype(np.int32)
    if hits.size:
        occ[hits[:, 0], hits[:, 1], hits[:, 2]] = 1
        d2 = ((hits - np.array(origin, np.int32)) ** 2).sum(1)
        _carve(occ, np.array(origin, np.int32), hits[np.argsort(-d2)].astype(np.int32))
    _ground_fill(occ, np.array(z_levels, np.int32))
    return occ


def reduce_patches_tri_state(occ, patch=PATCH):
    """(H,W,D) tri-state -> (H/P,W/P,D): any hit wins, else all-free is free, else unknown."""
    H, W, D = occ.shape
    g = occ.reshape(H // patch, patch, W // patch, patch, D).transpose(0, 2, 4, 1, 3)
    g = g.reshape(H // patch, W // patch, D, patch * patch)
    out = np.full(g.shape[:3], -1, np.int8)
    out[(g == 1).any(-1)] = 1
    out[np.logical_and((g == 0).all(-1), ~(g == 1).any(-1))] = 0
    return out


# --------------------------------------------------------------------- models

def _struct(weights, device):
    """The outdoor structure UNet, unchanged -- it was not retrained."""
    from octree_diff.models.outdoor.graph_unet_lr import UNet3DModel as StructUNet

    struct = StructUNet(full_depth=4, in_split_channels=8, model_channels=32,
                        out_split_channels=8, attention_resolutions=[2, 4, 8],
                        dims=3, num_heads=4).to(device)
    struct.load_state_dict(torch.load(os.path.join(weights, "structure_model.pt"),
                                      map_location=device))
    struct.eval()
    return struct


class DDPMStructure:
    """The 1000-step structure model, which samples the split grid differently.

    `SemCityOcto/checkpoints/structure_ddpm_ep110.pt`: 62.8M parameters, an
    epsilon-predictor on a linear beta schedule over a one-channel `[32,32,16]`
    split-big grid. It is the structure model the paper's outdoor FID notebook
    sampled from; `_struct`'s 15.7M DDIM model is the one notebook 01 used for
    completion. EMA weights are loaded when the checkpoint carries them.
    """

    def __init__(self, weights, device, ckpt="checkpoints/structure_ddpm_ep110.pt"):
        from octree_diff.diffusion.ddpm import ddpm_schedule
        from octree_diff.models.outdoor.graph_unet_lr import UNet3DModel as StructUNet

        self.model = StructUNet(full_depth=4, in_split_channels=1, model_channels=64,
                                out_split_channels=1, channel_mult=(1, 2, 4, 8),
                                attention_resolutions=[2, 4, 8], dims=3, num_heads=4,
                                timestep_repr="ddpm", T=1000).to(device)
        state = torch.load(os.path.join(weights, ckpt), map_location=device, weights_only=True)
        self.model.load_state_dict(state["model"])
        if "ema_shadow" in state:
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for k, v in state["ema_shadow"].items():
                    params[k].copy_(v)
        self.model.eval()
        self.ckpt = ckpt
        self.sched = ddpm_schedule(1000, device)
        self.resample = 1           # RePaint repeats per step when inpainting

    def sample_split(self, device):
        """An unconditional draw as the `[1,8,16,16,8]` split the DDIM model emits."""
        from octree_diff.diffusion.ddpm import sample_ddpm
        from octree_diff.octree.util_octree_stuff import splitbig2split

        big = torch.full((1, 1, 32, 32, 32), -1.0, device=device)
        big[..., 0:16] = sample_ddpm(self.model, (1, 1, 32, 32, 16), self.sched, device)
        return splitbig2split(big)[..., 0:8]

    @staticmethod
    def _to_big(split8, fill):
        """[1,8,16,16,8] split (or mask) -> the model's [1,1,32,32,16] layout."""
        from octree_diff.octree.util_octree_stuff import split2splitbig
        s = torch.full((1, 8, 16, 16, 16), float(fill), device=split8.device)
        s[..., 0:8] = split8
        return split2splitbig(s)[..., 0:16]

    def inpaint_split(self, z_known, mask, device, resample=1):
        """Complete the split where `mask` is 0, holding it to `z_known` where 1."""
        from octree_diff.diffusion.ddpm import inpaint_ddpm
        from octree_diff.octree.util_octree_stuff import splitbig2split

        big = torch.full((1, 1, 32, 32, 32), -1.0, device=device)
        big[..., 0:16] = inpaint_ddpm(self.model, self._to_big(z_known.to(device), -1.0),
                                      self._to_big(mask.to(device), 0.0), self.sched,
                                      device, resample=resample)
        return splitbig2split(big)[..., 0:8]


def load_structure(weights, device, kind="ddim"):
    return DDPMStructure(weights, device) if kind == "ddpm" else _struct(weights, device)


def load_models(weights, device):
    """The shipped outdoor semantic models, at the geometry they were trained at.

    Trained before latent scaling existed, so their latent scale is 1.0.
    """
    from octree_diff.models.semantic.graph_sem_vae import GraphVAE
    from octree_diff.models.semantic.graph_unet_hr import UNet3DModel as SemUNet

    vae = GraphVAE(depth=DEPTH, channel_in=32, nout=2, full_depth=FULL_DEPTH,
                   depth_stop=DEPTH, depth_out=DEPTH, latent_dim=LATENT_DIM,
                   num_classes=21, resblk_num=1,
                   patch_size=PATCH).to(device)   # patch 2 will not load these weights
    vae.load_state_dict(torch.load(os.path.join(weights, "vae_fdepth6_ldim8_8.pt"),
                                   map_location=device))
    vae.eval()

    sem = SemUNet(image_size=16, input_depth=DEPTH, full_depth=FULL_DEPTH,
                  in_channels=LATENT_DIM, model_channels=64, lr_model_channels=128,
                  out_channels=LATENT_DIM, num_res_blocks=[2, 2, 2], dropout=0.1,
                  channel_mult=[2, 4], use_checkpoint=False).to(device)
    sem.load_state_dict(torch.load(os.path.join(weights, "Model_is_good_8_6.pt"),
                                   map_location=device))
    sem.eval()

    return vae, sem, _struct(weights, device), LATENT_DIM, 1.0


def load_models_from_config(config_dir, weights, device, structure="ddim"):
    """The retrained outdoor semantic models, built from a config directory.

    Returns the latent dimension and scale alongside, because both differ from
    the shipped pair: latent_dim 32 rather than 8, and a latent scale that is
    emphatically not 1.0. The diffusion model was trained on latents multiplied
    by it, so a sample must be divided by it again before the VAE decodes --
    skipping that step hands the decoder latents a third too large.
    """
    from octree_diff.config import load_config
    from octree_diff.models.factory import build_semantic_unet, build_semantic_vae
    from octree_diff.training.latent_scale import load_latent_scale

    v_cfg = load_config(os.path.join(config_dir, "sem_vae_config.yaml"))
    d_cfg = load_config(os.path.join(config_dir, "sem_diffusion_config.yaml"))
    m = v_cfg["model"]
    if (m["patch_size"], m["depth_in"], m["full_depth"]) != (PATCH, DEPTH, FULL_DEPTH):
        raise ValueError(
            f"config geometry ({m['patch_size']}, {m['depth_in']}, {m['full_depth']}) "
            f"differs from this exporter's ({PATCH}, {DEPTH}, {FULL_DEPTH}); the "
            f"voxel<->octree helpers here close over the module-level constants")

    vae = build_semantic_vae(m, device)
    vae.load_state_dict(torch.load(os.path.join(REPO, v_cfg["training"]["checkpoint_path"]),
                                   map_location=device, weights_only=True))
    vae.eval()

    sem = build_semantic_unet(m, d_cfg["unet"], device)
    ckpt = os.path.join(REPO, d_cfg["training"]["checkpoint_path"])
    sem.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    sem.eval()

    scale = load_latent_scale(os.path.splitext(ckpt)[0] + "_meta.yaml")
    return vae, sem, load_structure(weights, device, structure), m["latent_dim"], scale


def _to_split(vol):
    """Dense (64,64,64) occupancy -> the [1,8,16,16,8] split grid the UNet eats."""
    from octree_diff.octree.util_dualoctree import octree2split_small
    from octree_diff.octree.util_octree_stuff import points2octree, voxel_grid_to_points
    octree = points2octree(voxel_grid_to_points(vol), depth=DEPTH, full_depth=FULL_DEPTH)
    return octree2split_small(octree, 4)[:, :, :, :, 0:8]


def split_to_octree(split):
    """[1,8,16,16,16] split grid -> a depth-6 octree matching the training graph.

    The outdoor twin of `pipeline.split_big_to_octree`. `split2octree_small`
    builds on a grid that is full at depth 4 (4096 nodes), and the notebooks'
    `create_child_octree` keeps that dense layer; the semantic models were
    trained on `points2octree(..., full_depth=3)` octrees, which hold ~1,200-1,400
    depth-4 nodes on a KITTI frame. That added ~2,500 dual-graph nodes (+7-10%)
    the UNet never saw. Rebuilding from the depth-6 node set reproduces the
    training graph; depths 5 and 6 are unchanged, which is asserted.
    """
    from octree_diff.octree.util_dualoctree import split2octree_small
    from octree_diff.octree.util_octree_stuff import points2octree, voxel_grid_to_points

    octree_s = split2octree_small(split, DEPTH, 4)
    x, y, z, _ = octree_s.xyzb(DEPTH, nempty=False)
    res = 2 ** DEPTH
    mask = torch.zeros((res, res, res))
    mask[x.long(), y.long(), z.long()] = 1
    octree = points2octree(voxel_grid_to_points(mask), depth=DEPTH, full_depth=FULL_DEPTH)
    assert (octree.nnum[DEPTH - 1:DEPTH + 1] == octree_s.nnum[DEPTH - 1:DEPTH + 1]).all()
    return octree


def conditioning(occ):
    """Tri-state occupancy -> (known latent, mask) at split resolution.

    mask is 1 where the observation is known. Verified end to end: with the mask
    fully on the sampler reproduces `z_known` exactly.
    """
    from octree_diff.octree.util_octree_stuff import get_non_empty_mask

    known = torch.from_numpy(occ != -1)
    padded_mask = torch.zeros((64, 64, 64))
    padded_mask[:, :, :GRID[2]] = get_non_empty_mask(known, PATCH)

    padded_z = torch.zeros((64, 64, 64))
    padded_z[:, :, :GRID[2]] = torch.from_numpy(
        reduce_patches_tri_state(occ).astype(np.float32))
    padded_z[padded_z == -1] = 0                      # unknown draws as empty

    mask = _to_split(padded_mask)
    mask[mask == -1] = 0
    return _to_split(padded_z), mask


def guidance_targets(labels, known, octree, device):
    """Per-node (label, observed) patches in the octree's node order.

    `reconstruct_voxel_from_patch` reads the decoder's logits with
    `octree.xyzb(DEPTH, nempty=False)`, so indexing the patched volumes the same
    way puts the targets in the decoder's own row order -- no voxel-space
    round trip, and nothing to misalign.
    """
    from octree_diff.octree.util_octree_stuff import voxel_to_patch

    side = 64 * PATCH
    pad_l = torch.zeros((1, side, side, 64), dtype=torch.long)
    pad_k = torch.zeros((1, side, side, 64), dtype=torch.long)
    pad_l[0, :GRID[0], :GRID[1], :GRID[2]] = torch.from_numpy(labels).long()
    pad_k[0, :GRID[0], :GRID[1], :GRID[2]] = torch.from_numpy(known).long()

    tl = voxel_to_patch(pad_l, PATCH)[0].to(device)            # [64,64,64,P,P]
    tk = voxel_to_patch(pad_k, PATCH)[0].to(device).bool()

    x, y, z, _ = octree.xyzb(DEPTH, nempty=False)
    x, y, z = x.long(), y.long(), z.long()
    return tl[x, y, z], tk[x, y, z]                            # [N,P,P] each


def guidance_weights(target, target_mask, free_weight):
    """Per-voxel loss weights, holding down the observed-free voxels.

    Ray traversal clears far more volume than it hits -- 4.2 observed-free
    voxels per hit inside the completed octree on 06:000930 -- so an unweighted
    mean makes the objective 81% "be empty". The structure stage has already
    pinned occupancy from the same observation, exactly and at every hit, so
    that pressure is largely redundant here; what guidance uniquely supplies is
    the *labels*. `free_weight=0` asks only for labels.
    """
    w = target_mask.float()
    if free_weight != 1.0:
        w = torch.where(target == 0, w * float(free_weight), w)
    return w


@torch.no_grad()
def synthesize(z_known, mask, vae, sem, struct, device, steps, resample,
               latent_dim=LATENT_DIM, latent_scale=1.0, guidance=None,
               sampler="legacy", verbose=False):
    """Structure diffusion (masked where `mask` is set) then latent semantics.

    `guidance` is a dict of `labels`, `known` and the sampler's guidance
    settings; it completes the *semantics* too, by pushing the decoded labels
    toward the observation. That is mode 4. Without it the semantic stage is
    sampled from the prior, which completes geometry only -- and because label 0
    is free, also deletes observed voxels the structure stage kept. See
    `docs/outdoor-semantic-conditioning.md`.
    """
    from octree_diff.diffusion.sampling import ddim_guided_graph
    from octree_diff.diffusion.util_sample_stuff import (
        ddim_sample_logsnr_cosine, ddim_sample_structure_logsnr_cosine_inpaint)
    from octree_diff.models.semantic.dual_octree import DualOctree
    from octree_diff.octree.util_octree_stuff import reconstruct_voxel_from_patch

    if guidance is not None or sampler != "legacy":
        # Guidance backpropagates through the decoder; freeze it so the pass
        # only reaches the latent and no parameter grads are accumulated.
        for prm in vae.parameters():
            prm.requires_grad_(False)

    if isinstance(struct, DDPMStructure):
        # `resample` is the DDIM sampler's per-step RePaint count; the DDPM
        # sampler takes 1000 steps rather than 50, so it gets its own
        # (`struct.resample`), set from --ddpm-resample.
        sample = (struct.inpaint_split(z_known, mask, device, struct.resample)
                  if mask.any() else struct.sample_split(device))
    else:
        z_T = torch.randn(1, 8, 16, 16, 8, device=device)
        sample = ddim_sample_structure_logsnr_cosine_inpaint(
            z_T, z_known.to(device), struct, mask.to(device), S=steps, jump_n_sample=resample)
    sample = torch.where(sample > 0, 1.0, -1.0)

    split = torch.ones((1, 8, 16, 16, 16)) * -1
    split[:, :, :, :, 0:8] = sample.cpu()
    octree = split_to_octree(split).to(device)

    doctree = DualOctree(octree)
    doctree.post_processing_for_docnn()
    z_init = torch.randn(doctree.total_num, latent_dim).to(device)

    def decode(code):
        # Training multiplied the VAE's latents by this; undo it before decoding.
        return vae.decode_code(code / latent_scale, doctree,
                               update_octree=False, pos=None)

    if guidance is None and sampler == "legacy":
        z = ddim_sample_logsnr_cosine(z_init, model=sem, doctree=doctree, S=steps)
    else:
        g = dict(labels=np.zeros(GRID, np.int32), known=np.zeros(GRID, bool),
                 weight=0.0, inner=1, free_weight=1.0, stop_t=0.0, ramp=0.0,
                 max_t=1.0)
        g.update(guidance or {})
        target, target_mask = guidance_targets(g["labels"], g["known"],
                                               octree.to(device), device)
        voxel_weight = guidance_weights(target, target_mask, g["free_weight"])
        z = ddim_guided_graph(sem, doctree,
                              lambda code: decode(code)["sem_voxs"][DEPTH],
                              target, voxel_weight, z_T=z_init, steps=steps,
                              device=device, weight=g["weight"], inner=g["inner"],
                              stop_t=g["stop_t"], ramp=g["ramp"],
                              max_t=g["max_t"], verbose=verbose)

    out = decode(z)
    vox = reconstruct_voxel_from_patch(out["sem_voxs"], octree.to(device),
                                       depth=DEPTH, shape=(1,) + GRID, patch_size=PATCH)
    return vox[0].detach().cpu().numpy().astype(np.int32)


def dense_conditioning(labels, keep_frac):
    """Known everywhere in the first `keep_frac` of x, unknown beyond -- outpainting.

    Roughly the 0.75 known / 0.25 generated split the paper extends scenes with.
    """
    occ = np.full(labels.shape, -1, dtype=np.int8)
    cut = int(round(labels.shape[0] * keep_frac))
    occ[:cut] = 0
    occ[:cut][labels[:cut] > 0] = 1
    return occ


# --------------------------------------------------------------------- driver

def guidance_opts(args):
    """The guidance settings, in one place so the manifest cannot drift."""
    return dict(weight=args.guidance_weight, inner=args.guidance_inner,
                free_weight=args.guidance_free_weight,
                stop_t=args.guidance_stop_t, ramp=args.guidance_ramp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="complete-geometry",
                    help="one of the five execution models, by name or number; "
                         "pass --mode list to print the table")
    ap.add_argument("--data", default=os.path.join(REPO, "data", "semantickitti"),
                    help="SemanticKITTI root, the folder holding dataset/sequences")
    ap.add_argument("--weights", default=os.path.join(REPO, "saved_model", "semcity"),
                    help="outdoor structure checkpoints: structure_model.pt and "
                         "checkpoints/structure_ddpm_ep110.pt")
    ap.add_argument("--frames", nargs="+", default=["00:000202"],
                    help="sequence:scan pairs, e.g. 00:000202")
    ap.add_argument("--samples", type=int, default=4, help="generate: how many")
    ap.add_argument("--keep", type=float, default=0.75,
                    help="extend: fraction of x kept as known")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "web_kitti"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--resample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance-weight", dest="guidance_weight", type=float,
                    default=0.15,
                    help="mode 4/5: fraction of the latent's norm moved toward "
                         "the observation at each step. 0 disables guidance, "
                         "which makes mode 4 identical to mode 3")
    ap.add_argument("--guidance-inner", dest="guidance_inner", type=int, default=3,
                    help="guidance steps per DDIM step; costs one decoder pass "
                         "each, no extra UNet evaluations")
    ap.add_argument("--guidance-free-weight", dest="guidance_free_weight",
                    type=float, default=0.0,
                    help="weight on observed-FREE voxels relative to hits. The "
                         "structure stage already pins occupancy, so the default "
                         "0 asks guidance for labels only; 1 is unweighted. Any "
                         "value above ~0 reintroduces label speckle -- measured, "
                         "0.02 is already most of the way to 1.0")
    ap.add_argument("--guidance-stop-t", dest="guidance_stop_t", type=float,
                    default=0.20,
                    help="stop guiding below this diffusion time, leaving the "
                         "last steps unguided. A LiDAR sweep's returns lie on "
                         "rings, so guidance pins labels on the rings while the "
                         "voxels between them follow the prior, and the "
                         "disagreement shows as concentric striping. The unguided "
                         "tail resolves across it. 0 guides to t=0 and stripes")
    ap.add_argument("--guidance-ramp", dest="guidance_ramp", type=float,
                    default=0.15, help="fade guidance in over this much t above "
                                       "--guidance-stop-t")
    ap.add_argument("--structure", default="ddpm", choices=["ddim", "ddpm"],
                    help="'ddim': structure_model.pt, 50 DDIM steps, the model "
                         "notebook 01 completes with. 'ddpm': the 1000-step "
                         "structure_ddpm_ep110.pt the paper's FID notebook samples "
                         "from; inpaints RePaint-style for modes 3-5")
    ap.add_argument("--ddpm-resample", dest="ddpm_resample", type=int, default=3,
                    help="--structure ddpm only: RePaint repeats per step when "
                         "inpainting; each one costs another 1000 model passes")
    ap.add_argument("--semantic-sampler", dest="semantic_sampler",
                    default="legacy", choices=["legacy", "guided"],
                    help="'legacy' keeps util_sample_stuff's stochastic sampler "
                         "for the unconditioned modes; 'guided' uses the DDIM "
                         "sampler on the schedule the model was trained with")
    ap.add_argument("--device", default=None)
    ap.add_argument("--config-dir", dest="config_dir", default=None,
                    help="e.g. configs/v2_outdoor for the retrained semantic "
                         "models; without it the shipped weights are used")
    args = ap.parse_args(argv)

    from octree_diff.inference import modes
    if str(args.mode).lower() == "list":
        print("outdoor execution models:\n" + modes.describe("outdoor"))
        return
    try:
        args.mode = modes.resolve(args.mode, "outdoor")
    except ValueError as e:
        sys.exit(str(e))
    modes.announce("outdoor", args.mode)

    from octree_diff.data.kitti.ssc_dataset import _unpack_bits
    from octree_diff.data.kitti.velodyne_to_voxel import velodene_to_voxel

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    km = yaml.safe_load(open(os.path.join(REPO, "octree_diff", "data", "kitti",
                                          "semantic-kitti.yaml")))
    inv, names, colors = km["learning_map_inv"], km["labels"], km["color_map"]
    n_classes = max(inv) + 1
    class_names = ["free"] * n_classes
    palette = [[255, 255, 255]] * n_classes
    for learn_id, raw in inv.items():
        if learn_id == 0:
            continue
        class_names[learn_id] = names[raw]
        palette[learn_id] = list(colors[raw][::-1])   # the yaml stores BGR

    # raw SemanticKITTI ids -> the 0..19 learning space, for the SSC voxel labels
    lm = km["learning_map"]
    remap = np.zeros(max(lm) + 1, dtype=np.int64)
    for raw, learn in lm.items():
        remap[raw] = learn

    print("loading models...")
    if args.config_dir:
        vae, sem, struct, latent_dim, latent_scale = load_models_from_config(
            args.config_dir, args.weights, device, args.structure)
        if isinstance(struct, DDPMStructure):
            struct.resample = args.ddpm_resample
        print(f"  retrained from {args.config_dir}: latent_dim {latent_dim}, "
              f"latent scale {latent_scale:.4f}")
    else:
        vae, sem, struct, latent_dim, latent_scale = load_models(args.weights, device)
        print(f"  shipped weights: latent_dim {latent_dim}, latent scale {latent_scale:.4f}")

    def load_frame(spec):
        seq, scan = spec.split(":")
        vol = np.asarray(velodene_to_voxel(args.data, seq, scan)).astype(np.int32)
        vol[vol == 255] = 0                 # 255 marks invalid, not a class
        return seq, scan, vol

    def load_truth(spec):
        """SSC voxel ground truth for one frame, invalid voxels zeroed.

        Same learning space as the voxelised scan (0 free, 1..19 classes), so it
        encodes with the same palette. `.invalid` voxels have no defined label
        and become 0 rather than being drawn; the fraction is returned so the
        manifest can record how much of the panel that removes.
        """
        seq, scan = spec.split(":")
        d = os.path.join(args.data, "dataset", "sequences", seq, "voxels")
        raw = np.fromfile(os.path.join(d, f"{scan}.label"), dtype=np.uint16)
        vol = remap[raw].reshape(GRID).astype(np.int32)
        inv = _unpack_bits(np.fromfile(os.path.join(d, f"{scan}.invalid"),
                                       dtype=np.uint8)).reshape(GRID) == 1
        vol[inv] = 0
        return vol, float(inv.mean())

    def emit(sid, name, vol):
        buf = encode(vol)
        vis, _ = check_roundtrip(vol, buf)
        fn = f"{sid}_{name}.bin"
        buf.tofile(os.path.join(args.out, fn))
        print(f"  {name:10s} {vis:7,} visible  {len(buf) * 4 / 1024:6.0f} KB")
        return {"file": fn, "count": int(len(buf))}

    scenes, variants = [], None

    if args.mode == modes.GENERATE:
        variants = [{"key": "sample", "label": "Generated",
                     "hint": "sampled from noise, nothing observed"}]
        zeros = torch.zeros(1, 8, 16, 16, 8)
        for i in range(args.samples):
            torch.manual_seed(args.seed + i)
            print(f"sample_{i:02d}: generating...")
            vol = synthesize(zeros, zeros, vae, sem, struct, device,
                             args.steps, 1, latent_dim, latent_scale)
            scenes.append({
                "id": f"sample_{i:02d}", "scene": f"sample {i}",
                "occupied_voxels": int((vol > 0).sum()),
                "classes": int(len(np.unique(vol[vol > 0]))),
                "variants": {"sample": emit(f"sample_{i:02d}", "sample", vol)},
            })

    elif args.mode == modes.EXTEND:
        variants = [{"key": "input", "label": "Known region",
                     "hint": f"the first {args.keep:.0%} of the scene, given as ground truth"},
                    {"key": "extended", "label": "Extended",
                     "hint": "the rest generated by outpainting"}]
        for spec in args.frames:
            seq, scan = spec.split(":")
            sid = f"{seq}_{scan}"

            # Mode 5 extends a full ground-truth map: SemanticKITTI's dense SSC
            # labels for the frame, `.invalid` voxels empty. (This used to
            # complete a single scan into a canvas first, with the semantic
            # stage unconditioned -- a model output, not ground truth.)
            canvas, inv_frac = load_truth(spec)
            cut = int(round(canvas.shape[0] * args.keep))
            print(f"{spec}: ground truth ({inv_frac:.0%} invalid); "
                  f"outpainting the last {1 - args.keep:.0%} of x...")
            torch.manual_seed(args.seed + 1)
            occ_ext = dense_conditioning(canvas, args.keep)
            z_known, mask = conditioning(occ_ext)
            # The kept half is observed, labels and all, so the same guidance
            # that completes a scan's semantics also stops the extension from
            # relabelling what it was told to keep.
            grown = synthesize(z_known, mask, vae, sem, struct, device,
                               args.steps, args.resample, latent_dim, latent_scale,
                               guidance=(dict(labels=canvas, known=occ_ext != -1,
                                              **guidance_opts(args))
                                         if args.guidance_weight > 0 else None),
                               sampler=args.semantic_sampler)
            if args.guidance_weight > 0:
                kept_m = (np.arange(canvas.shape[0])[:, None, None] < cut) & (canvas > 0)
                print(f"  of the kept half's {int(kept_m.sum()):,} voxels: "
                      f"{float((grown[kept_m] > 0).mean()):.3f} still occupied, "
                      f"{float((grown[kept_m] == canvas[kept_m]).mean()):.3f} "
                      f"keep their label")
            known_only = canvas.copy()
            known_only[cut:] = 0
            scenes.append({
                "id": sid, "scene": f"sequence {seq}, scan {scan}",
                "occupied_voxels": int((grown > 0).sum()),
                "classes": int(len(np.unique(grown[grown > 0]))),
                "keep_fraction": args.keep,
                "invalid_fraction": round(inv_frac, 4),
                "variants": {"input": emit(sid, "input", known_only),
                             "extended": emit(sid, "extended", grown)},
            })

    elif args.mode == modes.SEMANTICS_FROM_STRUCTURE:
        # The ground-truth layout is observed everywhere, so the structure mask
        # is fully on and the structure stage reproduces it; only the semantic
        # stage is sampled. This is the cleanest look at what the semantic
        # prior does on its own, with the geometry taken out of the question.
        variants = [{"key": "layout", "label": "Given layout",
                     "hint": "ground-truth geometry, labels discarded"},
                    {"key": "labelled", "label": "Labelled",
                     "hint": "semantics sampled onto that layout"}]
        for spec in args.frames:
            seq, scan = spec.split(":")
            sid = f"{seq}_{scan}"
            truth, invalid = load_truth(spec)
            occ = (truth > 0).astype(np.int8)       # nothing unknown
            print(f"{spec}: labelling the ground-truth layout "
                  f"({int((truth > 0).sum()):,} occupied voxels)...")
            torch.manual_seed(args.seed)
            z_known, mask = conditioning(occ)
            if not bool((mask > 0).all()):
                print(f"  warning: {float((mask > 0).float().mean()):.3f} of the "
                      f"structure mask is set, not all of it; the layout will "
                      f"only be approximately reproduced")
            done = synthesize(z_known, mask, vae, sem, struct, device,
                              args.steps, args.resample, latent_dim, latent_scale)
            scenes.append({
                "id": sid, "scene": f"sequence {seq}, scan {scan}",
                "occupied_voxels": int((truth > 0).sum()),
                "classes": int(len(np.unique(done[done > 0]))),
                "volume_invalid": round(invalid, 4),
                "variants": {"layout": emit(sid, "layout", (truth > 0).astype(np.int32)),
                             "labelled": emit(sid, "labelled", done)},
            })

    elif args.mode in (modes.COMPLETE_GEOMETRY, modes.COMPLETE_SEMANTIC):
        guided = args.mode == modes.COMPLETE_SEMANTIC
        variants = [{"key": "observed", "label": "LiDAR scan",
                     "hint": "a single sweep, voxelised"},
                    {"key": "completed", "label": "Completed",
                     "hint": ("completed from the scan, labels included" if guided
                              else "layout completed from the scan; labels sampled")},
                    {"key": "truth", "label": "Ground truth",
                     "hint": "SemanticKITTI SSC labels; invalid voxels shown empty"}]
        for spec in args.frames:
            seq, scan, labels = load_frame(spec)
            sid = f"{seq}_{scan}"
            occ = observed_state(labels)
            seen = float((occ != -1).mean())
            truth, invalid = load_truth(spec)
            print(f"{spec}: {int((labels > 0).sum()):,} hits, "
                  f"{seen:.0%} of the volume observed, "
                  f"{invalid:.0%} of the ground truth invalid")
            torch.manual_seed(args.seed)
            z_known, mask = conditioning(occ)
            done = synthesize(z_known, mask, vae, sem, struct, device,
                              args.steps, args.resample, latent_dim, latent_scale,
                              guidance=(dict(labels=labels, known=occ != -1,
                                             **guidance_opts(args))
                                        if guided else None),
                              sampler=args.semantic_sampler, verbose=guided)

            # Report against the scan itself, not the ground truth: this is the
            # number that showed the semantic stage was ignoring the observation.
            hit = labels > 0
            kept = done[hit] > 0
            agree = (done[hit] == labels[hit])[kept]
            fid = {"scan_hits": int(hit.sum()),
                   "hits_kept": round(float(kept.mean()), 4),
                   "hits_relabelled_same": round(float(agree.mean()), 4)
                                            if kept.any() else None}
            print(f"  of the scan's {fid['scan_hits']:,} hits: "
                  f"{fid['hits_kept']:.3f} still occupied, "
                  f"{fid['hits_relabelled_same']:.3f} keep their label")

            scenes.append({
                "id": sid, "scene": f"sequence {seq}, scan {scan}",
                "observation_fidelity": fid,
                "occupied_voxels": int((truth > 0).sum()),
                "classes": int(len(np.unique(truth[truth > 0]))),
                "volume_observed": round(seen, 4),
                "volume_invalid": round(invalid, 4),
                "variants": {"observed": emit(sid, "observed", labels),
                             "completed": emit(sid, "completed", done),
                             "truth": emit(sid, "truth", truth)},
            })

    else:
        sys.exit(f"--mode {args.mode} is not wired into this exporter")

    manifest = {
        "dataset": "semantickitti",
        "dims": list(GRID),
        "axes": "x, y, z -- z is up",
        "voxel_size_m": 0.2,
        "record": "uint8 x, y, z, label",
        "num_classes": n_classes,
        "class_names": class_names,
        "palette": palette,
        "default_cut": 1.0,        # outdoor scenes are flat; nothing to cut away
        "variants": variants,
        "sampling": {"structure": args.structure,
                     "ddpm_resample": args.ddpm_resample if args.structure == "ddpm" else None,
                     "steps": args.steps, "repaint_resample": args.resample,
                     "seed": args.seed, "weights": args.config_dir or "shipped",
                     "latent_scale": latent_scale,
                     "mode": args.mode,
                     "mode_number": modes.MODES[args.mode].number,
                     **{f"guidance_{k}": v
                        for k, v in guidance_opts(args).items()},
                     "structure_stage": modes.stages("outdoor", args.mode)[0],
                     "semantic_stage": modes.stages("outdoor", args.mode)[1]},
        "scenes": scenes,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")

    total = sum(v["count"] * 4 for s in scenes for v in s["variants"].values())
    print(f"\nwrote {args.out}/  --  {len(scenes)} scenes, {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
