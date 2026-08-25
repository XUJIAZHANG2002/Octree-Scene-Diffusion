"""Export SemanticKITTI completions to the web viewer's binary format.

The outdoor counterpart to export_web_scenes.py. Same encoding, same manifest
shape, so the viewer needs no changes -- it reads dims, palette, class names and
the variant list out of the manifest.

Two differences from indoor, both forced by the data:

1. **Two variants, not three.** SemanticKITTI's dense voxel ground truth lives in
   a `sequences/*/voxels/` download we do not have; only raw scans and per-point
   labels are here. So there is `observed` (one LiDAR sweep, voxelised) and
   `completed`, and no `truth` panel. Aggregating neighbouring sweeps via
   poses.txt would approximate one, but it would not be the official ground
   truth and should not be labelled as such.

2. **The observation is the scan itself**, not a mask over a complete scene. A
   single sweep fills about 0.6% of the volume. The known region is recovered by
   tracing back from each hit to the sensor origin, marking everything on the way
   as observed-empty -- so "unknown" means genuinely occluded, not just empty.

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
            elif occ[x, y, z] == 0:
                break          # already carved; the rest of this ray is done
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

def load_models(weights, device):
    from octree_diff.models.semantic.graph_sem_vae import GraphVAE
    from octree_diff.models.semantic.graph_unet_hr import UNet3DModel as SemUNet
    from octree_diff.models.outdoor.graph_unet_lr import UNet3DModel as StructUNet

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

    struct = StructUNet(full_depth=4, in_split_channels=8, model_channels=32,
                        out_split_channels=8, attention_resolutions=[2, 4, 8],
                        dims=3, num_heads=4).to(device)
    struct.load_state_dict(torch.load(os.path.join(weights, "structure_model.pt"),
                                      map_location=device))
    struct.eval()
    return vae, sem, struct


@torch.no_grad()
def complete(labels, occ, vae, sem, struct, device, steps, resample):
    from octree_diff.diffusion.util_sample_stuff import (
        ddim_sample_logsnr_cosine, ddim_sample_structure_logsnr_cosine_inpaint)
    from octree_diff.models.semantic.dual_octree import DualOctree
    from octree_diff.octree.util_dualoctree import octree2split_small, split2octree_small
    from octree_diff.octree.util_octree_stuff import (
        get_non_empty_mask, points2octree, reconstruct_voxel_from_patch,
        voxel_grid_to_points)

    def to_split(vol):
        """Dense (64,64,64) occupancy -> the [1,8,16,16,8] split grid the UNet eats."""
        octree = points2octree(voxel_grid_to_points(vol), depth=DEPTH, full_depth=FULL_DEPTH)
        return octree2split_small(octree, 4)[:, :, :, :, 0:8]

    known = torch.from_numpy(occ != -1)
    mask32 = get_non_empty_mask(known, PATCH)                 # (64,64,32)
    padded_mask = torch.zeros((64, 64, 64))
    padded_mask[:, :, :GRID[2]] = mask32

    z32 = torch.from_numpy(reduce_patches_tri_state(occ).astype(np.float32))
    padded_z = torch.zeros((64, 64, 64))
    padded_z[:, :, :GRID[2]] = z32
    padded_z[padded_z == -1] = 0                              # unknown draws as empty

    patch_mask = to_split(padded_mask)
    patch_mask[patch_mask == -1] = 0
    z_known = to_split(padded_z)

    z_T = torch.randn(1, 8, 16, 16, 8, device=device)
    sample = ddim_sample_structure_logsnr_cosine_inpaint(
        z_T, z_known.to(device), struct, patch_mask.to(device),
        S=steps, jump_n_sample=resample)
    sample = torch.where(sample > 0, 1.0, -1.0)

    split = torch.ones((1, 8, 16, 16, 16)) * -1
    split[:, :, :, :, 0:8] = sample.cpu()
    octree = split2octree_small(split, DEPTH, 4)

    doctree = DualOctree(vae.create_child_octree(octree).to(device))
    doctree.post_processing_for_docnn()
    z = ddim_sample_logsnr_cosine(torch.randn(doctree.total_num, LATENT_DIM).to(device),
                                  model=sem, doctree=doctree, S=steps)
    out = vae.decode_code(z, doctree, update_octree=False, pos=None)
    vox = reconstruct_voxel_from_patch(out["sem_voxs"], octree.to(device),
                                       depth=DEPTH, shape=(1,) + GRID, patch_size=PATCH)
    return vox[0].detach().cpu().numpy().astype(np.int32)


# --------------------------------------------------------------------- driver

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/home/brendan/SemCityOcto/dataset")
    ap.add_argument("--weights", default="/home/brendan/SemCityOcto")
    ap.add_argument("--frames", nargs="+", default=["00:000202"],
                    help="sequence:scan pairs, e.g. 00:000202")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "web_kitti"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--resample", type=int, default=20, help="RePaint jumps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

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

    print("loading models...")
    vae, sem, struct = load_models(args.weights, device)

    out_scenes = []
    for spec in args.frames:
        seq, scan = spec.split(":")
        labels = np.asarray(velodene_to_voxel(args.data, seq, scan)).astype(np.int32)
        labels[labels == 255] = 0                     # 255 marks invalid, not a class

        occ = observed_state(labels)
        seen = float((occ != -1).mean())
        print(f"{spec}: {int((labels > 0).sum()):,} hits, {seen:.0%} of the volume observed")

        torch.manual_seed(args.seed)
        done = complete(labels, occ, vae, sem, struct, device, args.steps, args.resample)
        recall = float((done[labels > 0] == labels[labels > 0]).mean())
        print(f"  label agreement on the observed hits: {recall:.4f}")

        variants = {}
        for name, vol in (("observed", labels), ("completed", done)):
            buf = encode(vol)
            vis, dropped = check_roundtrip(vol, buf)
            fn = f"{seq}_{scan}_{name}.bin"
            buf.tofile(os.path.join(args.out, fn))
            variants[name] = {"file": fn, "count": int(len(buf))}
            print(f"  {name:10s} {vis:7,} visible  {len(buf) * 4 / 1024:6.0f} KB")

        out_scenes.append({
            "id": f"{seq}_{scan}",
            "scene": f"sequence {seq}, scan {scan}",
            "occupied_voxels": int((labels > 0).sum()),
            "classes": int(len(np.unique(labels[labels > 0]))),
            "volume_observed": round(seen, 4),
            "accuracy": {"observed_hit_agreement": round(recall, 4)},
            "variants": variants,
        })

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
        "variants": [
            {"key": "observed", "label": "LiDAR scan", "hint": "a single sweep, voxelised"},
            {"key": "completed", "label": "Completed", "hint": "the model's completion"},
        ],
        "sampling": {"steps": args.steps, "repaint_resample": args.resample,
                     "seed": args.seed},
        "scenes": out_scenes,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")

    total = sum(v["count"] * 4 for s in out_scenes for v in s["variants"].values())
    print(f"\nwrote {args.out}/  --  {len(out_scenes)} scenes, {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
