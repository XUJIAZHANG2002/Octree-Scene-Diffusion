"""Indoor (Replica) inference: generation, inpainting and scene extension.

The notebooks in `notebooks/kitti/` are all outdoor and their structure model is
a different formulation from the one trained here — see `notebooks/README.md`.
This script is the indoor path.

    # unconditional
    python -m inference_scripts.inference_indoor --mode generate \
        --stages both --num_samples 8 --out out/gen

    # completion from a partial scene
    python -m inference_scripts.inference_indoor --mode inpaint \
        --input scene.pt --known-mask known.pt --out out/complete

    # extend along +x
    python -m inference_scripts.inference_indoor --mode outpaint \
        --input scene.pt --steps-out 4 --out out/extend

Everything runs headless. Nothing in `utils/util_visual_stuff.py` is imported —
both functions there block on an Open3D GUI window.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import yaml

from models.networks.diffusion_networks.graph_unet_hr import UNet3DModel
from models.networks.dualoctree_networks.dual_octree import DualOctree
from models.networks.dualoctree_networks.graph_sem_vae import GraphVAE
from models.structure_networks.structure_vae import VoxelVAE
from models.structure_networks.unet_3d import StructureUNet
from utils.config_loader import load_config
from utils.latent_scale import load_latent_scale
from utils.sampling import (
    ddim_inpaint_dense, ddim_inpaint_graph, ddim_sample_dense, ddim_sample_graph,
)
from utils.util_dualoctree import octree2split_small, split2octree_small
from utils.util_octree_stuff import (
    assign_octree_patch_features, get_non_empty_mask, points2octree,
    reconstruct_voxel_from_patch, split2splitbig, splitbig2split,
    voxel_grid_to_points, voxel_to_patch,
)

# Indoor geometry. A [128,128,64] label patch reduces to a cubic [64,64,64]
# occupancy mask at patch_size 2, so none of the z-padding / cropping that runs
# through the outdoor notebooks applies here.
GRID = (128, 128, 64)
PATCH_SIZE = 2
DEPTH = 6
OCTREE_FULL_DEPTH = 4     # for the split representation
SPLIT_DEPTH = 4


# --------------------------------------------------------------------------
# voxels <-> octree/split
# --------------------------------------------------------------------------

def labels_to_split_big(labels, device):
    """[H,W,D] int labels -> [1,1,32,32,32] split grid in {-1,+1}."""
    mask = get_non_empty_mask(labels.to(device), PATCH_SIZE)      # [64,64,64]
    if mask.sum() == 0:
        raise ValueError("input scene is empty")
    octree = points2octree(voxel_grid_to_points(mask), depth=DEPTH,
                           full_depth=OCTREE_FULL_DEPTH).to(device)
    return split2splitbig(octree2split_small(octree, SPLIT_DEPTH))


def split_big_to_octree(split_big, sem_vae_full_depth):
    """[1,1,32,32,32] split grid -> a depth-6 octree matching the training graph.

    `split2octree_small` yields an octree that is *full* at depth 4 (4096 nodes),
    because the split representation is defined on the dense depth-4 grid. The
    semantic VAE was trained on octrees built by `points2octree(..., full_depth=3)`,
    which are sparse there (~1900 nodes for a typical patch), and the dual-octree
    graph the UNet consumes includes those coarse leaves — so feeding it the
    dense version changes the graph it sees.

    The notebooks paper over this with `create_child_octree`, but that keeps the
    dense depth-4 layer. Extracting the depth-6 occupancy and rebuilding at
    full_depth 3 reproduces the training octree exactly (verified: identical
    `nnum` at every depth, identical `doctree.total_num`).
    """
    split_big = torch.where(split_big > 0, 1.0, -1.0)
    octree_s = split2octree_small(splitbig2split(split_big), DEPTH, OCTREE_FULL_DEPTH)

    x, y, z, _ = octree_s.xyzb(DEPTH, nempty=False)
    res = 2 ** DEPTH
    mask = torch.zeros(res, res, res, dtype=torch.bool, device=split_big.device)
    mask[x.long(), y.long(), z.long()] = True

    return points2octree(voxel_grid_to_points(mask), depth=DEPTH,
                         full_depth=sem_vae_full_depth).to(split_big.device)


def build_doctree(octree):
    doctree = DualOctree(octree)
    doctree.post_processing_for_docnn()
    return doctree


def voxel_mask_to_latent_mask(mask_vox):
    """[128,128,64] bool -> [1,1,32,32,32] float, by 4x4x2 max-blocks.

    The structure latent is spatially 1:1 with the [32,32,32] split grid, and
    each split cell covers a 4x4x2 block of voxels. VoxelVAE is fully
    convolutional so the mask transfers, though each latent cell has a few-voxel
    receptive field and the boundary is therefore approximate.
    """
    m = mask_vox.float()[None, None]                     # [1,1,128,128,64]
    return F.max_pool3d(m, kernel_size=(4, 4, 2), stride=(4, 4, 2))


def voxel_mask_to_node_mask(mask_vox, octree, doctree):
    """[128,128,64] bool -> [total_num, 1] float over the dual-octree latent rows.

    The latent has one row per dual-octree node, laid out (see
    `DualOctree.calc_batch_id`) as

        [ leaves@full_depth, leaves@full_depth+1, ..., leaves@DEPTH-1, all nodes@DEPTH ]

    so it is not enough to condition the depth-6 tail: the coarse leaf rows feed
    the decoder's upsampling path, and leaving them unconditioned lets noise from
    the generated region leak across the whole scene.

    A depth-6 node (x,y,z) covers voxels (2x..2x+1, 2y..2y+1, z) — the mapping
    `reconstruct_voxel_from_patch` uses on the way out. A coarser node at depth d
    covers a 2^(DEPTH-d) cube of those, and counts as known only if all of it is.
    """
    device = mask_vox.device
    res = 2 ** DEPTH

    # known-ness at depth-6 node resolution: all four voxels of the patch known
    m = mask_vox
    known = (m[0::2, 0::2, :] & m[1::2, 0::2, :] &
             m[0::2, 1::2, :] & m[1::2, 1::2, :]).float()      # [64,64,64]

    # coarser levels: a node is known only if every child is ("all" = min-pool)
    known_at = {DEPTH: known}
    for d in range(DEPTH - 1, octree.full_depth - 1, -1):
        prev = known_at[d + 1][None, None]
        known_at[d] = (-F.max_pool3d(-prev, kernel_size=2, stride=2))[0, 0]

    rows = []
    for d in range(octree.full_depth, DEPTH):
        leaf = octree.children[d] < 0
        x, y, z, _ = octree.xyzb(d, nempty=False)
        x, y, z = x.long()[leaf], y.long()[leaf], z.long()[leaf]
        rows.append(known_at[d][x, y, z])
    x, y, z, _ = octree.xyzb(DEPTH, nempty=False)
    rows.append(known_at[DEPTH][x.long(), y.long(), z.long()])

    node_mask = torch.cat(rows).to(device).unsqueeze(1)
    assert node_mask.shape[0] == doctree.total_num, (
        f"row count {node_mask.shape[0]} != doctree.total_num {doctree.total_num}")
    return node_mask


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

class IndoorPipeline:
    """Loads models on first use.

    The four stages finish training at very different times, so a structure-only
    run must not require the semantic diffusion checkpoint to exist yet.
    """

    def __init__(self, device="cuda"):
        self.device = torch.device(device)

        self.str_vae_cfg = load_config("configs/structure_vae_config.yaml")
        self.str_diff_cfg = load_config("configs/structure_diffusion_config.yaml")
        self.sem_vae_cfg = load_config("configs/sem_vae_config.yaml")
        self.sem_diff_cfg = load_config("configs/sem_diffusion_config.yaml")

        self.latent_dim = self.sem_vae_cfg["model"]["latent_dim"]
        self.str_z_ch = self.str_vae_cfg["model"]["z_channels"]
        self.sem_full_depth = self.sem_vae_cfg["model"]["full_depth"]

        self.str_scale = load_latent_scale(self._meta(self.str_diff_cfg))
        self.sem_scale = load_latent_scale(self._meta(self.sem_diff_cfg))

        self.paths = {
            "structure_vae": self.str_vae_cfg["training"]["checkpoint_path"],
            "structure_diffusion": self.str_diff_cfg["training"]["save_path"],
            "semantic_vae": self.sem_vae_cfg["training"]["save_path"],
            "semantic_diffusion": self.sem_diff_cfg["training"]["save_path"],
        }
        self._cache = {}

    @staticmethod
    def _meta(diff_cfg):
        return os.path.splitext(diff_cfg["training"]["save_path"])[0] + "_meta.yaml"

    def _load(self, name, build, ckpt):
        if name not in self._cache:
            if not os.path.exists(ckpt):
                raise FileNotFoundError(
                    f"{name} checkpoint not found at {ckpt} — has that stage finished training?")
            model = build().to(self.device).eval()
            model.load_state_dict(torch.load(ckpt, map_location=self.device,
                                             weights_only=True))
            self._cache[name] = model
            print(f"  loaded {name} from {ckpt}")
        return self._cache[name]

    @property
    def str_vae(self):
        c = self.str_vae_cfg["model"]
        return self._load("structure_vae",
                          lambda: VoxelVAE(z_channels=c["z_channels"],
                                           base=c["base_channels"],
                                           in_ch=c["in_channels"]),
                          self.str_vae_cfg["training"]["checkpoint_path"])

    @property
    def str_unet(self):
        u = self.str_diff_cfg["unet"]
        return self._load("structure_diffusion",
                          lambda: StructureUNet(in_ch=u["in_ch"], base_ch=u["base_ch"],
                                                time_emb_dim=u["time_emb_dim"]),
                          self.str_diff_cfg["training"]["save_path"])

    @property
    def sem_vae(self):
        m = self.sem_vae_cfg["model"]
        return self._load("semantic_vae",
                          lambda: GraphVAE(
                              depth=m["depth_in"], channel_in=m["channel_in"],
                              nout=m["nout"], full_depth=m["full_depth"],
                              depth_stop=m["depth_stop"], depth_out=m["depth_in"],
                              latent_dim=m["latent_dim"],
                              num_classes=m["total_classes"],
                              resblk_num=m["resblk_num"]),
                          self.sem_vae_cfg["training"]["save_path"])

    @property
    def sem_unet(self):
        m, u = self.sem_vae_cfg["model"], self.sem_diff_cfg["unet"]
        return self._load("semantic_diffusion",
                          lambda: UNet3DModel(
                              image_size=u["image_size"], input_depth=m["depth_in"],
                              full_depth=m["full_depth"], in_channels=m["latent_dim"],
                              model_channels=u["model_channels"],
                              lr_model_channels=u["lr_model_channels"],
                              out_channels=m["latent_dim"],
                              num_res_blocks=u["num_res_blocks"],
                              channel_mult=u["channel_mult"]),
                          self.sem_diff_cfg["training"]["save_path"])

    # -- structure ---------------------------------------------------------

    @torch.no_grad()
    def sample_structure(self, steps, eta=0.0):
        z = ddim_sample_dense(self.str_unet, (1, self.str_z_ch, 32, 32, 32),
                              steps=steps, device=self.device, eta=eta)
        return torch.where(self.str_vae.decode(z / self.str_scale) > 0, 1.0, -1.0)

    @torch.no_grad()
    def inpaint_structure(self, split_big_known, latent_mask, steps, eta=0.0,
                          resample=1, best_of=1, return_score=False):
        """Complete the layout, optionally keeping the best of several samples.

        Roughly one draw in twenty degrades badly (measured: layout IoU 0.54 where
        the other seeds all give 0.94). Those draws also disagree with the region
        the sensor actually observed — known-region agreement drops from ~0.98 to
        ~0.89 — so they can be rejected without any ground truth. `best_of > 1`
        samples repeatedly and keeps the candidate that best matches the input.
        """
        split_big_known = split_big_known.to(self.device)
        mu, _ = self.str_vae.encode(split_big_known)
        z_known = mu * self.str_scale
        mask = latent_mask.expand_as(z_known)

        known_cells = latent_mask[0, 0].bool()
        target = (split_big_known[0, 0] > 0) & known_cells

        best, best_score = None, -1.0
        for _ in range(max(best_of, 1)):
            z = ddim_inpaint_dense(self.str_unet, z_known, mask, steps=steps,
                                   device=self.device, eta=eta, resample=resample)
            cand = torch.where(self.str_vae.decode(z / self.str_scale) > 0, 1.0, -1.0)

            a = (cand[0, 0] > 0) & known_cells
            score = ((a & target).sum() / (a | target).sum().clamp_min(1)).item()
            if score > best_score:
                best, best_score = cand, score

        return (best, best_score) if return_score else best

    # -- semantics ---------------------------------------------------------

    @torch.no_grad()
    def sample_semantics(self, octree, steps, eta=0.0):
        doctree = build_doctree(octree)
        z = ddim_sample_graph(self.sem_unet, doctree, doctree.total_num,
                              self.latent_dim, steps=steps, device=self.device, eta=eta)
        return self._decode_semantics(z, doctree, octree)

    @torch.no_grad()
    def inpaint_semantics(self, octree, known_labels, node_mask, steps,
                          eta=0.0, resample=1):
        """Known semantics are encoded on the *completed* octree.

        Assigning the (partial) label volume to the completed octree and running
        the encoder avoids any cross-octree node matching: the latent rows are
        already in the right frame, and only the rows flagged by `node_mask` are
        used as conditioning.
        """
        octree = octree.to(self.device)
        patch_feat = voxel_to_patch(known_labels.to(self.device).long().unsqueeze(0),
                                    PATCH_SIZE)
        assign_octree_patch_features(patch_feat[0], octree, DEPTH)

        _, mu, doctree = self.sem_vae.extract_code(octree)
        z_known = mu * self.sem_scale

        z = ddim_inpaint_graph(self.sem_unet, doctree, z_known, node_mask,
                               steps=steps, device=self.device, eta=eta,
                               resample=resample)
        return self._decode_semantics(z, doctree, octree)

    def _decode_semantics(self, z, doctree, octree):
        out = self.sem_vae.decode_code(z / self.sem_scale, doctree,
                                       update_octree=False, pos=None)
        labels = reconstruct_voxel_from_patch(out["sem_voxs"], octree, depth=DEPTH,
                                              shape=(1,) + GRID)
        return labels[0]


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def split_big_to_occupancy(split_big):
    """[1,1,32,32,32] -> [128,128,64] binary occupancy, for structure-only output."""
    occ = (split_big[0, 0] > 0).long()                       # [32,32,32]
    return occ.repeat_interleave(4, 0).repeat_interleave(4, 1).repeat_interleave(2, 2)


def run_generate(pipe, args, out_dir):
    for i in range(args.num_samples):
        t0 = time.time()

        if args.structure_from:
            src = torch.load(args.structure_from, map_location=pipe.device,
                             weights_only=True)
            split_big = labels_to_split_big(src.squeeze(), pipe.device)
        else:
            split_big = pipe.sample_structure(args.steps, args.eta)

        if args.stages == "structure":
            torch.save(split_big_to_occupancy(split_big).cpu(),
                       os.path.join(out_dir, f"occupancy_{i:04d}.pt"))
        else:
            octree = split_big_to_octree(split_big, pipe.sem_full_depth)
            labels = pipe.sample_semantics(octree, args.steps, args.eta)
            torch.save(labels.cpu(), os.path.join(out_dir, f"labels_{i:04d}.pt"))
            torch.save(split_big_to_occupancy(split_big).cpu(),
                       os.path.join(out_dir, f"occupancy_{i:04d}.pt"))

        print(f"  sample {i}: {time.time() - t0:.1f}s")


def run_inpaint(pipe, args, out_dir):
    labels = torch.load(args.input, map_location=pipe.device, weights_only=True).squeeze().long()
    known = torch.load(args.known_mask, map_location=pipe.device, weights_only=True).squeeze().bool()
    if labels.shape != GRID or known.shape != GRID:
        raise ValueError(f"expected {GRID} tensors, got {tuple(labels.shape)} "
                         f"and {tuple(known.shape)}")

    known_labels = torch.where(known, labels, torch.zeros_like(labels))

    split_big_known = labels_to_split_big(known_labels, pipe.device)
    latent_mask = voxel_mask_to_latent_mask(known)
    split_big = pipe.inpaint_structure(split_big_known, latent_mask, args.steps,
                                       args.eta, args.repaint_resample, args.best_of)

    octree = split_big_to_octree(split_big, pipe.sem_full_depth)
    doctree = build_doctree(octree)
    node_mask = voxel_mask_to_node_mask(known, octree, doctree)

    out_labels = pipe.inpaint_semantics(octree, known_labels, node_mask, args.steps,
                                        args.eta, args.repaint_resample)

    torch.save(out_labels.cpu(), os.path.join(out_dir, "labels_completed.pt"))
    torch.save(split_big_to_occupancy(split_big).cpu(),
               os.path.join(out_dir, "occupancy_completed.pt"))
    print(f"  completed {int((~known).sum())} unknown voxels")


def run_outpaint(pipe, args, out_dir):
    """Extend along +x, half a patch (64 voxels) at a time."""
    labels = torch.load(args.input, map_location=pipe.device, weights_only=True).squeeze().long()
    if labels.shape != GRID:
        raise ValueError(f"expected a {GRID} tensor, got {tuple(labels.shape)}")

    scene = labels.clone()
    half = GRID[0] // 2

    for k in range(args.steps_out):
        t0 = time.time()

        # window = [second half of what we have | empty]; the known part is x < 64
        window = torch.zeros(GRID, dtype=torch.long, device=pipe.device)
        window[:half] = scene[-half:]
        known = torch.zeros(GRID, dtype=torch.bool, device=pipe.device)
        known[:half] = True

        split_big_known = labels_to_split_big(window, pipe.device)
        latent_mask = voxel_mask_to_latent_mask(known)
        split_big = pipe.inpaint_structure(split_big_known, latent_mask, args.steps,
                                           args.eta, args.repaint_resample, args.best_of)

        octree = split_big_to_octree(split_big, pipe.sem_full_depth)
        doctree = build_doctree(octree)
        node_mask = voxel_mask_to_node_mask(known, octree, doctree)
        out_labels = pipe.inpaint_semantics(octree, window, node_mask, args.steps,
                                            args.eta, args.repaint_resample)

        scene = torch.cat([scene, out_labels[half:]], dim=0)
        print(f"  step {k}: scene now {tuple(scene.shape)}  ({time.time() - t0:.1f}s)")

    torch.save(scene.cpu(), os.path.join(out_dir, "labels_extended.pt"))


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["generate", "inpaint", "outpaint"])
    p.add_argument("--stages", default="both", choices=["structure", "semantic", "both"],
                   help="generate only: which stages to run")
    p.add_argument("--structure-from", help="generate only: reuse a layout from this .pt")
    p.add_argument("--input", help="inpaint/outpaint: [128,128,64] int label tensor")
    p.add_argument("--known-mask", help="inpaint: [128,128,64] bool tensor, True = observed")
    p.add_argument("--steps-out", type=int, default=4, help="outpaint: extension steps")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--steps", type=int, default=100, help="DDIM steps")
    p.add_argument("--eta", type=float, default=0.0, help="0 = deterministic DDIM")
    p.add_argument("--repaint-resample", type=int, default=1,
                   help="RePaint jumps per step; >1 costs that many model evals")
    p.add_argument("--best-of", type=int, default=1,
                   help="sample the layout this many times and keep the one that best "
                        "agrees with the observed region; rejects the ~5%% of draws "
                        "that degrade")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.mode == "inpaint" and not (args.input and args.known_mask):
        p.error("--mode inpaint needs --input and --known-mask")
    if args.mode == "outpaint" and not args.input:
        p.error("--mode outpaint needs --input")
    if args.stages == "semantic" and not args.structure_from:
        p.error("--stages semantic needs --structure-from")

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("loading models...")
    pipe = IndoorPipeline(args.device)
    print(f"  latent scales: structure {pipe.str_scale:.4f}, semantic {pipe.sem_scale:.4f}")

    print(f"running {args.mode}...")
    {"generate": run_generate, "inpaint": run_inpaint, "outpaint": run_outpaint}[args.mode](
        pipe, args, args.out)

    with open(os.path.join(args.out, "manifest.yaml"), "w") as f:
        yaml.safe_dump({
            "mode": args.mode, "stages": args.stages, "steps": args.steps,
            "eta": args.eta, "repaint_resample": args.repaint_resample,
            "seed": args.seed, "num_samples": args.num_samples,
            "input": args.input, "known_mask": args.known_mask,
            "latent_scale": {"structure": pipe.str_scale, "semantic": pipe.sem_scale},
            "checkpoints": pipe.paths,
        }, f, sort_keys=False)

    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
