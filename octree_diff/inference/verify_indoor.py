"""Verification checks for the indoor pipeline.

Runs headless. Each check is skipped with a message if the checkpoint it needs
does not exist yet, so this is safe to run part-way through a training round.

    python -m inference_scripts.verify_indoor
    python -m inference_scripts.verify_indoor --checks vae generate
"""

import argparse
import glob
import os

import torch

from octree_diff.octree.build import build_semantic_octree
from octree_diff.inference.pipeline import (
    DEPTH, GRID, PATCH_SIZE, IndoorPipeline, build_doctree, labels_to_split_big,
    split_big_to_octree, voxel_mask_to_latent_mask, voxel_mask_to_node_mask,
)
from octree_diff.octree.util_octree_stuff import (
    get_non_empty_mask, points2octree, reconstruct_voxel_from_patch,
    voxel_grid_to_points,
)


from octree_diff.config import REPO_ROOT


def _data(rel):
    """Data lives under the repo's data/ dir, not the process cwd."""
    return os.path.join(REPO_ROOT, "data", rel)


def _load_patch(path, device):
    blob = torch.load(path, map_location="cpu")
    return blob["labels"].long().permute(1, 2, 0).contiguous().to(device)


def _gt_octree(labels, full_depth, device, with_features=True):
    if with_features:
        return build_semantic_octree(labels.unsqueeze(0), PATCH_SIZE, DEPTH,
                                     full_depth, device)
    return points2octree(voxel_grid_to_points(get_non_empty_mask(labels, PATCH_SIZE)),
                         depth=DEPTH, full_depth=full_depth).to(device)


# --------------------------------------------------------------------------

def check_vae(pipe, args):
    """VAE round-trips — isolates the bridge from the diffusion models."""
    print("\n[vae] structure VAE round-trip")
    ious = []
    for f in sorted(glob.glob(_data("split_outputs/*.pt")))[:args.n]:
        x = torch.load(f, weights_only=True).unsqueeze(0).to(pipe.device).float()
        with torch.no_grad():
            mu, _ = pipe.str_vae.encode(x)
            rec = torch.where(pipe.str_vae.decode(mu) > 0, 1.0, -1.0)
        a, b = x > 0, rec > 0
        ious.append(((a & b).sum() / (a | b).sum().clamp_min(1)).item())
    print(f"  IoU {sum(ious)/len(ious):.4f} over {len(ious)} splits   (expect ~1.0)")

    print("[vae] semantic VAE round-trip")
    accs, occ_accs = [], []
    for f in sorted(glob.glob(_data("patches_val/*.pt")))[:args.n]:
        labels = _load_patch(f, pipe.device)
        octree = _gt_octree(labels, pipe.sem_full_depth, pipe.device)
        with torch.no_grad():
            _, mu, doctree = pipe.sem_vae.extract_code(octree)
            out = pipe.sem_vae.decode_code(mu, doctree, update_octree=False, pos=None)
            rec = reconstruct_voxel_from_patch(out["sem_voxs"], octree, depth=DEPTH,
                                               shape=(1,) + GRID)[0]
        occ = labels > 0
        accs.append((rec == labels).float().mean().item())
        occ_accs.append((rec[occ] == labels[occ]).float().mean().item())
    print(f"  all-voxel acc {sum(accs)/len(accs):.4f}, occupied-voxel acc "
          f"{sum(occ_accs)/len(occ_accs):.4f} over {len(accs)} patches")
    print("  (far below the sem_vae training accuracy would mean the bridge is wrong)")


def _occupancy_fraction(split_big):
    return (split_big > 0).float().mean().item()


def _class_hist(labels, num_classes=92):
    return torch.bincount(labels.flatten().cpu(), minlength=num_classes).float()


def check_generate(pipe, args):
    """Unconditional samples vs the training distribution."""
    print("\n[generate] sampling")
    fracs, hists = [], []
    for i in range(args.n_gen):
        split_big = pipe.sample_structure(args.steps)
        fracs.append(_occupancy_fraction(split_big))
        octree = split_big_to_octree(split_big, pipe.sem_full_depth)
        labels = pipe.sample_semantics(octree, args.steps)
        hists.append(_class_hist(labels))
    gen_frac = sum(fracs) / len(fracs)

    real = []
    for f in sorted(glob.glob(_data("split_outputs/*.pt")))[:100]:
        real.append(_occupancy_fraction(torch.load(f, weights_only=True)))
    real_frac = sum(real) / len(real)
    print(f"  occupied fraction of the depth-5 grid: generated {gen_frac:.4f} vs "
          f"real {real_frac:.4f}")

    hist = torch.stack(hists).sum(0)
    hist_occ = hist.clone(); hist_occ[0] = 0
    present = int((hist_occ > 0).sum())
    top = hist_occ.argsort(descending=True)[:5]
    print(f"  classes present in samples: {present}/92   top-5 {top.tolist()}")
    if present < 5:
        print("  !! near-degenerate class histogram — the semantic model may have collapsed")

    counts_path = _data("replica_class_counts.pt")
    if os.path.exists(counts_path):
        real_counts = torch.load(counts_path, weights_only=False)["counts"].float()
        real_occ = real_counts.clone(); real_occ[0] = 0
        print(f"  training set top-5 {real_occ.argsort(descending=True)[:5].tolist()}")


def check_reference(pipe, args):
    """Compare against the lost shipped model's outputs."""
    ref_dir = args.ref_dir
    if not ref_dir:
        print("\n[reference] skipped — no --ref-dir given "
              "(samples from the original shipped model; not needed to validate a retrain)")
        return
    if not os.path.isdir(ref_dir):
        print(f"\n[reference] skipped — {ref_dir} not found")
        return
    print("\n[reference] shipped-model samples (data_gen/)")
    hist = torch.zeros(92)
    files = sorted(glob.glob(os.path.join(ref_dir, "*.pt")))[:args.n_ref]
    occ_fracs = []
    for f in files:
        lab = torch.load(f, map_location="cpu", weights_only=True)
        hist += _class_hist(lab)
        occ_fracs.append((lab > 0).float().mean().item())
    hist_occ = hist.clone(); hist_occ[0] = 0
    print(f"  {len(files)} samples: occupied voxel fraction {sum(occ_fracs)/len(occ_fracs):.4f}, "
          f"{int((hist_occ > 0).sum())}/92 classes present")
    print(f"  top-5 {hist_occ.argsort(descending=True)[:5].tolist()}")
    print("  compare these numbers against [generate] above")


def check_inpaint(pipe, args):
    """Mask half a val patch and complete it."""
    print("\n[inpaint] half-scene completion")
    labels = _load_patch(sorted(glob.glob(_data("patches_val/*.pt")))[0], pipe.device)
    known = torch.zeros(GRID, dtype=torch.bool, device=pipe.device)
    known[:GRID[0] // 2] = True
    known_labels = torch.where(known, labels, torch.zeros_like(labels))

    split_known = labels_to_split_big(known_labels, pipe.device)
    split_out = pipe.inpaint_structure(split_known, voxel_mask_to_latent_mask(known),
                                       args.steps, resample=args.resample)
    split_gt = labels_to_split_big(labels, pipe.device)
    h = split_gt.shape[2] // 2
    a, b = split_out[0, 0, :h] > 0, split_gt[0, 0, :h] > 0
    print(f"  structure known-half IoU vs GT: "
          f"{((a & b).sum() / (a | b).sum().clamp_min(1)).item():.4f}   (expect ~1.0)")

    octree = split_big_to_octree(split_out, pipe.sem_full_depth)
    doctree = build_doctree(octree)
    node_mask = voxel_mask_to_node_mask(known, octree, doctree)
    out = pipe.inpaint_semantics(octree, known_labels, node_mask, args.steps,
                                 resample=args.resample)

    occ_k, occ_u = (labels > 0) & known, (labels > 0) & ~known
    print(f"  semantic known-half acc {(out[occ_k] == labels[occ_k]).float().mean():.4f}, "
          f"unknown-half acc {(out[occ_u] == labels[occ_u]).float().mean():.4f}")
    print("  (known-half below the VAE round-trip means latents are leaking across "
          "the mask boundary through the graph decoder)")


def check_outpaint(pipe, args):
    """Extend along +x and look for seams."""
    print("\n[outpaint] extension along +x")
    labels = _load_patch(sorted(glob.glob(_data("patches_val/*.pt")))[0], pipe.device)
    scene = labels.clone()
    half = GRID[0] // 2
    fracs = []

    for k in range(args.n_out):
        window = torch.zeros(GRID, dtype=torch.long, device=pipe.device)
        window[:half] = scene[-half:]
        known = torch.zeros(GRID, dtype=torch.bool, device=pipe.device)
        known[:half] = True

        split_out = pipe.inpaint_structure(labels_to_split_big(window, pipe.device),
                                           voxel_mask_to_latent_mask(known),
                                           args.steps, resample=args.resample)
        octree = split_big_to_octree(split_out, pipe.sem_full_depth)
        doctree = build_doctree(octree)
        out = pipe.inpaint_semantics(octree, window,
                                     voxel_mask_to_node_mask(known, octree, doctree),
                                     args.steps, resample=args.resample)
        scene = torch.cat([scene, out[half:]], dim=0)
        fracs.append((out[half:] > 0).float().mean().item())

    # occupancy either side of each seam; a big jump means the windows do not agree
    print(f"  final scene {tuple(scene.shape)}")
    print(f"  occupied fraction per extension step: "
          f"{['%.3f' % f for f in fracs]}   (drift means the model is losing the conditioning)")
    for k in range(args.n_out):
        seam = GRID[0] + k * half
        before = (scene[seam - 4:seam] > 0).float().mean().item()
        after = (scene[seam:seam + 4] > 0).float().mean().item()
        print(f"  seam at x={seam}: occupancy {before:.3f} -> {after:.3f}")


CHECKS = {
    "vae": (check_vae, []),
    "generate": (check_generate, ["structure_diffusion", "semantic_diffusion"]),
    "reference": (check_reference, []),
    "inpaint": (check_inpaint, ["structure_diffusion", "semantic_diffusion"]),
    "outpaint": (check_outpaint, ["structure_diffusion", "semantic_diffusion"]),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checks", nargs="+", default=list(CHECKS), choices=list(CHECKS))
    p.add_argument("--steps", type=int, default=50)
    # hyphenated canonical spelling, underscore kept as an alias
    p.add_argument("--repaint-resample", "--resample", type=int, default=1,
                   dest="resample", help="RePaint jumps per step")
    p.add_argument("--n", type=int, default=10, help="patches per VAE check")
    p.add_argument("--n-gen", "--n_gen", type=int, default=8, dest="n_gen")
    p.add_argument("--n-ref", "--n_ref", type=int, default=100, dest="n_ref")
    p.add_argument("--n-out", "--n_out", type=int, default=4, dest="n_out")
    p.add_argument("--ref-dir", default=None,
                   help="directory of samples from a previously shipped model, for the "
                        "[reference] check; omit to skip it")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(0)
    pipe = IndoorPipeline(args.device)
    print(f"latent scales: structure {pipe.str_scale:.4f}, semantic {pipe.sem_scale:.4f}")

    for name in args.checks:
        fn, needs = CHECKS[name]
        missing = [n for n in needs if not os.path.exists(pipe.paths[n])]
        if missing:
            print(f"\n[{name}] skipped — not trained yet: {', '.join(missing)}")
            continue
        fn(pipe, args)


if __name__ == "__main__":
    main()
