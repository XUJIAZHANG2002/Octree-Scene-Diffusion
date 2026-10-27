"""Compute per-class voxel frequencies over a patch dataset.

Replica has no shipped class-count file (`class_counts_total_epoch.pt` in the
Replica-Dataset tree is a 21-entry KITTI leftover), so counts have to be derived
from the patches themselves.

    python -m octree_diff.data.compute_class_weights --src data/patches_128x128x64_s16 \
        --out data/replica_class_counts.pt

Saves a dict with the raw counts plus a few ready-to-use weightings:
    counts            [C] int64   voxels per class over the whole set
    freq              [C] float   counts / counts.sum()
    weights_inv       [C] float   1 / freq, normalised to mean 1
    weights_inv_sqrt  [C] float   1 / sqrt(freq), normalised to mean 1
    weights_enet      [C] float   (1 - b) / (1 - b^count), b = 0.9999 ("effective
                                  number of samples", Cui et al. 2019)

Empty classes get weight 0 so they cannot blow up the loss.
"""

import argparse
import os

import torch
import tqdm


def _normalise(w, present, clamp=None, max_ratio=None):
    """Mean-normalise over present classes, then optionally cap the spread.

    Two caps, because they do different things:

    `clamp` is the original symmetric cap, `[1/clamp, clamp]`. It is the wrong
    tool for this distribution and is off by default. After mean-normalising,
    Replica's common classes land at 0.01-0.10 -- all *below* 1/10 -- so a
    symmetric clamp of 10 raises floor, wall and ceiling onto the same 0.1 floor
    and destroys the ordering among them. That is the "flattens the top six
    classes" failure recorded in configs/CHANGELOG.md.

    `max_ratio` caps only the tail: no class may exceed `max_ratio` times the
    median present weight. The head is untouched. This is what the skew actually
    calls for -- 14 Replica classes hold 0.017% of the voxels but 61% of the
    summed weight, and the rarest (81 voxels in the whole training set) cannot
    be learned from that no matter how hard it is weighted.
    """
    w = w.clone()
    w[~present] = 0.0
    w[present] = w[present] / w[present].mean()
    if clamp is not None and clamp > 0:
        w[present] = w[present].clamp(1.0 / clamp, clamp)
        w[present] = w[present] / w[present].mean()
    if max_ratio is not None and max_ratio > 0:
        cap = w[present].median() * max_ratio
        w[present] = w[present].clamp(max=cap)
        w[present] = w[present] / w[present].mean()
    return w


def _count_volume(files, src, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for fname in tqdm.tqdm(files, desc="counting (volume)"):
        labels = torch.load(os.path.join(src, fname), map_location="cpu")["labels"]
        counts += torch.bincount(labels.flatten().long(), minlength=num_classes)
    return counts


def _count_loss(files, src, num_classes, patch_size, depth, full_depth, device):
    """Count classes the way `compute_semantic_loss` sees them.

    The loss is not evaluated over the whole volume — only over the 2x2 voxel
    patch under each depth-6 octree node, i.e. inside occupied regions. That
    distribution is very different from the volume's (free space is ~64% rather
    than ~94%), and it is the one the weights should be derived from.
    """
    from octree_diff.octree.util_octree_stuff import (
        get_non_empty_mask, points2octree, voxel_grid_to_points)

    counts = torch.zeros(num_classes, dtype=torch.long)
    off_x = torch.tensor([0, 0, 1, 1], device=device)
    off_y = torch.tensor([0, 1, 0, 1], device=device)

    for fname in tqdm.tqdm(files, desc="counting (loss)"):
        labels = torch.load(os.path.join(src, fname), map_location="cpu")["labels"]
        labels = labels.long().permute(1, 2, 0).contiguous().to(device)

        mask = get_non_empty_mask(labels, patch_size)
        if mask.sum() == 0:
            continue
        octree = points2octree(voxel_grid_to_points(mask), depth=depth,
                               full_depth=full_depth).to(device)

        x, y, z, _ = octree.xyzb(depth, nempty=False)
        x, y, z = x.long(), y.long(), z.long()
        tx = (x.view(-1, 1) * 2 + off_x.view(1, -1)).reshape(-1)
        ty = (y.view(-1, 1) * 2 + off_y.view(1, -1)).reshape(-1)
        tz = z.repeat_interleave(4)

        counts += torch.bincount(labels[tx, ty, tz].cpu(), minlength=num_classes)
    return counts


def compute(src, num_classes, include_free_space, basis="loss", clamp=0.0,
            max_ratio=0.0, min_count=0,
            patch_size=2, depth=6, full_depth=3, device="cuda"):
    files = sorted(f for f in os.listdir(src) if f.endswith(".pt"))
    if not files:
        raise RuntimeError(f"No .pt patches found under {src}")

    if basis == "loss":
        counts = _count_loss(files, src, num_classes, patch_size, depth, full_depth, device)
    else:
        counts = _count_volume(files, src, num_classes)

    stats = {"counts": counts, "num_classes": num_classes, "num_patches": len(files),
             "basis": basis, "clamp": clamp, "max_ratio": max_ratio,
             "min_count": min_count}

    tally = counts.clone()
    if not include_free_space:
        # Class 0 is free space and dominates by ~2 orders of magnitude; it is
        # still counted, just excluded from the weighting statistics.
        tally[0] = 0
    # Classes below `min_count` are excluded from the loss entirely. A class
    # with a few dozen voxels in the whole training set cannot be learned, and
    # inverse-frequency weighting hands exactly those classes the largest
    # weights -- so they contribute noise, not signal. cross_entropy with a
    # zero weight drops them from both numerator and denominator.
    present = tally >= max(int(min_count), 1)
    if min_count:
        dropped = int(((tally > 0) & ~present).sum())
        held = float(tally[(tally > 0) & ~present].sum()) / float(tally.sum())
        print(f"min_count={min_count}: dropping {dropped} classes "
              f"holding {held * 100:.4f}% of counted voxels")

    freq = tally.double() / tally.sum().double()
    stats["freq"] = freq.float()

    inv = torch.zeros(num_classes, dtype=torch.double)
    inv[present] = 1.0 / freq[present]
    stats["weights_inv"] = _normalise(inv, present, clamp, max_ratio).float()

    inv_sqrt = torch.zeros(num_classes, dtype=torch.double)
    inv_sqrt[present] = 1.0 / freq[present].sqrt()
    stats["weights_inv_sqrt"] = _normalise(inv_sqrt, present, clamp, max_ratio).float()

    beta = 0.9999
    enet = torch.zeros(num_classes, dtype=torch.double)
    enet[present] = (1.0 - beta) / (1.0 - beta ** tally[present].double())
    stats["weights_enet"] = _normalise(enet, present, clamp, max_ratio).float()
    stats["enet_beta"] = beta
    stats["include_free_space"] = include_free_space

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/patches_128x128x64_s16")
    parser.add_argument("--out", default="data/replica_class_counts.pt")
    parser.add_argument("--num_classes", type=int, default=92)
    parser.add_argument("--basis", default="loss", choices=["loss", "volume"],
                        help="'loss' counts only what compute_semantic_loss sees "
                             "(inside occupied octree nodes); 'volume' counts every voxel")
    parser.add_argument("--clamp", type=float, default=0.0,
                        help="symmetric cap to [1/clamp, clamp]; 0 disables. Off by "
                             "default -- it flattens the head on this distribution, "
                             "see _normalise")
    parser.add_argument("--max-ratio", dest="max_ratio", type=float, default=0.0,
                        help="cap any weight at max_ratio x the median present "
                             "weight; 0 disables. Caps the tail only")
    parser.add_argument("--min-count", dest="min_count", type=int, default=0,
                        help="classes with fewer than this many voxels get weight "
                             "0 and are excluded from the loss; 0 keeps all")
    parser.add_argument("--no_free_space", action="store_true",
                        help="exclude class 0 from the weighting (gives it weight 0 — "
                             "only sensible on the 'volume' basis)")
    args = parser.parse_args()

    stats = compute(args.src, args.num_classes, not args.no_free_space,
                    basis=args.basis, clamp=args.clamp, max_ratio=args.max_ratio,
                    min_count=args.min_count)
    torch.save(stats, args.out)

    counts = stats["counts"]
    order = counts.argsort(descending=True)
    print(f"\nsaved {args.out}   ({stats['num_patches']} patches, "
          f"{int(counts.sum())} voxels, {int((counts > 0).sum())}/{args.num_classes} classes present)")
    print("\n  most common            least common (of those present)")
    present = (counts > 0).nonzero().flatten()
    rare = counts[present].argsort()[:10]
    for i in range(10):
        c_hi = order[i].item()
        c_lo = present[rare[i]].item()
        print(f"  {c_hi:3d}: {counts[c_hi]:12d}   |   {c_lo:3d}: {counts[c_lo]:10d}"
              f"  (inv_sqrt w={stats['weights_inv_sqrt'][c_lo]:.2f})")


if __name__ == "__main__":
    main()
