"""Build the structural (occupancy) dataset used by the Structure VAE / Diffusion stages.

Reads the semantic voxel patches (the same ones the semantic stage trains on),
derives the occupancy octree and stores its dense split representation.

For indoor scenes (patch_size = 2) a [128, 128, 64] label patch reduces to a
[64, 64, 64] non-empty mask, i.e. a depth-6 octree. The split at full_depth 4 is
[1, 8, 16, 16, 16] which expands to a [1, 32, 32, 32] dense grid with values in
{-1, +1}.

    python -m octree_diff.data.build_split_dataset --src data/patches_128x128x64_s16 \
        --dst data/split_outputs --patch_size 2
"""

import argparse
import os

import torch
import tqdm

from octree_diff.octree.util_dualoctree import octree2split_small
from octree_diff.octree.util_octree_stuff import (
    get_non_empty_mask,
    points2octree,
    split2splitbig,
    voxel_grid_to_points,
)


def build(src, dst, patch_size, depth, full_depth, split_depth, device):
    os.makedirs(dst, exist_ok=True)

    files = sorted(f for f in os.listdir(src) if f.endswith(".pt"))
    if not files:
        raise RuntimeError(f"No .pt patches found under {src}")

    skipped = []
    for fname in tqdm.tqdm(files, desc="split"):
        blob = torch.load(os.path.join(src, fname), map_location="cpu")
        # (D, H, W) -> (H, W, D), matching VoxelPatchDataset
        labels = blob["labels"].long().permute(1, 2, 0).contiguous()

        mask = get_non_empty_mask(labels.to(device), patch_size)
        if mask.sum() == 0:
            skipped.append(fname)
            continue

        points = voxel_grid_to_points(mask)
        octree = points2octree(points, depth=depth, full_depth=full_depth).to(device)

        split = octree2split_small(octree, split_depth)
        split_big = split2splitbig(split)  # [1, 1, 2L, 2L, 2L], values in {-1, +1}

        torch.save(split_big.squeeze(0).cpu(), os.path.join(dst, fname))

    print(f"wrote {len(files) - len(skipped)} files to {dst}")
    if skipped:
        print(f"skipped {len(skipped)} empty patches: {skipped[:10]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/patches_128x128x64_s16")
    parser.add_argument("--dst", default="data/split_outputs")
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--full_depth", type=int, default=4)
    parser.add_argument("--split_depth", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    build(args.src, args.dst, args.patch_size, args.depth,
          args.full_depth, args.split_depth, args.device)


if __name__ == "__main__":
    main()
