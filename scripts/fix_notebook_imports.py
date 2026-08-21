"""Rewrite first-party imports across the notebooks.

The outdoor notebooks were saved from a working tree with a flatter layout, so their
imports do not resolve against this repo:

  from util_octree_stuff import *          -> the module lives under utils/
  from util_sample_stuff import *          -> same
  from velodyne_to_voxel import ...        -> recovered into dataset/
  ...graph_densed_sem_vae import GraphVAE  -> renamed to graph_sem_vae in 19ece4d

Kept as a script rather than done by hand so it can be re-run after the package move.

    python scripts/fix_notebook_imports.py --check     # report only
    python scripts/fix_notebook_imports.py             # rewrite in place
"""

import argparse
import glob
import json
import re

# (pattern, replacement) applied line-wise to code cells
# Two migrations, applied in order:
#   1. bare module names, from the flat working tree the notebooks were saved in
#   2. the pre-package-move layout (utils/, models/networks/, dataset/, ...)
# Longest module paths first so a general rule cannot eat a specific one.
RULES = [
    # legacy bare imports
    (r"^(\s*)from util_octree_stuff import", r"\1from octree_diff.octree.util_octree_stuff import"),
    (r"^(\s*)from util_sample_stuff import", r"\1from octree_diff.diffusion.util_sample_stuff import"),
    (r"^(\s*)from util_dualoctree import", r"\1from octree_diff.octree.util_dualoctree import"),
    (r"^(\s*)from velodyne_to_voxel import", r"\1from octree_diff.data.kitti.velodyne_to_voxel import"),
    (r"graph_densed_sem_vae", r"graph_sem_vae"),
    # pre-package-move module paths
    (r"\bmodels\.networks\.diffusion_networks\.ldm_diffusion_util\b", "octree_diff.models.common.ldm_diffusion_util"),
    (r"\bmodels\.networks\.diffusion_networks\.graph_unet_hr\b", "octree_diff.models.semantic.graph_unet_hr"),
    (r"\bmodels\.networks\.diffusion_networks\.graph_unet_lr\b", "octree_diff.models.outdoor.graph_unet_lr"),
    (r"\bmodels\.networks\.dualoctree_networks\.graph_sem_vae\b", "octree_diff.models.semantic.graph_sem_vae"),
    (r"\bmodels\.networks\.dualoctree_networks\.dual_octree\b", "octree_diff.models.semantic.dual_octree"),
    (r"\bmodels\.structure_networks\.structure_vae\b", "octree_diff.models.structure.structure_vae"),
    (r"\bmodels\.structure_networks\.unet_3d\b", "octree_diff.models.structure.unet_3d"),
    (r"\bdataset\.velodyne_to_voxel\b", "octree_diff.data.kitti.velodyne_to_voxel"),
    (r"\binference_scripts\.inference_indoor\b", "octree_diff.inference.pipeline"),
    (r"\butils\.util_octree_stuff\b", "octree_diff.octree.util_octree_stuff"),
    (r"\butils\.util_dualoctree\b", "octree_diff.octree.util_dualoctree"),
    (r"\butils\.util_sample_stuff\b", "octree_diff.diffusion.util_sample_stuff"),
    (r"\butils\.sensor_sim\b", "octree_diff.inference.sensor_sim"),
    (r"\butils\.vis_indoor\b", "octree_diff.viz.vis_indoor"),
]


def rewrite_source(lines):
    out, changed = [], 0
    for line in lines:
        new = line
        for pat, rep in RULES:
            new = re.sub(pat, rep, new)
        changed += new != line
        out.append(new)
    return out, changed


def process(path, check):
    nb = json.load(open(path))
    total = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell["source"], n = rewrite_source(cell["source"])
        total += n
    if total and not check:
        # ensure_ascii=False and the trailing newline match how Jupyter writes the
        # file, so the diff shows the import changes and nothing else
        with open(path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="report without writing")
    p.add_argument("--glob", default="notebooks/**/*.ipynb")
    args = p.parse_args()

    grand = 0
    for path in sorted(glob.glob(args.glob, recursive=True)):
        n = process(path, args.check)
        grand += n
        if n:
            print(f"{'would fix' if args.check else 'fixed':10s} {n:3d} import lines  {path}")
    print(f"\n{grand} import lines {'need fixing' if args.check else 'rewritten'}")


if __name__ == "__main__":
    main()
