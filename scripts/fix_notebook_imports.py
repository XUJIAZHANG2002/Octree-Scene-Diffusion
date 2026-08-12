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
RULES = [
    (r"^(\s*)from util_octree_stuff import", r"\1from utils.util_octree_stuff import"),
    (r"^(\s*)from util_sample_stuff import", r"\1from utils.util_sample_stuff import"),
    (r"^(\s*)from velodyne_to_voxel import", r"\1from dataset.velodyne_to_voxel import"),
    (r"graph_densed_sem_vae", r"graph_sem_vae"),
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
