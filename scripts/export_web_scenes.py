"""Export sample scenes to the compact binary format the web viewer reads.

Produces, per scene, three variants of the same volume so the page can toggle
between them under one camera:

    observed   what the simulated sensor saw   (labels masked by the known mask)
    completed  the model's completion          (the existing inpaint path)
    truth      ground truth

Completion goes through `octree_diff.inference.pipeline.run_inpaint`, the same
code `octree-diff complete` runs, rather than a reimplementation -- the point of
the page is to show what the shipped pipeline does.

Encoding: one .bin per variant, four uint8 per voxel (x, y, z, label). Voxels
whose six face-neighbours are all occupied are dropped: they cannot be seen from
outside the volume, so this is lossless for a surface renderer and removes most
of the payload. Records are sorted by label, which costs nothing and compresses
much better.

    python scripts/export_web_scenes.py --scenes 4 --out out/web
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# encoding

def surface_mask(occ):
    """Occupied voxels with at least one empty face-neighbour.

    Padding with False rather than np.roll matters: roll wraps, which would
    mark voxels on the volume boundary as interior and silently hollow out the
    floor and the walls.
    """
    p = np.zeros(tuple(n + 2 for n in occ.shape), dtype=bool)
    p[1:-1, 1:-1, 1:-1] = occ
    enclosed = (p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1] &
                p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1] &
                p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:])
    return occ & ~enclosed


def encode(vol):
    """(X, Y, Z) int volume -> (N, 4) uint8 records, sorted by label."""
    vol = np.asarray(vol)
    if any(n > 256 for n in vol.shape):
        raise ValueError(f"dims {vol.shape} do not fit in uint8 coordinates")
    if int(vol.max()) > 255:
        raise ValueError(f"label {int(vol.max())} does not fit in uint8")

    xs, ys, zs = np.nonzero(surface_mask(vol > 0))
    labels = vol[xs, ys, zs]
    order = np.lexsort((zs, ys, xs, labels))
    return np.stack([xs[order], ys[order], zs[order], labels[order]], 1).astype(np.uint8)


def decode(buf, dims):
    """Inverse of `encode`, for verification."""
    vol = np.zeros(dims, dtype=np.int32)
    vol[buf[:, 0], buf[:, 1], buf[:, 2]] = buf[:, 3]
    return vol


def check_roundtrip(vol, buf):
    """Every visible voxel survives exactly; every dropped one was fully enclosed."""
    vol = np.asarray(vol).astype(np.int32)
    occ = vol > 0
    vis = surface_mask(occ)
    back = decode(buf, vol.shape)

    assert np.array_equal(back[vis], vol[vis]), "visible voxels did not round-trip"
    assert not back[~vis].any(), "decoded a voxel that should have been dropped"
    dropped = occ & ~vis
    assert not surface_mask(occ)[dropped].any(), "dropped a voxel with an exposed face"
    return int(vis.sum()), int(dropped.sum())


# --------------------------------------------------------------------------
# completion

def complete(pipe, P, labels_path, known_path, out_dir, args):
    """Run the shipped inpaint path and return its completed label volume."""
    import torch

    tmp = tempfile.mkdtemp(prefix="webexport_")
    try:
        ns = types.SimpleNamespace(
            input=labels_path, known_mask=known_path,
            steps=args.steps, eta=0.0, repaint_resample=1, best_of=args.best_of)
        P.run_inpaint(pipe, ns, tmp)
        return torch.load(os.path.join(tmp, "labels_completed.pt"),
                          weights_only=True).squeeze().cpu().numpy().astype(np.int32)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def accuracy(pred, truth, known):
    """Semantic accuracy split by visibility, on two bases.

    `occupied` counts only voxels that are occupied in the ground truth -- the
    basis `octree-diff verify` uses, and the only honest one to quote. `all`
    counts every voxel including free space, which is ~92% of the volume and so
    reports ~0.9 no matter how the semantics turn out. Both are recorded here
    precisely so the second is never mistaken for the first.
    """
    occ = truth > 0
    out = {}
    for basis, base in (("occupied", occ), ("all", np.ones_like(occ))):
        for name, vis in (("observed", known), ("occluded", ~known)):
            sel = base & vis
            n = int(sel.sum())
            out[f"{basis}_{name}"] = (round(float((pred[sel] == truth[sel]).mean()), 4)
                                      if n else None)
    out["occupied_voxels"] = {"observed": int((occ & known).sum()),
                              "occluded": int((occ & ~known).sum())}
    inter = int(((pred > 0) & occ).sum())
    union = int((((pred > 0) | occ)).sum())
    out["occupancy_iou"] = round(inter / union, 4) if union else None
    return out


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", default=os.path.join(REPO, "data", "sample_scenes"))
    ap.add_argument("--out", default=os.path.join(REPO, "out", "web"))
    ap.add_argument("--scenes", type=int, default=4, help="how many scenes to export")
    ap.add_argument("--only", nargs="+", help="scene ids, overrides --scenes")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--best-of", dest="best_of", type=int, default=3,
                    help="rejects the ~1-in-20 structure draw that degrades")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run completions even if cached")
    args = ap.parse_args(argv)

    import torch
    from octree_diff.viz.vis_indoor import CLASS_NAMES, class_colors

    index = yaml.safe_load(open(os.path.join(args.sample_dir, "index.yaml")))
    scenes = index["scenes"]
    scenes = ([s for s in scenes if s["id"] in args.only] if args.only
              else scenes[:args.scenes])
    if not scenes:
        sys.exit("no scenes selected")

    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "_completions")
    os.makedirs(cache, exist_ok=True)

    need_model = args.force or any(
        not os.path.exists(os.path.join(cache, f"{s['id']}.npy")) for s in scenes)

    pipe = P = None
    if need_model:
        from octree_diff.inference import pipeline as P
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed)
        print("loading models...")
        pipe = P.IndoorPipeline(device)
        print(f"  latent scales: structure {pipe.str_scale:.4f}, "
              f"semantic {pipe.sem_scale:.4f}")

    dims = None
    out_scenes = []

    for meta in scenes:
        sid = meta["id"]
        lp = os.path.join(args.sample_dir, f"{sid}_labels.pt")
        kp = os.path.join(args.sample_dir, f"{sid}_known.pt")
        truth = torch.load(lp, weights_only=True).squeeze().numpy().astype(np.int32)
        known = torch.load(kp, weights_only=True).squeeze().numpy().astype(bool)
        dims = dims or list(truth.shape)
        assert list(truth.shape) == dims, f"{sid}: inconsistent dims"

        cached = os.path.join(cache, f"{sid}.npy")
        if os.path.exists(cached) and not args.force:
            done = np.load(cached)
            print(f"{sid}: completion from cache")
        else:
            print(f"{sid}: completing ({meta['volume_observed']:.0%} of the volume observed)...")
            torch.manual_seed(args.seed)
            done = complete(pipe, P, lp, kp, args.out, args)
            np.save(cached, done)

        acc = accuracy(done, truth, known)
        print(f"  semantic acc (occupied voxels)  observed {acc['occupied_observed']}"
              f"  occluded {acc['occupied_occluded']}")
        print(f"  occupancy IoU {acc['occupancy_iou']}")

        variants = {}
        for name, vol in (("observed", np.where(known, truth, 0)),
                          ("completed", done),
                          ("truth", truth)):
            buf = encode(vol)
            vis, dropped = check_roundtrip(vol, buf)
            fn = f"{sid}_{name}.bin"
            buf.tofile(os.path.join(args.out, fn))
            variants[name] = {"file": fn, "count": int(len(buf))}
            print(f"  {name:10s} {vis:7,} visible  ({dropped:,} enclosed dropped)  "
                  f"{len(buf) * 4 / 1024:6.0f} KB")

        out_scenes.append({
            "id": sid,
            "scene": meta["replica_scene"],
            "occupied_voxels": meta["occupied_voxels"],
            "classes": meta["classes"],
            "volume_observed": meta["volume_observed"],
            "accuracy": acc,
            "variants": variants,
        })

    palette = (class_colors(len(CLASS_NAMES)) * 255).round().astype(int).tolist()
    manifest = {
        "dataset": "replica",
        "dims": dims,
        "axes": "x, y, z -- z is up",
        "voxel_size_m": index["voxel_size_m"],
        "record": "uint8 x, y, z, label",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "palette": palette,
        "structure_classes": {"floor": 1, "wall": 2, "ceiling": 3},
        "sampling": {"steps": args.steps, "best_of": args.best_of, "seed": args.seed},
        "scenes": out_scenes,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")

    total = sum(v["count"] * 4 for s in out_scenes for v in s["variants"].values())
    print(f"\nwrote {args.out}/  --  {len(out_scenes)} scenes, {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
