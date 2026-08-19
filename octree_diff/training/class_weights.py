import os

import torch


def load_class_weights(scheme, path, num_classes):
    """Per-class weights for the semantic cross-entropy.

    `scheme` is one of:
        none      uniform (what round 1 trained with)
        inv_sqrt  1 / sqrt(freq)
        enet      effective number of samples (Cui et al. 2019)
        inv       1 / freq

    Weights come from the file written by `dataset/compute_class_weights.py`.
    Classes absent from the training set get weight 0, so they cannot contribute
    a gradient for a label that never occurs.
    """
    scheme = (scheme or "none").lower()
    if scheme in ("none", "uniform"):
        print("class weights: uniform")
        return torch.ones(num_classes)

    key = f"weights_{scheme}"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python -m octree_diff.data.compute_class_weights` first")

    stats = torch.load(path, map_location="cpu", weights_only=False)
    if key not in stats:
        raise KeyError(f"{key} not in {path}; available: "
                       f"{[k for k in stats if k.startswith('weights_')]}")

    w = stats[key].float()
    if w.numel() != num_classes:
        raise ValueError(f"{key} has {w.numel()} entries, expected {num_classes}")

    present = w > 0
    print(f"class weights: {scheme} from {path} "
          f"(basis={stats.get('basis', 'volume')}, clamp={stats.get('clamp')})")
    print(f"  {int(present.sum())}/{num_classes} classes weighted, "
          f"range {w[present].min():.3f}-{w[present].max():.3f}, "
          f"free space (class 0) = {w[0]:.3f}")
    return w
