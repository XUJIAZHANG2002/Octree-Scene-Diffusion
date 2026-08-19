"""Inline (matplotlib) visualisation for indoor label volumes.

Deliberately does not use Open3D: `utils/util_visual_stuff.py` blocks on a GUI
window, which is no use in a notebook running over SSH. Everything here renders
inline and returns a Figure.

Label ids in this dataset are a remapped space, not raw Replica class ids — the
three dominant classes were identified geometrically (id 1 sits at the bottom of
every patch, id 3 at the top, id 2 is thin and vertical), so only those are
named. The rest are shown by id.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

FLOOR, WALL, CEILING = 1, 2, 3
STRUCTURE_NAMES = {0: "empty", FLOOR: "floor", WALL: "wall", CEILING: "ceiling"}


def class_colors(num_classes=92, seed=0):
    """Stable per-class RGB palette. Structure classes get fixed, readable colours."""
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.25, 0.95, size=(num_classes, 3))
    colors[0] = (1.0, 1.0, 1.0)
    colors[FLOOR] = (0.55, 0.45, 0.35)
    colors[WALL] = (0.80, 0.78, 0.72)
    colors[CEILING] = (0.62, 0.68, 0.78)
    return colors


def label_name(i):
    return STRUCTURE_NAMES.get(i, f"class {i}")


def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def plot_3d(labels, ax=None, title=None, hide=(CEILING,), max_points=20000,
            elev=28, azim=-60, size=2.0, colors=None):
    """3-D scatter of occupied voxels, ceiling hidden by default so you can see in."""
    lab = _to_np(labels)
    colors = class_colors(max(int(lab.max()) + 1, 92)) if colors is None else colors

    keep = lab > 0
    for h in hide:
        keep &= lab != h
    xs, ys, zs = np.nonzero(keep)
    cs = lab[xs, ys, zs]

    if len(xs) > max_points:
        sel = np.random.default_rng(0).choice(len(xs), max_points, replace=False)
        xs, ys, zs, cs = xs[sel], ys[sel], zs[sel], cs[sel]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), subplot_kw={"projection": "3d"})
    ax.scatter(xs, ys, zs, c=colors[cs], s=size, marker="s", linewidths=0)
    ax.set_xlim(0, lab.shape[0]); ax.set_ylim(0, lab.shape[1]); ax.set_zlim(0, lab.shape[2])
    ax.set_box_aspect((lab.shape[0], lab.shape[1], lab.shape[2]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if title:
        ax.set_title(f"{title}\n{int(keep.sum())} voxels", fontsize=10)
    return ax


def plot_slice(labels, z, ax=None, title=None, colors=None):
    """Horizontal cross-section at height z — the clearest view of walls/layout."""
    lab = _to_np(labels)
    colors = class_colors(max(int(lab.max()) + 1, 92)) if colors is None else colors
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(colors[lab[:, :, z]], origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f"z = {z}", fontsize=10)
    return ax


def plot_state(state, ax=None, z=None, title="sensor visibility"):
    """The tri-state visibility volume: unknown / free / observed surface."""
    st = _to_np(state)
    z = st.shape[2] // 3 if z is None else z
    palette = np.array([[0.15, 0.15, 0.18],     # -1 unknown
                        [0.85, 0.90, 0.95],     #  0 free
                        [0.90, 0.35, 0.25]])    #  1 observed
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(palette[st[:, :, z] + 1], origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    frac = (st != -1).mean()
    ax.set_title(f"{title}  (z={z})\n{frac:.0%} of the volume observed", fontsize=10)
    return ax


def legend_for(labels_list, ax, top=10, colors=None):
    """Legend of the most common non-empty classes across the given volumes."""
    counts = np.zeros(200, dtype=np.int64)
    for lab in labels_list:
        l = _to_np(lab).ravel()
        counts[: int(l.max()) + 1] += np.bincount(l, minlength=int(l.max()) + 1)
    counts[0] = 0
    colors = class_colors(200) if colors is None else colors
    order = counts.argsort()[::-1][:top]
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                          markerfacecolor=colors[i], markeredgecolor="none",
                          label=f"{label_name(i)}  ({counts[i]:,})")
               for i in order if counts[i] > 0]
    ax.legend(handles=handles, loc="center", frameon=False, fontsize=9, ncol=2)
    ax.axis("off")


def compare_row(volumes, titles, hide=(CEILING,), figsize=None, **kw):
    """Row of 3-D panels sharing one colour palette."""
    n = len(volumes)
    colors = class_colors(200)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5 * n, 4.5),
                             subplot_kw={"projection": "3d"})
    axes = np.atleast_1d(axes)
    for ax, vol, t in zip(axes, volumes, titles):
        plot_3d(vol, ax=ax, title=t, hide=hide, colors=colors, **kw)
    fig.tight_layout()
    return fig


def completion_metrics(pred, gt, known):
    """Accuracy split by whether the sensor could see the voxel."""
    pred, gt, known = _to_np(pred), _to_np(gt), _to_np(known)
    occ = gt > 0
    out = {}
    for name, sel in (("observed", occ & known), ("occluded", occ & ~known)):
        out[f"{name}_acc"] = float((pred[sel] == gt[sel]).mean()) if sel.any() else float("nan")
        out[f"{name}_n"] = int(sel.sum())
    p, g = pred > 0, gt > 0
    out["occupancy_iou"] = float((p & g).sum() / max((p | g).sum(), 1))
    hidden = ~known
    ph, gh = p & hidden, g & hidden
    out["occluded_occupancy_iou"] = float((ph & gh).sum() / max((ph | gh).sum(), 1))
    return out
