"""Simulate what a sensor at a viewpoint can actually see inside a voxel scene.

The KITTI notebooks build their observed/unobserved mask by reverse ray-tracing
from the LiDAR origin. There is no indoor equivalent in the repo, so this is it:
cast rays from a viewpoint through the voxel grid and mark each voxel free,
occupied-and-seen, or unknown (occluded / out of range / out of field of view).

That tri-state is what turns a complete scene into a scene-completion problem.
"""

import numpy as np
import torch


def ray_directions(n_az=720, n_el=160, fov_v=(-35.0, 35.0), fov_h=(-180.0, 180.0)):
    """Unit direction vectors on an azimuth x elevation grid. Returns [R, 3]."""
    az = np.deg2rad(np.linspace(fov_h[0], fov_h[1], n_az, endpoint=False))
    el = np.deg2rad(np.linspace(fov_v[0], fov_v[1], n_el))
    az, el = np.meshgrid(az, el, indexing="ij")
    ce = np.cos(el)
    return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], -1).reshape(-1, 3)


def simulate_sensor(occupancy, origin, n_az=720, n_el=160, fov_v=(-35.0, 35.0),
                    fov_h=(-180.0, 180.0), max_range=None, step=0.5, device="cpu"):
    """March rays from `origin` through `occupancy` and return a visibility state.

    Args:
        occupancy: [X, Y, Z] bool/int tensor, True where a voxel is solid.
        origin:    (x, y, z) viewpoint in voxel coordinates.
        max_range: in voxels; defaults to the grid diagonal.
        step:      march increment in voxels. 0.5 is fine for a grid this size.

    Returns:
        [X, Y, Z] int8 tensor:  -1 unknown, 0 free, 1 occupied and observed.
    """
    occ = torch.as_tensor(occupancy).to(device).bool()
    X, Y, Z = occ.shape
    if max_range is None:
        max_range = float(np.sqrt(X ** 2 + Y ** 2 + Z ** 2))

    dirs = torch.as_tensor(ray_directions(n_az, n_el, fov_v, fov_h),
                           dtype=torch.float32, device=device)
    org = torch.tensor(origin, dtype=torch.float32, device=device)

    state = torch.full((X, Y, Z), -1, dtype=torch.int8, device=device)
    alive = torch.ones(dirs.shape[0], dtype=torch.bool, device=device)
    bounds = torch.tensor([X, Y, Z], device=device)

    for t in np.arange(step, max_range, step):
        if not alive.any():
            break
        p = org + dirs[alive] * float(t)
        v = p.floor().long()

        inside = ((v >= 0).all(1) & (v < bounds).all(1))
        # a ray that has left the grid is done
        idx_alive = alive.nonzero(as_tuple=True)[0]
        alive[idx_alive[~inside]] = False
        v = v[inside]
        if v.numel() == 0:
            continue

        hit = occ[v[:, 0], v[:, 1], v[:, 2]]

        # free space up to the first surface
        free = v[~hit]
        state[free[:, 0], free[:, 1], free[:, 2]] = torch.maximum(
            state[free[:, 0], free[:, 1], free[:, 2]], torch.tensor(0, dtype=torch.int8, device=device))

        # the surface itself is observed, and the ray stops there
        seen = v[hit]
        state[seen[:, 0], seen[:, 1], seen[:, 2]] = 1
        alive[idx_alive[inside][hit]] = False

    return state


def pick_viewpoint(occupancy, height_frac=0.4, search_frac=0.35):
    """A free voxel near the middle of the scene, at roughly eye height.

    Picks the free voxel with the most free neighbours in a central box, so the
    sensor does not end up wedged inside furniture.
    """
    occ = torch.as_tensor(occupancy).bool()
    X, Y, Z = occ.shape
    z = int(Z * height_frac)

    x0, x1 = int(X * (0.5 - search_frac)), int(X * (0.5 + search_frac))
    y0, y1 = int(Y * (0.5 - search_frac)), int(Y * (0.5 + search_frac))

    # openness = free voxels in a small column around each candidate
    sl = (~occ[x0:x1, y0:y1, max(z - 4, 0):z + 5]).float()
    openness = sl.mean(-1)

    # prefer the middle of the room: an open corner sees very little of the scene
    dev = openness.device
    gx = torch.linspace(-1, 1, openness.shape[0], device=dev).unsqueeze(1)
    gy = torch.linspace(-1, 1, openness.shape[1], device=dev).unsqueeze(0)
    centrality = 1.0 - (gx ** 2 + gy ** 2).sqrt().clamp(max=1.0)

    score = openness + 0.5 * centrality
    score[occ[x0:x1, y0:y1, z]] = -1.0               # must itself be free

    flat = int(score.argmax())
    dx, dy = flat // score.shape[1], flat % score.shape[1]
    if score[dx, dy] <= 0:
        raise ValueError("no free voxel found at that height — try another height_frac")
    return (x0 + int(dx), y0 + int(dy), z)


def observed_labels(labels, state):
    """Apply a visibility state to a label volume: unknown voxels become 0."""
    known = state != -1
    return torch.where(known, labels, torch.zeros_like(labels)), known
