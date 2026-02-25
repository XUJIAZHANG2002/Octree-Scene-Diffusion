# visualize_voxel_labels.py
import numpy as np
import open3d as o3d
import hashlib
import torch

def visualize_voxel_labels(labels: torch.Tensor, voxel_size=0.05, origin=(0,0,0)):
    """
    Visualize a (64,128,128) voxel tensor with semantic IDs.
    Args:
        labels: torch.Tensor (64,128,128) int
        voxel_size: float (meters)
        origin: tuple (x,y,z) world coords for (0,0,0)
    """
    labels = labels.cpu().numpy()
    Dz, Hy, Wx = labels.shape

    vg = o3d.geometry.VoxelGrid()
    vg.voxel_size = float(voxel_size)
    vg.origin = np.array(origin, float)

    def color_for_id(k: int):
        h = hashlib.md5(str(int(k)).encode()).digest()
        rgb = np.array([h[0], h[7], h[14]], float) / 255.0
        return 0.2 + 0.7 * rgb

    idx = np.argwhere(labels > 0)  # (N,3): z,y,x
    gids = labels[tuple(idx.T)]

    for (x,y,z), gid in zip(idx, gids):
        vx = o3d.geometry.Voxel()
        vx.grid_index = np.array([int(x), int(y), int(z)], dtype=int)
        vx.color = color_for_id(int(gid))
        vg.add_voxel(vx)

    print(f"Visualizing voxel grid: shape={labels.shape}, filled={len(idx)}")
    o3d.visualization.draw_geometries([vg])
