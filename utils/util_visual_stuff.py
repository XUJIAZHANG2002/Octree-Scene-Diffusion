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

    
def visualize_kitti_instance(voxel_grid):

    import numpy as np
    import open3d as o3d
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # === Step 0: Class Name Map ===
    class_names = [
        'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
        'person', 'bicyclist', 'motorcyclist', 'road', 'parking',
        'sidewalk', 'other-ground', 'building', 'fence',
        'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
    ]
    semkitti_colors = {
        0:  [1.0, 1.0, 1.0],        # empty (air) — white
        1:  [0.4, 0.6, 0.99],       # car — light blue
        2:  [0.2, 0.95, 0.95],      # bicycle — cyan
        3:  [1.0, 0.6, 0.0],        # motorcycle — orange
        4:  [0.4, 0.0, 0.6],        # truck — dark purple
        5:  [0.2, 0.2, 1.0],        # other-vehicle — blue
        6:  [1.0, 0.0, 0.0],        # person — red
        7:  [1.0, 0.0, 0.8],        # bicyclist — pink-red
        8:  [1.0, 0.0, 1.0],        # motorcyclist — fuchsia
        9:  [1.0, 0.0, 1.0],        # road — magenta
        10: [1.0, 0.6, 1.0],        # parking — light pink
        11: [0.2, 0.0, 0.4],        # sidewalk — dark purple
        12: [0.8, 0.0, 0.4],        # other-ground — dark red
        13: [1.0, 1.0, 0.0],        # building — yellow
        14: [1.0, 0.4, 0.2],        # fence — orange-red
        15: [0.0, 0.8, 0.0],        # vegetation — green
        16: [0.5, 0.3, 0.1],        # trunk — brown
        17: [0.6, 0.9, 0.4],        # terrain — light green
        18: [1.0, 0.9, 0.5],        # pole — pale yellow
        19: [1.0, 0.0, 0.2],        # traffic sign — bright red
        20: [0.5, 0.5, 0.5],       # invalid — gray
        255: [0.5, 0.5, 0.5],       # invalid — gray
    }
    label_id_to_name = {i+1: name for i, name in enumerate(class_names)}  # 1–19
    label_id_to_name[255] = "invalid"

    # === Step 1: Extract data and labels ===
    # grid_pred = voxel_pred[0].cpu().numpy()

    grid= voxel_grid.cpu().numpy()




    coords = np.argwhere(grid != 0)  # skip air (label 0)
    # coords = coords.permute(1, 0).contiguous()  # [3, N] → [N, 3]
    # coords = coords.cpu().numpy()
    labels = grid[coords[:, 0], coords[:, 1], coords[:, 2]]
    unique_vals = np.unique(labels)
    # print((grid == grid_gt[:, :, 0:32]).mean())
    # === Step 2: Assign colors ===
    label_to_color = {}
    cmap = plt.get_cmap("tab20b", 20)
    for val in unique_vals:
        if val == 255:
            label_to_color[val] = np.array([0.5, 0.5, 0.5])  # gray for invalid
        else:
            label_to_color[val] = np.array(semkitti_colors.get(val, [0.0, 0.0, 0.0]))

    # === Step 3: Build Open3D VoxelGrid ===
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = 1.0
    for (x, y, z), label in zip(coords, labels):
        voxel = o3d.geometry.Voxel()
        voxel.grid_index = np.array([x, y, z], dtype=int)
        voxel.color = label_to_color[label]
        voxel_grid.add_voxel(voxel)

    # === Step 4: Visualize voxel grid ===
    o3d.visualization.draw_geometries([voxel_grid], window_name="SemanticKITTI VoxelGrid")

    # === Step 5: Plot category-color legend ===
    legend_handles = []
    for val in unique_vals:
        if val == 0:
            continue  # skip air
        color = label_to_color[val]
        label = label_id_to_name.get(val, f"label_{val}")
        patch = Patch(facecolor=color, edgecolor='black', label=f"{val}: {label}")
        legend_handles.append(patch)

    plt.figure(figsize=(10, 6))
    plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0, 0.5))
    plt.axis('off')
    plt.title("Semantic Label → Color Legend")
    plt.show()
