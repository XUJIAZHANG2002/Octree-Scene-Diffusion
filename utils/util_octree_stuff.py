import torch
import os
import numpy as np
import torch
import ocnn
from ocnn.octree import Points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from ocnn.octree import Points
import math
def voxel_grid_to_points(voxel_grid, use_semantic_octree = False, features=None):
    '''
    Converts a binary voxel grid into a Points object for octree construction.

    Args:
      voxel_grid (Tensor): shape (D, H, W), values 0 or 1 indicating occupancy
      features (Tensor, optional): shape (D, H, W, C) with per-voxel features

    Returns:
      Points: with `points` in [-1, 1]^3 and optional `features`
    '''
    if isinstance(voxel_grid, np.ndarray):
        voxel_grid = torch.from_numpy(voxel_grid)
    assert voxel_grid.ndim == 3, "Input voxel grid must be 3D"
    D, H, W = voxel_grid.shape
    assert D == H == W, "Assumes cubic voxel grid"

    # Get occupied voxel coordinates
    if not use_semantic_octree:
        occupied = (voxel_grid > 0)
    else:
        occupied = (voxel_grid >= 0)
    coords = occupied.nonzero(as_tuple=False).float()  # shape (N, 3)
    voxel_size = 1.0 / D  # Each voxel occupies this space
    coords = coords / D * 2 - 1 + voxel_size  # Centered in each cell
    if coords.shape[0] == 0:
        raise ValueError("No occupied voxels found!")

    # Extract features if provided
    feats = None
    if features is not None:
        assert features.shape[:3] == voxel_grid.shape
        feats = features[occupied].float()  # shape (N, C)

    return Points(points=coords, features=feats)

def points2octree(points,depth=4, full_depth=2):
    octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth)
    octree.build_octree(points)
    return octree


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

def reconstruct_voxel_from_patch(sem_voxs, octree, depth, shape=(1, 128, 128, 64)):
    B, H, W, D = shape
    device = sem_voxs[depth].device
    logits = sem_voxs[depth]
    
    # 1. Get coordinates directly (the "enlightened" way)
    x, y, z, b = octree.xyzb(depth, nempty=False)
    x, y, z = x.long(), y.long(), z.long()

    # 2. Get predictions and handle the patch
    # Assuming sem_voxs[depth] is [N, C, 2, 2] or [N, C, 4, 4]
    # We take the argmax first to save memory
    preds = logits.argmax(dim=1) # [N, 2, 2]
    h_p, w_p = preds.shape[1], preds.shape[2]
    
    # 3. Vectorized coordinate expansion
    # Replicating your scale-by-2 logic for XY
    off_x = torch.arange(h_p, device=device)
    off_y = torch.arange(w_p, device=device)
    
    # Calculate global grid coordinates using broadcasting
    # (x*2 + offset)
    grid_x = (x.view(-1, 1, 1) * 2 + off_x.view(1, -1, 1)).expand(-1, h_p, w_p)
    grid_y = (y.view(-1, 1, 1) * 2 + off_y.view(1, 1, -1)).expand(-1, h_p, w_p)
    grid_z = z.view(-1, 1, 1).expand(-1, h_p, w_p)

    # 4. Paint the voxel grid
    recon_voxel = torch.zeros((H, W, D), dtype=torch.long, device=device)
    recon_voxel[grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)] = preds.reshape(-1)

    return recon_voxel.unsqueeze(0)
def assign_octree_patch_features(vox_patches, octree, depth, patch_size=2):
    """
    Simplified: Map Octree nodes to patches directly.
    vox_patches: [H_grid, W_grid, D_grid, patch_size, patch_size]
    """
    device = vox_patches.device
    
    # 1. Get existing node coordinates directly from octree
    # x, y, z, b are tensors of shape [N_nodes_at_depth]
    x, y, z, b = octree.xyzb(depth, nempty=False)
    x, y, z, b = x.long(), y.long(), z.long(), b.long()

    # 2. Sample the patches at these specific coordinates
    # vox_patches is [B, 32, 32, 32, 2, 2] -> we use (b, x, y, z) to index
    # If vox_patches is [32, 32, 32, 2, 2] without batch dim, adjust accordingly
    valid_feats = vox_patches[x, y, z] 

    # 3. Create and assign the feature tensor
    # We ensure the dtype and device match the sampled features
    num_nodes = octree.nnum[depth]
    features = torch.zeros((num_nodes, patch_size, patch_size), 
                           device=device, dtype=valid_feats.dtype)
    
    # The indices for these nodes are simply 0 to N-1 because we queried them in order
    features[:len(valid_feats)] = valid_feats
    
    octree.features[depth] = features

import os
import numpy as np
import random

def load_random_voxel_instance(base_dir="dataset/chunk_161616"):
    # Randomly select mansion and floor
    mansion_id = random.randint(0, 9)
    floor_id = random.randint(0, 2)
    
    folder_path = os.path.join(base_dir, f"mansion{mansion_id}", f"floor{floor_id}")
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # List all .npy files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    if not files:
        raise FileNotFoundError(f"No .npy files in {folder_path}")

    # Randomly pick one file
    filename = random.choice(files)
    full_path = os.path.join(folder_path, filename)
    voxel_np = np.load(full_path).astype(np.int8)  # shape (16, 16, 16)

    voxel_tensor = torch.from_numpy(voxel_np).unsqueeze(0).long()  # [1, 16, 16, 16]
    return voxel_tensor[0], filename, folder_path
def load_random_voxel_instance_large(base_dir="dataset/chunk_161616"):
    # Randomly select mansion and floor
    mansion_id = random.randint(0, 9)
    floor_id = 0
    
    folder_path = os.path.join(base_dir, f"mansion{mansion_id}", f"floor{floor_id}")
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # List all .npy files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    if not files:
        raise FileNotFoundError(f"No .npy files in {folder_path}")

    # Randomly pick one file
    filename = random.choice(files)
    full_path = os.path.join(folder_path, filename)
    voxel_np = np.load(full_path).astype(np.int8)  # shape (16, 16, 16)

    voxel_tensor = torch.from_numpy(voxel_np).unsqueeze(0).long()  # [1, 16, 16, 16]
    return voxel_tensor[0], filename, folder_path
def octreeTovoxel(octree, depth=4, grid_size=16, assign_features = True,device='cuda'):
    import torch
    import itertools

    # Step 1: 构建全部 [x, y, z, b] 坐标
    coords = torch.tensor(
        list(itertools.product(range(grid_size), range(grid_size), range(grid_size))),
        dtype=torch.int32, device=device
    )  # [4096, 3]
    batch = torch.zeros((coords.size(0), 1), dtype=torch.int32, device=device)
    xyzb = torch.cat([coords, batch], dim=1)  # [4096, 4]

    # Step 2: 查询 octree
    idxs = octree.search_xyzb(xyzb, depth=depth, nempty=True)  # [4096]

    # Step 3: 准备 feature tensor
    if assign_features:
        features = torch.zeros((grid_size ** 3,), dtype=torch.long, device=device)
        valid = idxs != -1
        features[valid] = octree.features[depth][idxs[valid], 0].long()
    else:
        features = torch.zeros((grid_size ** 3,), dtype=torch.long, device=device)
        valid = idxs != -1
        features[valid] = 1     
    # Step 4: reshape 为 voxel tensor
    voxel_tensor = features.view(grid_size, grid_size, grid_size)  # [D, H, W]
    return voxel_tensor
    # # Iterate over all (i, j, k) positions
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         for k in range(grid_size):
    #             query = torch.tensor([[ i, j, k,0]], dtype=torch.int32, device=device)  # (b, x, y, z)
    #             idx = octree.search_xyzb(query, depth=depth, nempty=True)
    #             if idx.item() != -1:
    #                 voxel_tensor[i, j, k] = octree.features[depth][idx.item()][0].long()
import matplotlib.pyplot as plt
import numpy as np

def visualize_voxel_prediction(voxel_tensor_pred, save_path="voxel_vis.png", vis=True):
    """
    Visualize and/or save predicted voxel tensor using matplotlib 3D voxels.
    
    :param voxel_tensor_pred: torch.Tensor of shape [16, 16, 16], dtype long
    :param save_path: path to save the image
    :param vis: whether to display the figure with plt.show()
    """
    voxel_np = voxel_tensor_pred.cpu().numpy()
    voxel_np = np.transpose(voxel_np, (0, 2, 1))  # [Z, X, Y]

    mask = voxel_np != 0

    # Colormap: tab20b, 31 classes
    cmap = plt.cm.get_cmap("tab20b", 31)
    facecolors = np.zeros(voxel_np.shape + (4,), dtype=np.float32)
    facecolors[mask] = cmap(voxel_np[mask] % 31)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(mask, facecolors=facecolors, edgecolor='k', linewidth=0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.title("Predicted Semantic Voxel Grid")
    plt.tight_layout()

    plt.savefig(save_path)
    if vis:
        plt.show()
    else:
        plt.close(fig)
def compute_octree_loss(logits, octree_out):
    import torch
    import torch.nn.functional as F
    weights = [1.0] * 16
    output = dict()
    for d in logits.keys():
        logitd = logits[d]
        label_gt = octree_out.nempty_mask(d).long()
        output['loss_%d' % d] = F.cross_entropy(logitd, label_gt) * weights[d]
        output['accu_%d' % d] = logitd.argmax(1).eq(label_gt).float().mean()
    return output
def visualize_vae_output(output, voxel_size = 16, depth=4):
    sem_voxs = output['sem_voxs']
    octree_out = output['octree_out']

    D, H, W = voxel_size, voxel_size, voxel_size
    voxel_pred = torch.zeros((D, H, W), dtype=torch.long).cuda()

    coords = torch.stack(torch.meshgrid(
        torch.arange(D),
        torch.arange(H),
        torch.arange(W),
        indexing='ij'), dim=-1).reshape(-1, 3).to(octree_out.device)  # [4096, 3]

    batch = torch.zeros((coords.size(0), 1), dtype=torch.int32, device=coords.device)
    xyzb = torch.cat([coords.int(), batch], dim=1)  # [N, 4]

    idxs = octree_out.search_xyzb(xyzb, depth=depth, nempty=False)  # [N]
    valid = idxs != -1
    valid_idxs = idxs[valid].long()
    coords_valid = coords[valid]
    pred_classes = sem_voxs[depth][valid_idxs].argmax(dim=1)

    voxel_pred[coords_valid[:, 0], coords_valid[:, 1], coords_valid[:, 2]] = pred_classes

    # === Visualize ===
    os.makedirs("log/img", exist_ok=True)
    vis_path = f"log/img/test.png"
    visualize_voxel_prediction(voxel_pred.cpu(), save_path=vis_path, vis=True)


def precompute_semantic_target(self, voxel_gt_volume, depth):
    # Get the coordinates of all nodes at 'depth'
    # xyzb shape: [N, 4] -> (x, y, z, batch)
    xyzb = self.xyzb(depth, nempty=False)
    
    # Direct indexing into the 128x128x64 volume
    # This replaces the meshgrid/search logic
    x, y, z = xyzb[:, 0].long(), xyzb[:, 1].long(), xyzb[:, 2].long()
    
    # Store this target inside the octree object
    self.semantic_target = voxel_gt_volume[0, x, y, z] # [N]
import torch
import torch.nn.functional as F



def compute_semantic_loss_clean(
    sem_voxs, octree, voxel_tensor, class_weights, depth=6
):
    device = voxel_tensor.device
    logits = sem_voxs[depth] 
    N, C, H, W = logits.shape

    # === Step 1: Unpack the tuple from xyzb ===
    # octree.xyzb returns (x, y, z, batch) as a tuple of tensors
    x, y, z, b = octree.xyzb(depth, nempty=False)
    
    # Ensure they are long for indexing
    x, y, z, b = x.long(), y.long(), z.long(), b.long()

    # === Step 2: Handle the Patch logic ===
    # Replicating your old logic: 2x2 offsets in XY
    off_x = torch.tensor([0, 0, 1, 1], device=device)
    off_y = torch.tensor([0, 1, 0, 1], device=device)

    # Calculate target indices: [N, 4]
    # We use broadcasting to create the N*4 coordinates
    target_x = (x.view(-1, 1) * 2 + off_x.view(1, -1)).reshape(-1)
    target_y = (y.view(-1, 1) * 2 + off_y.view(1, -1)).reshape(-1)
    
    # z and b just need to be repeated to match the 4 points per node
    target_z = z.repeat_interleave(4)
    target_b = b.repeat_interleave(4)

    # === Step 3: Extract Labels ===
    # voxel_tensor is [B, H, W, D]
    gt_labels = voxel_tensor[target_b, target_x, target_y, target_z].long()

    # Match logits to the 2x2 area (first 2x2 of the 4x4 patch)
    pred_logits = logits[:, :, :2, :2].permute(0, 2, 3, 1).reshape(-1, C)

    # === Step 4: Loss ===
    mask = gt_labels != -1
    
    loss = F.cross_entropy(
        pred_logits[mask], 
        gt_labels[mask], 
        weight=class_weights.to(device), 
        reduction='mean'
    )

    return {
        f"sem_loss_{depth}": loss,
        f"sem_accu_{depth}": (pred_logits[mask].argmax(dim=1) == gt_labels[mask]).float().mean(),
    }
def voxel_to_patch(vox, patch_size=8):

    B, H, W, D = vox.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "Height/Width must be divisible by patch size"

    vox = vox.view(B, H//P, P, W//P, P, D)               # [B, 32, 4, 32, 4, D]
    vox = vox.permute(0, 1, 3, 5, 2, 4)                  # [B, 32, 32, D, 4, 4]
    return vox  # [B, 32, 32, D, 4, 4]
def patch_to_voxel(patches):

    B, H, W, D, p1, p2 = patches.shape
    assert p1 == p2, "patch must be square"

    vox = patches.permute(0, 1, 4, 2, 5, 3)              # [B, 32, 4, 32, 4, D]
    vox = vox.reshape(B, H * p1, W * p2, D)              # [B, 128, 128, D]
    return vox
def get_non_empty_mask(vox, patch_size=8):

    H, W, D = vox.shape
    P = patch_size
    assert H % P == 0 and W % P == 0

    # reshape to group patches
    vox_reshaped = vox.view(H//P, P, W//P, P, D)           # [32, 4, 32, 4, 32]
    vox_reshaped = vox_reshaped.permute(0, 2, 4, 1, 3)     # [32, 32, 32, 4, 4]

    # check if any voxel in each patch is non-zero
    mask = (vox_reshaped != 0).any(dim=(-1, -2))           # [32, 32, 32]
    return mask
