import torch
import os
import numpy as np
import torch
import ocnn
from ocnn.octree import Points
import matplotlib.pyplot as plt
import math
import random
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
