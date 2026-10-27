"""Shared octree construction.

The five-call chain

    get_non_empty_mask -> voxel_grid_to_points -> points2octree
                       -> voxel_to_patch -> assign_octree_patch_features

appeared verbatim in train_sem_vae, train_sem_diffusion, compare_sem_vae and
verify_indoor. It lives here so training and inference cannot drift apart -- a
silent divergence between them is exactly what makes a trained checkpoint decode
to nonsense.

This is deliberately not in `training/`: inference uses it too, and importing
training from inference would be the wrong direction.
"""

from octree_diff.octree.util_octree_stuff import (
    assign_octree_patch_features, get_non_empty_mask, points2octree,
    voxel_grid_to_points, voxel_to_patch,
)


def build_semantic_octree(labels, patch_size, depth, full_depth, device=None):
    """Build the patch-featured octree the semantic VAE consumes.

    labels: [1, X, Y, Z] integer label volume (batch of one -- the octree build
            forces batch 1). Returns the octree with patch features attached.

    Indoor is cubic after patching (128/2 = 64 in every axis) but outdoor is
    not: 256x256x32 at patch 4 gives a 64x64x32 grid, and `voxel_grid_to_points`
    asserts a cubic grid. The KITTI notebooks pad Z up to 64 before building the
    octree, so that is done here, for the patch features as well as the mask --
    `assign_octree_patch_features` indexes them by octree node coordinate, which
    runs to 63 in the padded grid.

    `patch_size` is also forwarded explicitly: that function defaults to 2, so
    an outdoor build that omitted it would allocate 2x2 feature tiles for 4x4
    patches.
    """
    import torch

    mask = get_non_empty_mask(labels[0], patch_size)
    patches = voxel_to_patch(labels, patch_size)[0]

    n = max(mask.shape)
    if tuple(mask.shape) != (n, n, n):
        pad = torch.zeros((n, n, n), dtype=mask.dtype, device=mask.device)
        pad[:mask.shape[0], :mask.shape[1], :mask.shape[2]] = mask
        mask = pad
        pad_p = torch.zeros((n, n, n) + tuple(patches.shape[3:]),
                            dtype=patches.dtype, device=patches.device)
        pad_p[:patches.shape[0], :patches.shape[1], :patches.shape[2]] = patches
        patches = pad_p

    octree = points2octree(voxel_grid_to_points(mask), depth=depth,
                           full_depth=full_depth)
    if device is not None:
        octree = octree.to(device)
    assign_octree_patch_features(patches, octree, depth, patch_size=patch_size)
    return octree
