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
    """
    mask = get_non_empty_mask(labels[0], patch_size)
    octree = points2octree(voxel_grid_to_points(mask), depth=depth,
                           full_depth=full_depth)
    if device is not None:
        octree = octree.to(device)
    assign_octree_patch_features(voxel_to_patch(labels, patch_size)[0], octree, depth)
    return octree
