
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



def compute_semantic_loss(
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
    # Each node owns a PxP tile in XY and one voxel in Z. P comes from the
    # decoder output (2 indoor, 4 outdoor) rather than being assumed.
    P = H
    ppn = P * P                                   # voxels per node
    off_x = torch.arange(P, device=device).repeat_interleave(P)
    off_y = torch.arange(P, device=device).repeat(P)

    # Calculate target indices: [N, P*P] flattened
    target_x = (x.view(-1, 1) * P + off_x.view(1, -1)).reshape(-1)
    target_y = (y.view(-1, 1) * P + off_y.view(1, -1)).reshape(-1)

    # z and b repeated to match the P*P points per node
    target_z = z.repeat_interleave(ppn)
    target_b = b.repeat_interleave(ppn)

    # === Step 3: Extract Labels ===
    # voxel_tensor is [B, H, W, D]
    gt_labels = voxel_tensor[target_b, target_x, target_y, target_z].long()

    # logits are [N, C, P, P]; flatten to match the target ordering above
    pred_logits = logits.permute(0, 2, 3, 1).reshape(-1, C)

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