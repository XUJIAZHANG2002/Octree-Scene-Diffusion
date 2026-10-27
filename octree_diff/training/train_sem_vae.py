import glob
import os

import numpy as np
import torch
import tqdm
from octree_diff.octree.build import build_semantic_octree
from octree_diff.models.factory import build_semantic_vae
from octree_diff.data.voxel_dataset import get_dataloader
from octree_diff.training.losses import compute_semantic_loss, compute_octree_loss
from octree_diff.config import load_stage
from octree_diff.training.class_weights import load_class_weights

def train(config_dir=None, device=None):
    # Load and split config for cleaner access
    full_config, _ = load_stage("sem_vae", config_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m_cfg = full_config["model"]
    t_cfg = full_config["training"]

    # 1. Setup Data. Indoor reads prepared .pt patches; outdoor decodes
    # SemanticKITTI voxel labels on the fly (~11 ms a frame, so materialising a
    # derived set would cost ~46 GB for no speedup).
    dataset_kind = t_cfg.get("dataset", "replica_patches")
    if dataset_kind == "kitti_ssc":
        from octree_diff.data.kitti.ssc_dataset import (
            VAL_SEQUENCES, SSCVoxelDataset, TRAIN_SEQUENCES, get_ssc_dataloader)
        loader = get_ssc_dataloader(t_cfg["data_dir"], batch_size=t_cfg["batch_size"],
                                    stride=t_cfg.get("stride", 5),
                                    sequences=TRAIN_SEQUENCES)
    else:
        loader = get_dataloader(t_cfg["data_dir"], batch_size=t_cfg["batch_size"])

    # 2. Initialize Model using m_cfg
    vae = build_semantic_vae(m_cfg, device)
    patch_size = m_cfg["patch_size"]
    optimizer = torch.optim.Adam(vae.parameters(), lr=t_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"])

    # The KL term is a regulariser on the latent the diffusion model later has
    # to fit, not a reconstruction objective. At the original implicit weight of
    # 1.0 it was measured at 42% of the total loss on the shipped checkpoint,
    # which pulls the latent hard toward N(0,1) and costs the decoder the detail
    # it needs. Latent-diffusion autoencoders typically run this several orders
    # of magnitude lower; `latent_scale` separately handles the scale, so the
    # KL is not carrying that job. Defaults to 1.0 so untouched configs are
    # unchanged.
    kl_weight = float(t_cfg.get("kl_weight", 1.0))
    grad_clip = float(t_cfg.get("grad_clip", 0.0))
    print(f"kl_weight={kl_weight}  grad_clip={grad_clip or 'off'}")

    class_weights = load_class_weights(
        t_cfg.get("class_weights", "none"),
        t_cfg.get("class_weights_file", "data/replica_class_counts.pt"),
        m_cfg["total_classes"],
    ).to(device)

    # Held-out monitoring. The shipped recipe saved every epoch with no
    # validation, so the final model was whichever epoch happened to be last.
    # That is tolerable for 0.68M parameters and risky for 11M on 2079 patches.
    val_dir = t_cfg.get("val_dir")
    val_n = t_cfg.get("val_n", 16)
    val_vols = []
    if dataset_kind == "kitti_ssc":
        vds = SSCVoxelDataset(t_cfg["data_dir"], sequences=VAL_SEQUENCES,
                              stride=t_cfg.get("stride", 5))
        step = max(len(vds) // val_n, 1)
        for i in range(0, min(len(vds), step * val_n), step):
            val_vols.append(vds[i][0].numpy().astype(np.int32))
        print(f"validation: {len(val_vols)} held-out frames from sequence(s) "
              f"{','.join(VAL_SEQUENCES)}")
    elif val_dir and os.path.isdir(val_dir):
        for f in sorted(glob.glob(os.path.join(val_dir, "*.pt")))[:val_n]:
            lab = np.asarray(torch.load(f, weights_only=False)["labels"]).astype(np.int32)
            # val patches are stored Z-first; the pipeline works in x, y, z
            val_vols.append(np.transpose(lab, (2, 1, 0)))
        print(f"validation: {len(val_vols)} held-out patches from {val_dir}")

    @torch.no_grad()
    def validate():
        from octree_diff.octree.util_octree_stuff import reconstruct_voxel_from_patch
        vae.eval()
        accs = []
        for gt in val_vols:
            v = torch.from_numpy(gt).unsqueeze(0).to(device).long()
            oc = build_semantic_octree(v.clamp(min=0), patch_size, m_cfg["depth_in"],
                                       m_cfg["full_depth"], device)
            o = vae(oc, octree_out=oc, update_octree=False)
            rec = reconstruct_voxel_from_patch(
                o["sem_voxs"], oc, depth=m_cfg["depth_in"],
                shape=(1,) + gt.shape, patch_size=patch_size)[0].cpu().numpy()
            occ = gt > 0            # -1 (unobserved) and 0 (free) both excluded
            accs.append(float((rec[occ].astype(np.int32) == gt[occ]).mean()))
        vae.train()
        return float(np.mean(accs))

    best = -1.0
    best_path = t_cfg["checkpoint_path"].replace(".pt", "_best.pt")

    # 3. Training Loop
    for epoch in range(t_cfg["epochs"]):
        vae.train()
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        
        for vox, _ in pbar:
            vox = vox.to(device).long()
            
            # Preprocessing. SemanticKITTI marks unobserved voxels -1; they must
            # not count as occupied when building the octree, but they stay -1 in
            # the loss target, where compute_semantic_loss drops them.
            octree = build_semantic_octree(vox.clamp(min=0), patch_size,
                                           m_cfg["depth_in"],
                                           m_cfg["full_depth"], device)
            
            # Forward Pass
            output = vae(octree, octree_out=octree, update_octree=False)
            
            # Loss Calculation
            sem_loss_dict = compute_semantic_loss(
                output['sem_voxs'], output['octree_out'], vox,
                class_weights
            )
            oct_loss_dict = compute_octree_loss(output['logits'], octree)
            
            total_loss = kl_weight * output['kl_loss'] + sem_loss_dict["sem_loss_6"]
            total_loss += sum(v for k, v in oct_loss_dict.items() if 'loss_' in k)

            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{total_loss.item():.3f}", 
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "acc": f"{sem_loss_dict.get('sem_accu_6', 0.0):.3f}",
                "kl": f"{output['kl_loss'].item():.4f}",
                "sem": f"{sem_loss_dict['sem_loss_6'].item():.4f}"
            })

        scheduler.step()

        os.makedirs(os.path.dirname(t_cfg["checkpoint_path"]), exist_ok=True)
        torch.save(vae.state_dict(), t_cfg["checkpoint_path"])

        if val_vols:
            acc = validate()
            flag = ""
            if acc > best:
                best = acc
                torch.save(vae.state_dict(), best_path)
                flag = "  <- best, saved"
            print(f"epoch {epoch}: held-out occupied-voxel accuracy {acc:.4f}{flag}",
                  flush=True)

if __name__ == "__main__":
    train()