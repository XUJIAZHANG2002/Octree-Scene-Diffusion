"""Compare two semantic VAE checkpoints on held-out patches.

Intended for the class-weighting A/B: round 1 trained with uniform weights,
round 2 with inv_sqrt. Weighting is expected to trade a little overall accuracy
for a lot of rare-class accuracy, so the headline number to watch is the **macro**
(mean per-class) figure, not the micro one.

    python -m octree_diff.inference.compare_sem_vae \
        --a saved_model/<baseline>.pt --a-name uniform \
        --b saved_model/vae_model_5.pt --b-name inv_sqrt
"""

import argparse
import glob

import torch

from octree_diff.octree.build import build_semantic_octree
from octree_diff.models.semantic.graph_sem_vae import GraphVAE
from octree_diff.config import load_config
from octree_diff.octree.util_octree_stuff import reconstruct_voxel_from_patch
from octree_diff.viz.vis_indoor import label_name

DEPTH, PATCH = 6, 2


def load_vae(path, m, device):
    vae = GraphVAE(
        depth=m["depth_in"], channel_in=m["in_channels"], nout=m["nout"],
        full_depth=m["full_depth"], depth_stop=m["depth_stop"],
        depth_out=m["depth_in"], latent_dim=m["latent_dim"],
        num_classes=m["total_classes"], resblk_num=m["resblk_num"],
    ).to(device).eval()
    vae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return vae


@torch.no_grad()
def evaluate(vae, files, m, device, num_classes):
    """Per-class confusion tallies over occupied ground-truth voxels."""
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)
    pred_total = torch.zeros(num_classes, dtype=torch.long)

    for f in files:
        labels = torch.load(f, map_location="cpu")["labels"]
        labels = labels.long().permute(1, 2, 0).contiguous().to(device)

        octree = build_semantic_octree(labels.unsqueeze(0), PATCH, DEPTH,
                                       m["full_depth"], device)

        _, mu, doctree = vae.extract_code(octree)
        out = vae.decode_code(mu, doctree, update_octree=False, pos=None)
        rec = reconstruct_voxel_from_patch(out["sem_voxs"], octree, depth=DEPTH,
                                           shape=(1, 128, 128, 64))[0]

        occ = labels > 0
        gt, pr = labels[occ].cpu(), rec[occ].cpu()
        total += torch.bincount(gt, minlength=num_classes)
        pred_total += torch.bincount(pr, minlength=num_classes)
        correct += torch.bincount(gt[gt == pr], minlength=num_classes)

    return correct, total, pred_total


def summarise(correct, total, pred_total):
    present = total > 0
    micro = correct.sum().item() / max(total.sum().item(), 1)
    per_class = correct[present].float() / total[present].float()
    macro = per_class.mean().item()
    # IoU per class: correct / (gt + pred - correct)
    denom = (total + pred_total - correct).float().clamp_min(1)
    iou = (correct.float() / denom)[present]
    return {"micro_acc": micro, "macro_acc": macro, "miou": iou.mean().item(),
            "classes": int(present.sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="baseline checkpoint")
    p.add_argument("--b", required=True, help="checkpoint to compare against it")
    p.add_argument("--a-name", default="A")
    p.add_argument("--b-name", default="B")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--val", default="data/patches_val")
    p.add_argument("--device", default="cuda")
    p.add_argument("--top", type=int, default=12, help="per-class rows to print")
    args = p.parse_args()

    m = load_config("configs/sem_vae_config.yaml")["model"]
    nc = m["total_classes"]
    files = sorted(glob.glob(f"{args.val}/*.pt"))[:args.n]
    print(f"evaluating on {len(files)} held-out patches\n")

    res = {}
    for name, path in [(args.a_name, args.a), (args.b_name, args.b)]:
        vae = load_vae(path, m, args.device)
        res[name] = evaluate(vae, files, m, args.device, nc)
        del vae
        torch.cuda.empty_cache()
        print(f"  {name:10s} <- {path}")

    print(f"\n{'':12s} {'micro acc':>10} {'macro acc':>10} {'mIoU':>8}")
    for name in (args.a_name, args.b_name):
        s = summarise(*res[name])
        print(f"{name:12s} {s['micro_acc']:>10.4f} {s['macro_acc']:>10.4f} {s['miou']:>8.4f}")
    sa, sb = summarise(*res[args.a_name]), summarise(*res[args.b_name])
    print(f"{'delta':12s} {sb['micro_acc']-sa['micro_acc']:>+10.4f} "
          f"{sb['macro_acc']-sa['macro_acc']:>+10.4f} {sb['miou']-sa['miou']:>+8.4f}")
    print("\nmacro/mIoU rising while micro dips slightly is the intended trade-off.")

    # rarest classes, where weighting should show up most
    ca, ta, _ = res[args.a_name]
    cb, tb, _ = res[args.b_name]
    present = (ta > 0).nonzero().flatten()
    order = present[ta[present].argsort()][:args.top]
    print(f"\nrarest {args.top} classes in the val set:")
    print(f"{'class':>14} {'gt voxels':>10} {args.a_name:>12} {args.b_name:>12} {'delta':>8}")
    for i in order.tolist():
        a = ca[i].item() / ta[i].item()
        b = cb[i].item() / tb[i].item()
        print(f"{label_name(i):>14} {ta[i].item():>10,} {a:>12.3f} {b:>12.3f} {b-a:>+8.3f}")


if __name__ == "__main__":
    main()
