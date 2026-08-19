# Training config changelog

A record of what the committed configs said versus what each retraining round
actually used, so successive rounds can be compared rather than guessed at.

The "shipped" column is `git show 0afe81f:configs/*.yaml` — the configs as
released. They are **not** known to be the settings the paper's models were
trained with; that recipe was lost along with the indoor weights.

---

## Round 1 — Replica indoor retrain, started 2026-08-05

Trained on 2079 Replica patches (14 scenes, `apartment_0` is 1137 of them) on a
single RTX 4070 Ti Super (16 GB), conda env `octfusion`.

### Data provenance

| dataset | path | notes |
|---|---|---|
| semantic patches | `data/patches_128x128x64_s16` (symlink into the Replica release: `data_patch/train`) | 2079 patches, `labels` `[64,128,128]` int32 + `meta`, labels 0–91 |
| val patches | `data/patches_val` (symlink into the Replica release: `data_patch/val`) | 232 patches |
| structure splits | `data/split_outputs` | **derived, not stored** — regenerate with `python -m dataset.build_split_dataset` (patch_size 2, depth 6, full_depth 4 → `[1,32,32,32]` in {−1,+1}) |
| class counts | `data/replica_class_counts.pt` | `python -m dataset.compute_class_weights` |

The repo shipped no indoor data-prep code; `dataset/build_split_dataset.py` was
written for this round. The `split2octree_small` round trip was verified exact
(`nnum` identical before and after).

### Structure VAE — unchanged

`z_channels 2, base_channels 128, in_channels 1, bs 4, lr 1e-4, 20 epochs`

Trained cleanly. Final recon 108 summed over a batch of 4 × 32³ ≈ **8e-4
MSE/voxel**. Measured latent std **0.4997**.

### Semantic VAE — unchanged

`depth_in 6, depth_stop 6, full_depth 3, latent_dim 16, total_classes 92,
resblk_num 2, channel_in 32, nout 2, patch_size 2, bs 1, lr 1e-4, 20 epochs`

Class weights left **uniform** (`torch.ones(92)`) this round — see "Parked" below.
Measured latent std **0.337** (partially-trained checkpoint).

### Structure diffusion — changed

| knob | shipped | round 1 | why |
|---|---|---|---|
| `batch_size` | 2 | **12** | batch 2 on `[2,32,32,32]` latents is a very noisy gradient. 12 rather than 16 so `sem_diff` co-fits on the same 16 GB card (at bs 16 the pair peaked at 15.4/16 GB) |
| `t_max` (epochs) | 200 | **600** | 174 steps/epoch × 600 = 104k steps @ bs 12, vs 208k @ bs 2 |
| `lr` | 1e-4 | **2e-4** + 500-step warmup + cosine decay to 5% | larger batch supports a higher rate |
| loss reduction | `sum` | **`mean`** | interpretability; AdamW is ~scale-invariant here so this is not expected to change behaviour |
| timestep sampling | `torch.rand(1)` | **`torch.rand(B)`** | every sample in the batch was being given the same noise level |
| EMA | none | **0.999** | EMA weights saved to `save_path`, raw to `*_raw.pt` |
| `grad_clip` | none | **1.0** | |
| `latent_scale` | none | **`auto` → 2.0029** | see below |
| architecture | `in_ch 2, base_ch 128, time_emb_dim 256` | unchanged | 92.2M params |

### Semantic diffusion — changed

| knob | shipped | round 1 | why |
|---|---|---|---|
| `t_max` (epochs) | 20 | **60** | 20 epochs at batch 1 is only ~42k steps |
| effective batch | 1 | **1 × accum 8** | the octree build forces batch 1; accumulate instead |
| `lr` | 1e-4 | 1e-4 + 500-step warmup + cosine decay | |
| EMA | none | **0.999** | |
| `grad_clip` | 1.0 | 1.0 | unchanged |
| `latent_scale` | none | **`auto`** | see below |
| architecture | `mc 64, lr_mc 128, res [1,1,1], mult [1,2]` | **unchanged** | 2.4M params. Notebook inference builds from this config, and 2079 patches from 14 scenes does not obviously support more capacity. Capacity is the first knob to try if quality falls short |

### Latent scaling — new, and it changes the inference contract

Both VAEs emit latents with std well under 1 (structure 0.50, semantic 0.34),
which the cosine log-SNR schedule is not calibrated for. Round 1 multiplies
latents by `1/std` during diffusion training. The resolved factor is written
beside each checkpoint:

- `saved_model/structure_diffusion_meta.yaml` → `latent_scale: 2.0029`
- `saved_model/unet_sem_diffusion_meta.yaml`

**Sampled latents must be divided by this factor before decoding.** Read it with
`utils/latent_scale.py::load_latent_scale`. Set `latent_scale: 1.0` in the config
to reproduce the shipped (unscaled) behaviour.

### Infrastructure fixes — not hyperparameters, but training did not run without them

- `main.py` launched the train scripts as subprocesses without the repo root on
  the path → `ModuleNotFoundError: models.structure_networks`. Now sets `cwd` and
  `PYTHONPATH`.
- `train_structure_vae.py::__main__` called `inference()` with `train()`
  commented out.
- Checkpoint saving was guarded by `epoch % 10 == 0` (or `% 5`) in three scripts,
  so with `epochs: 20` the last save was at epoch 10 and the final weights were
  discarded. Now every epoch.
- `dataset/voxel_dataset.py::get_dataloader` had `num_workers=0`; loading 9 MB
  patches serially starved the GPU. Now 4.

### Known quirk, left as-is

`models/networks/diffusion_networks/graph_unet_hr.py:258` only runs
`middle_block1`/`middle_block2` when `unet_lr is not None`. Single-model training
passes `unet_lr=None`, so the semantic UNet effectively has **no bottleneck** —
the deepest encoder features pass straight to the decoder via skips, and the
middle-block parameters never receive a gradient. Training and inference agree on
this, so it is correct-as-shipped, but it is worth knowing that a slice of the
checkpoint is dead weight, and enabling it is a capacity option.

---

## Parked for round 2

**Class weights.** `data/replica_class_counts.pt` holds `counts`, `freq`,
`weights_inv`, `weights_inv_sqrt`, `weights_enet` (β=0.9999) over 2.18B voxels.
90/92 classes are present. The imbalance is severe: class 0 (free space) is
**93.6%** of all voxels and the rarest present class has **81** voxels — an
18000:1 ratio among non-free classes.

Agreed first thing to try: **`weights_inv_sqrt`**. Plain `weights_inv` would be
violent at that ratio.

Applying it means retraining `sem_vae` **and then** `sem_diff`, since the
diffusion model is trained on that VAE's latents. Deferred until round 1's
samples have been evaluated.

---

## Round 2 — semantic branch only, frequency-weighted, started 2026-08-07

Structure branch (`str_vae`, `str_diff`) is **unchanged and reused** from round 1.
Round 1's semantic checkpoints are preserved in `saved_model/round1_uniform/`.

### The change

`configs/sem_vae_config.yaml`:

```yaml
class_weights: inv_sqrt
class_weights_file: "data/replica_class_counts.pt"
```

`train_sem_vae.py` previously hardcoded `torch.ones(92)`. It now loads the scheme
via `utils/class_weights.py`.

### Two corrections to how the weights were derived

1. **Basis.** Round 1's counts were taken over the whole `[128,128,64]` volume,
   where free space is 93.6%. But `compute_semantic_loss` only evaluates the 2x2
   voxel patch under each *occupied* depth-6 octree node — 28M voxels, not 157M,
   with free space at 64.2%. `dataset/compute_class_weights.py --basis loss` now
   counts through the octree exactly as the loss gathers. The old volume-basis
   file is kept as `data/replica_class_counts_volume.pt`.
2. **Free space had weight 0.** Class 0 was excluded from the weighting stats, so
   `weights_inv_sqrt[0]` came out 0.0 — empty voxels would have produced no
   gradient at all, and a node's 2x2 patch is often partly empty. Class 0 is now
   included and weighted 0.0104.

A clamp on the weight range was tried and rejected: at 10:1 it flattened the six
most common classes (a 48:1 count range) onto the same floor. It is unnecessary
for `inv_sqrt`, where a class's contribution scales as `count x weight ∝ sqrt(count)`,
so rare classes cannot run away. `--clamp 0`.

Effect of each scheme on the loss (share of total weighted contribution):

| scheme | free-space share | rarest-20 combined |
|---|---|---|
| none (round 1) | 64.2% | ~0% |
| **inv_sqrt** | **19.6%** | **1.6%** |
| enet (beta=0.9999) | 64.2% | 0.06% |
| inv | 1.1% | 20.0% |

`enet` is a near no-op at this beta; `inv` is violent. `inv_sqrt` was chosen.

### Result — semantic VAE, 40 held-out patches

(`python -m inference_scripts.compare_sem_vae`)

| | micro acc | macro acc | mIoU |
|---|---|---|---|
| uniform (round 1) | 0.9741 | 0.7130 | 0.6497 |
| **inv_sqrt (round 2)** | 0.9693 | **0.9395** | **0.7873** |
| delta | −0.0048 | **+0.2265** | **+0.1376** |

Micro accuracy is dominated by floor/wall/ceiling and barely moves. Macro
accuracy and mIoU jump, because the uniform model was reconstructing many rare
classes at literally 0.000 — e.g. classes 43, 44, 78, 79 went from 0.00–0.25 to
0.84–1.00. Individual rare-class figures are noisy (some have only tens of voxels
in the val set), but the direction is consistent across all of them.

Final training accuracy 0.949 (vs 0.987 for uniform) — the expected micro cost.

### Semantic diffusion

Retrained on the new VAE's latents; no config change of its own. Measured
`latent_scale` 3.0764 (round 1: 3.5223).

Both stages ran ~3x faster than round 1 simply because the GPU was not shared
with `str_diff`: `sem_vae` 2.7 h, `sem_diff` ~2.9 h.
