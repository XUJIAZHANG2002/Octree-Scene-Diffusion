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

---

## Round 3 — indoor + outdoor retrain, started 2026-10-15

Driven by the project page: the renders from the round-2 weights were speckled,
and the cause turned out to be in the semantic VAE rather than in sampling.
Configs live in `configs/v2` (Replica) and `configs/v2_outdoor` (SemanticKITTI);
the round-2 configs at the top level are unchanged.

### Four defects, in order of impact

1. **Capacity was hardcoded.** `GraphVAE.__init__` set `self.channels` to a
   fixed list regardless of config, giving 32 channels at depth 6 and **0.68M
   parameters using 1 GiB of 16**. `latent_dim` was wired in, but it only moves
   `KL_conv`/`post_KL_conv` — about 1.5k parameters — so raising it alone did
   nothing. A `width_mult` was added, which scales the whole list.

2. **The KL term was unweighted.** At weight 1.0 it was 42% of total loss and
   had crushed the posterior to `latent_std` 0.392, well under the unit variance
   the diffusion schedule assumes. `kl_weight` was added; 1e-3 is the knee.

3. **Class weights ran with capping off.** A 1753:1 spread put 61% of the weight
   mass on 14 classes holding 0.017% of the voxels. `--max-ratio` (a tail-only
   cap against the median) and `--min-count` were added, and the `--clamp`
   default changed 10.0 -> 0.0, because symmetric clamping flattens the head on
   this distribution — the effect already documented in round 2.

4. **`patch_size` was never passed to the VAE.** Dropped in three places: the
   factory, `inference/pipeline.py` (which bypassed the factory entirely and
   constructed `GraphVAE(...)` directly, so v2 checkpoints would not load), and
   `build_semantic_octree`, whose default of 2 would have tiled outdoor's 4x4
   patches as 2x2.

An A/B over 3-epoch arms separated these. Capacity dominates and the class weights were nearly
irrelevant to the speckling:

| arm | params | micro | macro | incoherent | latent_std |
|---|---|---|---|---|---|
| baseline, kl 1.0 | 0.68M | 0.9255 | 0.8315 | 0.0044 | 0.392 |
| kl 1e-2 | 0.68M | 0.9519 | 0.8717 | 0.0041 | 1.089 |
| kl 1e-2 + capped weights | 0.68M | 0.9527 | 0.8791 | 0.0041 | 1.079 |
| kl 1e-3 | 0.68M | 0.9555 | 0.8794 | 0.0039 | 1.283 |
| kl 1e-3, width 4 | 2.29M | 0.9883 | 0.9410 | 0.0019 | 1.041 |
| **kl 1e-3, width 8** | **11.01M** | **0.9956** | **0.9672** | **0.0007** | 0.913 |

Width 8 is the largest that fits; the first pass of this A/B was scored on the
training loader and was re-run on `data/patches_val` before anything was chosen
(the ranking held, 0.9956 vs 0.9948).

### Semantic VAE — Replica

`latent_dim 32` (was 16), `resblk_num 3` (was 2), `width_mult 8.0`,
`kl_weight 1e-3`, held-out validation on `data/patches_val` with
best-checkpoint selection. 20 epochs. Best held-out occupied-voxel accuracy
**0.9996** (`out/retrain_sem_vae.log`).

### Semantic VAE — SemanticKITTI

New: the outdoor stage had no training path in this repo. `total_classes 21`,
`patch_size 4`, `full_depth 3`, `width_mult 8.0`, `kl_weight 1e-3`, 11.03M
parameters, 2.03 GiB, 20 epochs in 9.6 h. Best held-out **0.9993**.

`octree_diff/data/kitti/ssc_dataset.py` decodes `sequences/*/voxels/*.label`
on the fly — 11 ms a frame against ~1 s steps, so materialising a derived set
would have cost ~46 GB for no speedup. Stride 5 (consecutive frames differ in
about 4% of voxels) gives 3,834 train / 815 val frames, sequence 08 held out as
in the SSC benchmark. `.invalid` voxels become -1 so the loss drops them, and
the octree is built from `labels.clamp(min=0)` because
`get_non_empty_mask` treats any non-zero label as occupied.

`data/kitti_class_counts.pt`, over 300 frames spread across the sequences:
74.7M supervised voxels, 20 classes, weights 0.042–2.987. The spread is only
71:1 and the rarest class (motorcycle) still has 8,658 voxels, so unlike Replica
there is no unlearnable tail and no `--min-count` is needed.

### Result — semantic VAE, held-out

Round-trip reconstruction on held-out data. Indoor: 60 of the 232 patches in
`data/patches_val`. Outdoor: 60 frames from sequence 08 at stride 20.

| | micro | macro | classes | incoherent | latent_std | params |
|---|---|---|---|---|---|---|
| shipped indoor | 0.9663 | 0.9340 | 50.5 | 0.0038 | 0.332 | 0.68M |
| **retrained indoor** | **0.9995** | **0.9814** | 32.5 | **0.0006** | 0.606 | 11.01M |
| real indoor patches | 1.0 | 1.0 | 32.1 | 0.0005 | — | — |
| shipped outdoor | 0.7311 | 0.5408 | 17.5 | 0.0300 | 0.757 | 0.63M |
| **retrained outdoor** | **0.9991** | **0.9920** | 12.4 | **0.0100** | 0.750 | 11.03M |
| real outdoor frames | 1.0 | 1.0 | 11.0 | 0.0110 | — | — |

`classes` and `incoherent` are read against the ground-truth row, not minimised:
the shipped models put 50.5 classes in an indoor patch that holds 32.1 and 17.5
in an outdoor frame that holds 11.0, and leave 6x (indoor) and 3x (outdoor) as
many voxels sharing a label with no neighbour. That is the speckling. Both
retrains land on the real distribution.

The outdoor gap is far wider than the indoor one — 0.73 against 0.97 micro —
because the shipped outdoor VAE is smaller still (0.63M, `latent_dim` 8, a
single resblock) and outdoor frames hold a denser, less repetitive scene than a
Replica patch.

### Semantic diffusion

Indoor: 13.08M parameters (`model_channels 128`, `lr_model_channels 256`,
`num_res_blocks [2,2,2]`), 180 epochs, `latent_scale` pinned at
**1.651021933697025**. Outdoor: 37.04M (`192/384`, `[3,3,3]`), 60 epochs x 3,834
frames at accum 8 = 28,755 optimiser steps, ~17 h, `latent_scale` pinned at
**1.330161874580**.

`latent_scale` is pinned rather than left at `auto` because a resume re-measured
it (1.6510 -> 1.6700) and silently invalidated the checkpoint — inference
divides by this number. Resume support was added to `train_sem_diffusion.py`
(raw weights plus EMA, `warmup_steps 0` on resume) and `EMA.load_state_dict`
now restores the step count.

Generated-scene statistics over seed sweeps, indoor, against real patches at 31.7 classes / 0.0005 incoherent:

| weights | classes | incoherent |
|---|---|---|
| round 2 | 76.9 | 0.0122 |
| v2 VAE + 60-epoch diffusion | 56.1 | 0.0091 |
| v2 VAE + 180-epoch diffusion | 51.2 | 0.0059 |

### Structure stage — not retrained

Investigated and left alone. The structure VAE round-trips the split grid at
IoU **1.0000**, and generated scenes are ceiling-less 8% of the time against 5%
for real data, so the open layouts the semantic model handles worst are not an
artefact of structure. An earlier recommendation to retrain structure was
withdrawn on that measurement.

### Compatibility

**`width_mult` changes parameter shapes, so the released checkpoints are no
longer drop-in.** `saved_model/vae_model_5.pt` and the shipped outdoor
`vae_fdepth6_ldim8_8.pt` load only under their own geometry, which the
README lists for each domain. Config and
checkpoint must be kept together; `v2` and `v2_outdoor` checkpoints live in
`saved_model/v2/` and `saved_model/v2_outdoor/`.

---

## Round 4 — inference fixes for the project page, started 2026-10-25

No retraining. Every change here is in how the trained models are sampled,
measured against training-sequence statistics and then checked by eye.

### Outdoor structure: the paper's DDPM model, not notebook 01's DDIM one

The outdoor exporters used `SemCityOcto/structure_model.pt` (15.7M, 50 DDIM
steps), the model notebook 01 completes with. The paper's outdoor FID notebook
(`vae_fid_kid.ipynb`) and the outpainting notebooks sample
`SemCityOcto/checkpoints/structure_ddpm_ep110.pt` instead: 62.8M parameters,
ε-prediction, 1000 ancestral steps, EMA weights. The repo's
`models/outdoor/graph_unet_lr.py` gained the `timestep_repr="ddpm"` branch it
needs, `diffusion/ddpm.py` the sampler plus RePaint-style inpainting, and the
outdoor scripts a `--structure {ddim,ddpm}` switch, now defaulting to `ddpm`.

24 unconditional draws each, same semantic models, against 40 training frames:

| | occupied | 8×8×2 blocks | class-mix KL | vegetation | terrain | building | road |
|---|---|---|---|---|---|---|---|
| training data | 0.049 | 3159 | 0.237 | 0.379 | 0.077 | 0.136 | 0.164 |
| DDIM `structure_model.pt` | 0.068 | 4003 | 0.283 | 0.642 | 0.107 | 0.079 | 0.064 |
| DDPM `structure_ddpm_ep110.pt` | 0.036 | 2862 | 0.234 | 0.466 | 0.079 | 0.142 | 0.144 |

The class-mix KL is the per-frame KL against the training mean; real frames
score 0.237 against it, so DDPM draws are as close to the mean mix as real
scenes are. BEV realism (below) agrees: median 1.44 vs 1.65. By eye, DDPM
draws are streets — road corridors with sidewalks and façades — where DDIM
draws were dominated by vegetation blobs. DDPM under-fills by about a quarter.

For completion, RePaint with 3 resamples per step keeps the scan as well as
the DDIM model did (0.993–0.998 of hits) and moves BEV realism from 2.22–2.83
to 1.36–1.67 on held-out frames 08:000093 and 08:000073.

**Release:** the DDPM checkpoint is not in `saved_model/`; it has to ship with
the outdoor weights for `--structure ddpm` to work outside this machine.

### Outdoor semantic graph rebuilt as trained

`split2octree_small` + `create_child_octree` gave the semantic stage a dense
depth-4 layer (4096 nodes against 1,200–1,400 in training) and ~2,500 extra
dual-graph nodes. `split_to_octree` rebuilds with `points2octree(full_depth=3)`,
the indoor `split_big_to_octree` fix. Small effect: classes per scene 17.6 →
16.1 over 16 draws; retention unchanged.

### Indoor completion and extension: guided semantics

Pinning indoor semantic latents encoded from partial labels kept 0.59–0.82 of
the observed labels on the sample scenes; the outdoor decoder-space guidance
keeps 0.885–0.997 (`semantic_stage="guided"`, now the default for modes 4 and
5 in both domains). Most of the remaining loss was the structure stage dropping
part of the observed geometry, which semantic guidance cannot restore; see the
next section. Cost: 2–5 min per completion against ~20 s.

### Indoor completion: 64 candidate layouts, not 3

With `best_of=3`, seven of the ten sample scenes kept only 0.89–0.92 of the
observed occupied voxels. Splitting the loss (`scene_02`, `scene_07`): the
structure VAE round-trips every partial view exactly and the known latents stay
pinned to the end, yet the sampled layout covers only 0.926–0.933 of the
observed voxels, while the semantic stage loses under 0.015 on top. The decoder
mixes generated neighbour latents into cells the sensor only partly saw, and on
these masks 80–90% of occupied observed cells are partly seen.

Over 24 single layouts per scene, the share that keeps >=0.98 is 1/24 (02),
3–6/24 (06–09), 11/24 (05), 18/24 (04); bad layouts keep 0.67–0.97. The
known-cell agreement `inpaint_structure` already ranks by predicts retention
(correlation 0.96–1.00 per scene), but good and bad score levels overlap across
scenes, so there is no fixed early-stop threshold. More RePaint resampling
(3, 6) did not help. Rerunning the seven scenes at `best_of=64` (~2 s per layout):

| | observed voxels kept | labels kept | draws passing 0.98 / 0.97 |
|---|---|---|---|
| best_of 3 (scenes 02, 04–09, 21 draws) | 0.889–0.991 | 0.834–0.975 | 1 / 21 |
| best_of 64 (same scenes, 21 draws) | 0.986–0.997 | 0.938–0.993 | 13 / 21 |

By eye the old draws for 06–09 were unrelated rooms; the new ones keep the
observed staircase and walls. Scenes 04 and 05 still relabel 3–6% in the
semantic stage. `--best-of` now defaults to 64 for completion in
`export_web_scenes.py`, and the README says so.

### Outdoor extension starts from ground truth

Mode 5 is "take a full map and outpaint beyond it". The outdoor exporter had
been completing a single scan, with the semantic stage unconditioned, and
extending that. It now extends the frame's SSC ground truth (invalid voxels
empty).

### Tried and rejected

- **Autoguidance** (Karras et al. 2024) with `unet_sem_v2_60ep.pt` as the weak
  model, indoor generation, 12 fixed layouts: classes per scene 48.4 / 51.6 /
  53.8 / 58.2 and incoherence 0.0048 / 0.0056 / 0.0070 / 0.0090 at w = 1 / 1.5 /
  2 / 2.5 — monotonically worse. Not kept in the release.
- **More semantic steps with the legacy outdoor sampler**: 50 / 100 / 200 steps
  give incoherence 0.0091 / 0.0162 / 0.0321. `util_sample_stuff.ddim_sample_logsnr_cosine`
  re-noises with fresh noise each step, so steps add speckle. Swapping it for
  true DDIM (`sampling.ddim_sample_graph`, eta 0) on DDPM layouts is a wash:
  class-mix KL 0.115 → 0.099 and BEV 1.40 → 1.33, but incoherence 0.0122 →
  0.0166 against a real 0.0118. Generation stays on the legacy sampler at 50
  steps; completion and extension already run true DDIM inside the guided
  sampler.

### Selection

Outdoor draws are scored per sample on the BEV VAE the paper's outdoor FID/KID
used (`bev_vae_epoch50.pt`): k-NN distance to 4,000 real training BEVs over the
median real-to-real distance, where held-out real frames score ~1.3; and on the
class-mix KL against training frames. Completion and extension draws must first
keep at least 98% of the observed voxels and 97% of their labels. The project
page's scenes were picked by eye from the top of those rankings. The sweep and
scoring tools are not part of the release.
