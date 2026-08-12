# Notebooks

These are the notebooks the paper's results were actually produced with. They are
**all outdoor / SemanticKITTI**. For indoor (Replica) inference use
`inference_scripts/inference_indoor.py` instead — see `notebooks/indoor/README.md`.

`inference_scripts/inference.py` is a later attempt at packaging this up. It does
not work; the defects are catalogued in `inference_scripts/NOTES.md`.

## Renamed from

| was | now |
|---|---|
| `test7.ipynb` | `kitti/01_completion_ddim_logsnr.ipynb` |
| `test8.ipynb` | `kitti/02_outpaint_ddpm_library.ipynb` |
| `test9.ipynb` | `kitti/03_completion_then_outpaint.ipynb` |
| `test9 copy.ipynb` | `kitti/04_semantic_latent_inpaint_batch_export.ipynb` |
| `test_cumuti.ipynb` | `kitti/05_cumulti_zeroshot_extend_along_x.ipynb` |

## What each one holds

**`01_completion_ddim_logsnr.ipynb`** — the earliest, and the only notebook that
is fully continuous-log-SNR. Uses a different structure model
(`in_split_channels=8, model_channels=32`) on octant-coded `[1,8,16,16,8]`.
Contains the original `split_outputs` generation cell that
`dataset/build_split_dataset.py` is derived from, the numba reverse-ray-trace
occupancy carving, and — cell 23 — the only
`ddim_sample_structure_logsnr_cosine_inpaint`: x0-prediction, continuous
schedule, deterministic DDIM. That is the formulation the *current* training code
uses, which makes this cell the reference for any new structure inpainting.

**`02_outpaint_ddpm_library.ipynb`** — the library notebook. First definitions of
`build_ddpm_sampling_buffers`, `sample_ddpm`, `q_sample`, `q_forward_xt_to`,
`sample_ddpm_inpaint_blended`, `_build_repaint_times`,
`sample_ddpm_inpaint_repaint`, `split2splitbig`/`splitbig2split`,
`spiral_centers_halfstep`, `build_mask_by_neighbors`, `outpaint_square_spiral`,
`construct_octree_dict`, `sem_paint_octree`, `visualize_structure`. Its
`outpaint_latent_canvas`, `_copy_block_into` and `define_mask_and_gt_for_tile`
are **broken drafts** — use the versions in `03_`.

**`03_completion_then_outpaint.ipynb`** — the most complete pipeline: scan →
occupancy carve → tri-state patch reduce → structure RePaint → octree → semantic
DDIM → `.label` export → mIoU, then spiral outpainting from a known canvas. Holds
the **corrected** `_copy_block_into` / `define_mask_and_gt_for_tile` /
`outpaint_latent_canvas`.

**`04_semantic_latent_inpaint_batch_export.ipynb`** — despite the old name, *not*
a duplicate of `03_`; only the first ~30 cells overlap. Unique content: the
2000-scan batch export loop (the most script-like cell in the repo), and the only
working **latent-space semantic inpainting from ground truth** — `extract_code`
→ `mu_gt`, per-index `search_xyzb` copy into `z_gt`/`mask` with
`offset = doctree.total_num - doctree.nnum[6]`, then
`ddim_sample_logsnr_cosine_inpaint`. Contains no outpainting.

**`05_cumulti_zeroshot_extend_along_x.ipynb`** — CU-Multi zero-shot. Raw `.bin`
point-cloud ingest with intensity filtering and outlier removal, a non-square
voxel grid, and the sliding +x scene-extension loop. Note
`outpaint_structure_along_x` is an **empty stub**; the real implementation is the
inline `while` loop in the cell above it.

## Why they will not run as-is against this repo

Not fixed on purpose — recorded so nobody rediscovers it the hard way.

1. **`graph_densed_sem_vae` was renamed.** Commit `19ece4d` renamed
   `models/networks/dualoctree_networks/graph_densed_sem_vae.py` →
   `graph_sem_vae.py` (94% similar; only `mpu.NeuralMPU` was dropped — the
   `extract_code` / `decode_code` / `create_child_octree` API is identical). One
   import line, not a lost module.
2. **`reconstruct_voxel_from_patch` lost its `patch_size` argument.** Every
   notebook calls it with `patch_size=4`, but the current signature
   (`utils/util_octree_stuff.py:53`) is
   `(sem_voxs, octree, depth, shape=(1,128,128,64))` and hardcodes `x*2`/`y*2` —
   i.e. it is now indoor/patch-2 only. Notebook calls raise `TypeError`.
   Restoring patch-4 means turning that `*2` into a parameter.
3. **They need the old KITTI checkpoints and hyperparameters** —
   `~/SemCityOcto/vae_fdepth6_ldim8_8.pt`, `Model_is_good_8_6.pt`,
   `checkpoints/structure_ddpm_ep*.pt`, with `latent_dim=8`, `num_classes=21`,
   `patch_size=4`. They will not load the Replica checkpoints in `saved_model/`.
4. **Imports assume a flat cwd** — `from util_sample_stuff import *` rather than
   `from utils.util_sample_stuff import *`.

## Landmines in the code itself

If you lift functions out of these notebooks, know that:

- `outpaint_square_spiral` has a leftover `if i == 2: break` that truncates the
  spiral after three tiles (in `03_` and `05_`; commented out in `02_`).
- `ddim_sample_logsnr_cosine_inpaint`'s re-noising branch is guarded by `t > 1`,
  but `t ∈ [0,1]` — it is dead code, so `jump_n_sample` just repeats the same
  DDIM step and shrinks `z_t`. It also blends the *noised* `z_gt` into an *x0*
  prediction, which is not what RePaint does.
- `ddim_sample_structure_logsnr_cosine_inpaint` has the same dead guard, hiding a
  `NameError` (`noised_data` is referenced before assignment) that would fire if
  it ever ran.
- The structure models here are **ε-prediction, discrete DDPM (T=1000, linear
  betas), operating directly on the split grid**. The current
  `train_structure_diffusion.py` is **x0-prediction, continuous log-SNR, on a
  VoxelVAE latent**. The `sample_ddpm*` family cannot be pointed at the new
  checkpoints.
