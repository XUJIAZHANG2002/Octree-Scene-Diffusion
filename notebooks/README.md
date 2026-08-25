# Notebooks

These are the notebooks the paper's outdoor results were actually produced with.
For indoor (Replica) see `notebooks/indoor/`, which has two notebooks of its own.

Both halves are also driven from the CLI:

```bash
octree-diff generate | complete | extend | verify
```

Historical defect notes live in `docs/inference-notes.md`.

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
`octree_diff/data/build_split_dataset.py` is derived from, the numba reverse-ray-trace
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

## Running them

Fixed and verified: `01_completion_ddim_logsnr.ipynb` runs end to end against the
recovered `octree_diff/data/kitti/velodyne_to_voxel.py` and the original KITTI
checkpoints. What it needed:

1. **`patch_size=4` on `GraphVAE`.** Outdoor was trained with a 4x4 patch decoder;
   the default is 2, and the outdoor checkpoint will not load into it
   (`patch_sem_predict.upconv.*` shape mismatch). This was the single reason the
   outdoor VAE appeared unloadable.
2. **`visualize_structure` was defined inline in `02_`**, so the other notebooks
   called it undefined and only worked if `02_` had been run first in the same
   kernel. It now lives in `octree_diff/viz/open3d_viewers.py`.
3. **Imports and paths.** Module paths were rewritten for the package layout
   (`scripts/fix_notebook_imports.py`), and the machine-specific checkpoint and
   dataset roots are now environment variables:

   ```bash
   export OCTREE_DIFF_KITTI_WEIGHTS=/path/to/kitti/checkpoints   # default weights/kitti
   export OCTREE_DIFF_KITTI_DATA=/path/to/semantic-kitti          # default data/kitti
   ```

You need the **outdoor** checkpoints (`vae_fdepth6_ldim8_8.pt`,
`Model_is_good_8_6.pt`, `structure_ddpm_ep*.pt`) with `latent_dim=8`,
`num_classes=21`, `patch_size=4`. These notebooks will not load the Replica
checkpoints in `saved_model/`, and the reverse is equally true.

Use the `octfusion` environment, not the lean inference one — the outdoor path
needs `open3d`, `numba` and `scikit-learn`. Open3D's `draw_geometries` degrades
gracefully with no display (it warns and returns), so the notebooks run headless;
you just do not get the interactive 3D windows.

**Known remaining defect:** `01_`, cell 18 (`(occupancy_vis != gt).sum()`) raises
`AttributeError` — a stale diagnostic line comparing mismatched types. Nothing
downstream uses it. Pre-existing; left alone rather than guessed at.

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
  `octree_diff/training/train_structure_diffusion.py` is **x0-prediction, continuous log-SNR, on a
  VoxelVAE latent**. The `sample_ddpm*` family cannot be pointed at the new
  checkpoints.
