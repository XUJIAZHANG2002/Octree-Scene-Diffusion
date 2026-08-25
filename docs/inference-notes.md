> **Historical.** These notes were written before the code was reorganised into the
> `octree_diff` package, and the module paths below are the pre-restructure ones
> (`utils/`, `inference_scripts/`, `dataset/`). `inference_scripts/inference.py`, the
> script this document exists to warn about, has since been deleted. Kept as the
> record of *why*, and of what the outdoor notebooks each contain.

# Inference — state of play

Written during the Replica indoor retrain (2026-08-06): a record of what works,
what doesn't, and where the real code lives.

**Indoor inference now lives in `inference_scripts/inference_indoor.py`**
(generation / inpainting / +x outpainting, all headless) with checks in
`inference_scripts/verify_indoor.py`. `inference.py` below is the *outdoor*
script and is left untouched.

## `inference_scripts/inference.py` should not be trusted

It has never been run end to end and does not currently produce a sample. The
defects, in the order they bite:

1. **The denoising loops are no-ops.** Both loops do
   ```python
   eps   = (z - alpha * pred_x0) / sigma
   z     = alpha * pred_x0 + sigma * eps
   ```
   which algebraically simplifies to `z = z`. The model output is discarded and
   `z` stays at its initial noise for all 1000 steps. A DDIM step has to advance
   the schedule — take `alpha_next`/`sigma_next` from `t[i+1]` — exactly as
   `utils/util_sample_stuff.py::ddim_sample_structure_logsnr_cosine` (structure)
   and `::ddim_sample_logsnr_cosine` (semantic) already do. Those two helpers
   look correct; the script should call them instead of hand-rolling the loop.

2. **The structure decoder output is treated as a voxel grid.** `str_vae.decode`
   returns the *split* grid — a `[1,1,32,32,32]` tensor in {-1,+1} encoding
   depth-5 octree occupancy — not scene occupancy. The script feeds it to
   `get_non_empty_mask(..., 1)` and then `points2octree(depth=6)`, which builds
   an octree at the wrong resolution from the wrong quantity. The correct bridge
   is the one the notebooks use:
   ```python
   split = splitbig2split(split_big)              # [1,1,32,32,32] -> [1,8,16,16,16]
   octree = split2octree_small(split, 6, 4)       # input_depth=6, full_depth=4
   ```
   (Both live in `utils/util_dualoctree.py` / `utils/util_octree_stuff.py`, and
   the round trip is exact — verified against `dataset/build_split_dataset.py`.)

3. **`extract_code` is used where generation is intended.** `sem_vae.extract_code(octree)`
   *encodes* an octree that already carries semantic features; at generation time
   there are none. The notebooks instead build the dual octree directly:
   ```python
   octree_out = vae.create_child_octree(octree_in).cuda()
   doctree = DualOctree(octree_out); doctree.post_processing_for_docnn()
   z_T = torch.randn(doctree.total_num, LATENT_DIM).cuda()
   ```
   Note `doctree.total_num`, not `doctree[depth].npt` as the script uses.

4. **Decoding skips the patch reconstruction.** The script argmaxes `sem_voxs`
   directly; the notebooks call `reconstruct_voxel_from_patch(output['sem_voxs'],
   octree, depth=6, shape=..., patch_size=...)` to get labels back on the voxel
   grid. For indoor the shape is `(1, 128, 128, 64)` with `patch_size=2`.

5. **Latent scaling is not applied.** The retrained diffusion models are trained
   on VAE latents multiplied by a scale factor (see below), so sampled latents
   must be divided by it before `decode`/`decode_code`.

## Latent scale (new, from this retrain)

Both VAEs produce latents with std well under 1 (structure ≈ 0.50, semantic
≈ 0.34), which the cosine log-SNR schedule is not calibrated for, so the
diffusion training scripts now rescale to ~unit variance. The factor is written
next to each checkpoint:

- `saved_model/structure_diffusion_meta.yaml`
- `saved_model/unet_sem_diffusion_meta.yaml`

Read it with `utils/latent_scale.py::load_latent_scale` and divide sampled
latents by it before decoding. Diffusion checkpoints are the **EMA** weights;
the raw optimiser weights are saved alongside as `*_raw.pt`.

## Where the real inference code lives

All five notebooks are **outdoor / SemanticKITTI** (`velodene_to_voxel`,
`patch_size=4`, 256×256×32 grids, checkpoints from `~/SemCityOcto`). None of
them contain the Replica indoor path — that will have to be adapted, mainly
`patch_size` 4 → 2 and grid 256×256×32 → 128×128×64.

| notebook | what it holds |
|---|---|
| `test7.ipynb` | Earliest outpainting pass. `ddim_sample_structure_logsnr_cosine_inpaint`, `outpaint_scheduler`, `octree2voxel`. Also the `split_outputs` generation cell that `dataset/build_split_dataset.py` is based on. |
| `test8.ipynb` | DDPM outpainting + semantic inpainting. `build_ddpm_sampling_buffers`, `sample_ddpm`, `sample_ddpm_inpaint_repaint`, `sample_ddpm_inpaint_blended`, `outpaint_latent_canvas`, `outpaint_square_spiral`, `sem_paint_octree`, `split2splitbig`/`splitbig2split`. The cleanest full structure→semantic chain is here. |
| `test9.ipynb` | Outpainting from a known canvas ("## 1. outpaint structure", "## 2. outpaint sem"). Superset of test8's outpainting; adds canvas save/restore. |
| `test9 copy.ipynb` | Variant of test9 focused on repaint-style inpainting (`_build_repaint_times`, `ddim_sample_logsnr_cosine_inpaint`). Contains the loop that generated `split_outputs` for KITTI. |
| `test_cumuti.ipynb` | CU-Multi zero-shot / scene extension. `outpaint_structure_along_x` walks the scene along +x, 128 voxels at a time. Closest thing to the paper's scene-extension figure. |

Suggested cleanup when we get to it: lift `sample_ddpm*`, `outpaint_*` and
`sem_paint_octree` out of `test8`/`test_cumuti` into `utils/util_sample_stuff.py`
(where the two DDIM samplers already live), then rewrite `inference.py` on top of
them with a `--patch_size` / `--grid_shape` switch for indoor vs outdoor.

## What the indoor script does differently (2026-08-06)

Two things `inference_indoor.py` had to get right that the notebooks do not:

1. **The octree fed to the semantic stage.** `split2octree_small` returns an
   octree that is *full* at depth 4 (4096 nodes), because the split
   representation lives on the dense depth-4 grid. The semantic VAE was trained
   on `points2octree(..., full_depth=3)` octrees, which are sparse there (~1900
   nodes for a typical patch) — and the dual-octree graph the UNet consumes
   includes those coarse leaves, so the dense version is a different graph from
   anything seen in training. The notebooks route through `create_child_octree`,
   which keeps the dense layer. Extracting the depth-6 occupancy and rebuilding
   at `full_depth=3` reproduces the training octree exactly — verified identical
   `nnum` at every depth and identical `doctree.total_num` (42967).

2. **Conditioning the coarse latent rows.** The dual-octree latent is laid out
   `[leaves@3, leaves@4, leaves@5, all nodes@6]`. Masking only the depth-6 tail
   (the obvious reading of `offset = total_num - nnum[6]`) leaves every coarse
   leaf row unconditioned, and those rows feed the decoder's upsampling path — so
   noise from the generated region leaks across the whole scene. Conditioning all
   rows whose node lies inside the known region raised known-half accuracy from
   0.62 to 0.74 with an untrained stand-in model, and takes the known-node
   fraction from 0.44 to 0.64.

Verified so far (with `str_diff` mid-training and `sem_diff` not yet started):

- structure VAE round-trip IoU **1.0000**
- semantic VAE round-trip **0.937** on occupied voxels (`sem_vae` at epoch 5/20)
- structure inpainting preserves the known half at IoU **1.0000**
- semantic inpainting with `mask=1` reproduces the plain VAE round-trip
  bit-for-bit (0.8799 = 0.8799 on the same patch), i.e. the masking machinery is
  exact
- reference: the lost model's `data_gen/` samples are 5.4% occupied with 87/92
  classes present, top-5 `[2, 3, 1, 38, 24]` — which matches the training set's
  own top-5. That is the target for the retrained model's samples.
