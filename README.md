# Octree Diffusion for Semantic Scene Generation and Completion

Official implementation of **Octree Diffusion for Semantic Scene Generation and
Completion**, accepted at **ICRA 2026**.

Xujia Zhang, Brendan Crowe, Christoffer Heckman · [arXiv:2509.16483](https://arxiv.org/abs/2509.16483)

A structured generative framework for semantic 3D scene generation and completion.
Scenes are represented as octrees rather than dense voxel grids, which is what makes
diffusion over large 3D environments tractable: the model spends capacity only where
there is geometry. Generation is factored into two stages — a **structure** model that
decides *where* surfaces are, and a **semantic** model that decides *what* they are,
conditioned on the geometry the first stage produced.

The same machinery covers three tasks: unconditional **generation**, **semantic scene
completion** from a partial observation, and unbounded **scene extension**.

---

## Status

The code is complete and runs end to end for both the indoor (Replica) and outdoor
(SemanticKITTI) halves.

**The indoor weights in this repo are a retrain, not the originals used for the
paper** — those were lost. The retrain follows the same recipe with class-weighted
semantics; `configs/CHANGELOG.md` records every config difference between the shipped
model and this one. Measured on 10 held-out patches / 8 samples via
`octree-diff verify`:

| | |
|---|---|
| structure VAE round-trip IoU | 0.9999 |
| semantic VAE accuracy (all / occupied voxels) | 0.9897 / 0.9661 |
| generated occupied fraction (real: 0.1802) | 0.1877 |
| distinct classes in samples | 88 / 92 |
| completion: structure IoU on the observed half | 1.0000 |
| completion: semantic accuracy, observed / occluded | 0.9104 / 0.3488 |

For reference, samples from the original lost model score 87/92 classes on the same
measure, so the retrain is comparable on class coverage.

Semantic training uses `inv_sqrt` class weighting. Against an unweighted first
round, on 40 held-out patches (`octree_diff.inference.compare_sem_vae`):

| | micro acc | macro acc | mIoU |
|---|---|---|---|
| uniform | 0.9741 | 0.7130 | 0.6497 |
| `inv_sqrt` | 0.9693 | **0.9395** | **0.7873** |

Trading 0.005 micro accuracy for 0.23 macro is the intended behaviour: rare classes
go from never predicted to mostly correct.

---

## Installation

The environment builds on [OctFusion](https://github.com/octree-nn/octfusion).

```bash
git clone https://github.com/XUJIAZHANG2002/Octree-Scene-Diffusion.git
cd Octree-Scene-Diffusion
conda create -n octfusion python=3.11 -y && conda activate octfusion
```

> **Install torch and torchvision first, together, from the CUDA index.**
> `ocnn` depends on `torchvision`. If `torchvision` is absent when `ocnn` is
> installed, pip resolves it from PyPI and pulls a newer `torch` with it, silently
> replacing a working CUDA build with one your driver cannot run. This is the single
> most common way to end up with `torch.cuda.is_available() == False` here.

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

For **indoor inference only**, `requirements-inference.txt` is a much smaller set
with no Open3D, Numba or scikit-learn:

```bash
conda create -n octree-nb python=3.11 -y && conda activate octree-nb
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-inference.txt && pip install -e .
```

---

## Data

**Indoor — Replica.** Get it from
[facebookresearch/Replica-Dataset](https://github.com/facebookresearch/Replica-Dataset),
then produce the voxel patches (`[64,128,128]` int32 label volumes plus metadata) that
the pipeline consumes. Point `data/patches_128x128x64_s16` and `data/patches_val` at
the train and val splits — symlinks are fine.

Two further inputs are **derived**, and a fresh clone will fail without them:

```bash
# structure training set: [1,32,32,32] split grids in {-1,+1}
python -m octree_diff.data.build_split_dataset --src data/patches_128x128x64_s16 --dst data/split_outputs

# class frequencies and loss weights
python -m octree_diff.data.compute_class_weights --basis loss
```

> `--basis loss` matters. It counts only the voxels inside occupied octree nodes —
> exactly what `compute_semantic_loss` supervises. Counting over the whole dense
> volume instead yields weights dominated by free space and measurably worse
> training. Class 0 (free space) must keep a non-zero weight; zeroing it removes all
> gradient on empty space and degrades the result.

**Outdoor — SemanticKITTI.** Velodyne point clouds (80 GB) from
[semantic-kitti.org](https://www.semantic-kitti.org/dataset.html).

---

## Training

Four models in two stages. `patch_size` is **2 for indoor, 4 for outdoor** — set it in
`configs/sem_vae_config.yaml`.

**The order matters.** Each diffusion stage loads the VAE checkpoint named by its
`vae_ref.checkpoint`, so its VAE must have finished first. Running them out of order
now fails at startup with an explicit message rather than training against a missing
or stale VAE.

```bash
# Stage A — structure
octree-diff train --stage str_vae      # occupancy latent space
octree-diff train --stage str_diff     # generates structural latents

# Stage B — semantics
octree-diff train --stage sem_vae      # octree-based semantic features
octree-diff train --stage sem_diff     # labels conditioned on structure
```

Both accept `--config-dir` and `--device`. Configs live in `configs/`; the key names
are checked for consistency across files at startup — a UNet width that the VAE's
latents cannot feed, or a `vae_ref.checkpoint` that disagrees with the VAE config
that writes it, is reported as an error instead of producing a checkpoint that loads
but is wrong.

---

## Inference

```bash
# unconditional generation
octree-diff generate --num-samples 8 --steps 100 --out out/gen

# semantic scene completion from a partial observation
octree-diff complete --input scene.pt --known-mask known.pt --best-of 3 --out out/complete

# unbounded extension along +x
octree-diff extend --input scene.pt --steps-out 4 --best-of 3 --out out/extend

# reproduce the numbers in the Status table
octree-diff verify
```

`--input` is a `[128,128,64]` integer label tensor; `--known-mask` is a `[128,128,64]`
bool tensor where `True` means observed. Each run writes a `manifest.yaml` recording
the checkpoints, seed, sampler settings and latent scales it used.

**Use `--best-of 3` for completion and extension.** Roughly one structure draw in
twenty collapses. It is detectable without ground truth — a degraded draw disagrees
with the *observed* region, which a good draw reproduces almost exactly — so sampling
a few layouts and keeping the best one removes the failure mode. Extension needs it
most, because each step conditions on the previous one and a bad step corrupts
everything after it.

> **Latent scaling is part of the checkpoint contract.** The diffusion schedule
> assumes roughly unit-variance latents, but both VAEs produce latents with standard
> deviation well below 1. Latents are multiplied by `latent_scale` during training and
> **must be divided by it again before decoding**. The value is written to a
> `*_meta.yaml` sidecar next to each diffusion checkpoint at training time.
> If that sidecar is missing, the loader silently falls back to `1.0` and everything
> still runs — producing over-scaled latents that decode to plausible-looking rubbish.
> **Ship the sidecars with the weights.**

The pipeline is also usable directly:

```python
from octree_diff import IndoorPipeline
pipe = IndoorPipeline("cuda")
```

---

## Notebooks

Interactive walkthroughs, saved with outputs so they can be read without running.

**`notebooks/indoor/`** — runs in the lean inference environment, needs nothing but
repo data and the checkpoints in `saved_model/`.

- `01_scene_completion_from_sensor_view.ipynb` — load a ground-truth patch, place a
  sensor, ray-cast to determine visibility, discard everything occluded, complete the
  scene, and score against the ground truth it never saw.
- `02_generation_and_extension.ipynb` — unconditional sampling, re-labelling one
  layout several times, a population comparison against real patches, and sliding +x
  extension with seam diagnostics.

**`notebooks/kitti/`** — the outdoor notebooks the paper's results came from. These
need the `octfusion` environment (Open3D, Numba, scikit-learn), the outdoor
checkpoints, and:

```bash
export OCTREE_DIFF_KITTI_WEIGHTS=/path/to/kitti/checkpoints
export OCTREE_DIFF_KITTI_DATA=/path/to/semantic-kitti
```

See `notebooks/README.md` for what each contains and which functions in them are
known-broken drafts.

---

## Weights

Not in the repo. The indoor release contains the four checkpoints, their
`*_meta.yaml` latent-scale sidecars, the configs used, and sample scenes. Unpack
`weights/*` into `saved_model/`.

---

## Known issues

- The indoor weights are a retrain; see **Status**.
- `notebooks/kitti/01_completion_ddim_logsnr.ipynb` cell 18 raises `AttributeError`
  on a stale diagnostic line. Nothing downstream depends on it.
- Several functions inside the KITTI notebooks are superseded drafts —
  `notebooks/README.md` lists them.
- Open3D visualisation needs a display. Headless it warns and returns, so the
  notebooks still run; you just get no interactive 3D window.

---

## Citation

```bibtex
@misc{zhang2026octreediffusionsemanticscene,
      title={Octree Diffusion for Semantic Scene Generation and Completion},
      author={Xujia Zhang and Brendan Crowe and Christoffer Heckman},
      year={2026},
      eprint={2509.16483},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.16483},
}
```

## Acknowledgements

This codebase builds directly on two projects, and carries code derived from both:

- **[OctFusion](https://arxiv.org/abs/2408.14732)** — the dual-octree graph networks
  under `octree_diff/models/` and much of `octree_diff/octree/`.
- **[SemCity](https://arxiv.org/abs/2403.07773)** — the semantic scene completion
  formulation and the outdoor data handling.

The dual-octree network code originates with Peng-Shuai Wang's Dual Octree Graph
Networks (MIT licence).
