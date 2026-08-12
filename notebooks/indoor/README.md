# Indoor (Replica) notebooks

**`01_scene_completion_from_sensor_view.ipynb`** — the verification notebook. Loads a
ground-truth patch, places a sensor in the room, ray-casts to work out what it can
actually see, discards everything occluded, completes the scene with the retrained
models, and scores the result against the ground truth it never saw. Saved with
outputs, so you can read it without running anything.

It is the indoor counterpart to `../kitti/03_completion_then_outpaint.ipynb`. The
outdoor notebooks carve their observed/unobserved mask by reverse ray-tracing from the
LiDAR origin; `utils/sensor_sim.py` is the indoor equivalent.

Supporting modules, both new:

- `utils/sensor_sim.py` — `pick_viewpoint`, `simulate_sensor` (tri-state free /
  observed / unknown), `observed_labels`
- `utils/vis_indoor.py` — inline matplotlib rendering (`plot_3d`, `plot_slice`,
  `plot_state`, `compare_row`) and `completion_metrics`. Deliberately avoids Open3D,
  which blocks on a GUI window and is useless over SSH.

Note: `utils/util_dualoctree.py` calls `matplotlib.use("Agg")` at import time, so a
notebook must re-enable `%matplotlib inline` **after** importing anything from the
pipeline or its figures will silently not render.

## Environment

The notebook runs in the conda env **`octree-nb`**, registered as the Jupyter kernel
*"Python (octree-nb)"* — pick that kernel, not the system python (which has no `ocnn`).

```bash
conda activate octree-nb
jupyter lab notebooks/indoor/01_scene_completion_from_sensor_view.ipynb
```

To rebuild it from scratch, see the header of `requirements-inference.txt`. The one
trap: **install `torch` and `torchvision` together from the CUDA index before
anything else.** `ocnn` depends on `torchvision`, so if it is missing pip resolves it
from PyPI and pulls a newer `torch` along with it, replacing the working cu121 build
with one the driver cannot run (`torch.cuda.is_available()` goes False).

The older `octfusion` env also works and is what the models were trained in; both
produce identical numbers on this notebook.

## The same thing from the command line

```bash
python -m inference_scripts.inference_indoor --mode inpaint \
    --input scene.pt --known-mask known.pt --repaint-resample 4 --best-of 3 \
    --out out/complete

python -m inference_scripts.verify_indoor      # non-interactive checks
```

## Label ids

The ids in this dataset are a **remapped** space, not raw Replica class ids — the
mapping was not preserved by the preprocessing, and applying Replica's own ids gives
implausible results (it makes "base-cabinet" and "basket" the two most common classes
in every room). The three dominant classes were identified geometrically instead:

| id | class | evidence |
|---|---|---|
| 1 | floor | mean height 1.2 of 64, covers 60% of the footprint |
| 2 | wall | mid-height, thin (6% footprint coverage) |
| 3 | ceiling | mean height 51.7 of 64, covers 66% of the footprint |

Everything else is shown by id.
