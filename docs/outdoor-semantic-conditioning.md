# Why the outdoor semantic stage cannot be inpainted by latent pinning

## What was wrong

`scripts/export_web_scenes_kitti.py --mode complete` built a known latent and a
mask, handed them to the structure sampler, and then called the
**unconditional** semantic sampler, `ddim_sample_logsnr_cosine`. The
observation never entered the semantic stage.

Measured on SemanticKITTI 06:000930, dense volumes, nothing dropped for
visibility (15,652 scan hits):

| | |
|---|---|
| inside a node the structure stage kept | 1.000 |
| occupied after the semantic decode | 0.647 |
| same label as the scan, of those occupied | 0.584 |

The structure inpaint is exact — every observed hit survives it. The semantic
decode then labels 35% of them class 0, which is *free*, so they vanish. That
is why the figures looked like the scan was being overwritten: it was, in the
second stage.

The extension path has the same defect. Of the half it is told to keep, 70–75%
is still occupied and 78–80% keeps its label.

Indoor is unaffected: `pipeline.py` pins both stages, via `ddim_inpaint_dense`
on the structure latent and `ddim_inpaint_graph` with a per-node mask on the
semantics. On `scene_03`, pinning the semantic stage is worth observed accuracy
0.929 against 0.843 and occupancy IoU 0.743 against 0.686.

## A bug underneath all of this

`_dda_trace_mark_free` stopped a ray as soon as it met a voxel another ray had
already carved, on the reasoning that the rest of the path was done. That holds
only if the earlier ray continued along *this* ray's remaining path, and rays
leaving one origin in different directions do not, so it under-carved. Worse,
`_carve` runs the rays under `prange`, so which ray reached a voxel first
depended on thread scheduling: **two calls in one process differed by 35,244
voxels**. Every measurement taken before this was fixed carries that noise, and
an earlier revision of this document wrote the resulting +/-0.01 spread off as
"the ray casting is not bit-reproducible" without asking why.

Writing only `-1 -> 0` and never reading back makes the pass idempotent and
order-independent. The observed fraction of the volume goes from 0.099 to
0.166 -- the sweep had been seeing 40% less free space than it measured -- and
results are now reproducible. The numbers below are post-fix.

## Why the indoor fix does not port

The obvious repair is to copy `inpaint_semantics`: encode the scan's labels onto
the completed octree, build a node mask, call `ddim_inpaint_graph`. A node at
depth 6 owns a `patch_size x patch_size` tile in XY and one voxel in Z, so the
mask needs an aggregation rule over the tile — and a single latent vector
encodes the whole tile jointly, so the only sound rule is AND: pin a node only
when every voxel of its tile was observed.

Two measurements say that does not work outdoors.

**1. AND discards about half the observation.** Outdoor `patch_size` is 4, so a
tile is 16 voxels against 4 indoors. Across four frames:

| frame | nodes with any voxel observed | with **all** observed | hit-bearing nodes fully observed | scan hits inside an AND-pinned node |
|---|---|---|---|---|
| 06:000930 | 0.209 | 0.117 | 0.420 | 0.536 |
| 06:000465 | 0.175 | 0.083 | 0.422 | 0.581 |
| 05:002100 | 0.147 | 0.083 | 0.418 | 0.543 |
| 02:000210 | 0.198 | 0.118 | 0.345 | 0.527 |

A hit-bearing node has on average 71-72% of its tile observed. AND would throw
away a little under half the hits -- a real cost, but on its own not
disqualifying. (Before the carve fix these read 0.006-0.007 fully observed and
0.24-0.30 of hits, which made AND look hopeless for the wrong reason.)

**2. The latent is not a function of its own tile.** `extract_code` is a graph
network over the dual octree, so a node's latent has a receptive field spanning
its neighbours. Encoding the ground-truth labels, then the same labels with
everything unobserved zeroed, on *the same octree*, and comparing latent rows
(06:000930, 25,024 nodes at depth 6, latent dim 32, relative L2):

| node | n | median relative latent error |
|---|---|---|
| own tile fully observed | 4,980 | **0.611** |
| own tile partly observed | 5,275 | 0.662 |
| own tile unobserved | 14,769 | 0.667 |

Encoding the same input twice differs by 0.00000, so 0.611 is signal, not
sampling noise. Even where the tile is entirely observed the latent sits 61%
away from its target, and **the error does not fall with distance from the
unobserved region**: nodes at Chebyshev distance 2 from any partly-observed
node measure 0.624, no better than those at distance 1. The receptive field is
simply wider than the observed region ever gets, so no erosion radius buys a
clean interior and no aggregation rule over the tile fixes this.

This is also the likely reason indoor observed accuracy is 0.93 rather than
~0.99 despite the semantic VAE reconstructing held-out patches at 0.9995: the
same effect, milder, because a 2x2 tile needs a quarter as much of the
neighbourhood to be right.

## What was implemented: decoder-space guidance

Latent pinning fails because it needs the *encoder* to be patch-local. Guidance
never touches the encoder. At each DDIM step the x0 prediction is decoded to
label logits, a masked cross-entropy against the scan's labels is evaluated at
the observed voxels, and the gradient is backpropagated to the latent:

```
x0 <- x0 - weight * (|x0| / |g|) * g,   g = d/dx0 CE(decode(x0), scan)[observed]
```

The gradient is renormalised against x0's own norm, so `weight` means "fraction
of the latent's magnitude moved per step" and carries across noise levels and
scenes, which a bare step size would not.

Targets are built per node, not in voxel space: `reconstruct_voxel_from_patch`
reads the decoder's logits with `octree.xyzb(DEPTH, nempty=False)`, so indexing
the patched label and known volumes the same way puts targets in the decoder's
own row order with nothing to misalign. No tile-completeness is required and no
aggregation rule is needed -- a tile with 3 of 16 voxels observed contributes
those 3.

Because label 0 is *free*, pushing an observed voxel toward its true class also
stops the decode from deleting it, so one mechanism fixes both failures.

`octree_diff/diffusion/sampling.py: ddim_guided_graph, guidance_step`. It uses
`loss.backward()` rather than `torch.autograd.grad`, because the graph
convolutions checkpoint themselves through `ckpt_conv_wrapper` with
`use_reentrant=True` and that is incompatible with `grad()`; the exporter
freezes the decoder's parameters first so the pass only reaches the latent.

> **The three tuning sections that follow were measured before the carve bug was
> fixed**, so their absolute numbers are low and noisy. They are kept because
> the mechanisms they identify -- the free-space term causing speckle, ring
> aliasing, what tapering does -- were each re-checked afterwards and still
> hold. The settings actually shipped, and their post-fix numbers, are in
> "Where this landed" below.

### Measured, 06:000930, 15,652 scan hits

Of the scan's hits: how many survive the semantic decode, and how many keep
their label. The `weight 0` row is mode 3 under the same sampler, so the
comparison isolates guidance.

| weight | inner | hits kept | label kept | seconds |
|---|---|---|---|---|
| legacy sampler, no guidance | - | 0.647 | 0.584 | 31 |
| 0.00 | 1 | 0.646 | 0.650 | 31 |
| 0.05 | 1 | 0.682 | 0.729 | 31 |
| 0.10 | 1 | 0.700 | 0.825 | 31 |
| 0.20 | 1 | 0.788 | 0.881 | 31 |
| 0.30 | 1 | 0.865 | 0.862 | 32 |
| 0.10 | 2 | 0.939 | 0.941 | 52 |
| 0.10 | 3 | 0.984 | 0.965 | 72 |
| 0.10 | 4 | **0.996** | 0.971 | 91 |
| 0.05 | 4 | 0.971 | **0.978** | 92 |
| 0.15 | 3 | 0.855 | 0.966 | 71 |
| 0.20 | 2 | 0.983 | 0.907 | 51 |

`weight` above 0.10 does not help and can hurt -- `0.15 / 3` fell to 0.855 hits
kept. Inner steps are what buy agreement: recomputing the gradient after moving
is worth far more than moving further along a stale one. Returns flatten after
inner 3.

**Read these rows with their noise band.** The `numba prange` ray casting in
`observed_state` is not bit-reproducible, so `occ` -- and therefore the
structure completion and everything after it -- differs between runs. The
guided configurations are stable to about +/-0.01, because guidance is what
determines the labels. The *unguided* ones are not: the same configuration
measured 0.650 label agreement once and 0.570 another time, because nothing
constrains them. Do not compare unguided rows across runs.

### What the observation should be asked to enforce

Guidance applied to every *known* voxel -- the hits plus everything ray
traversal cleared -- roughly doubled the label speckle: the fraction of
occupied voxels sharing no label with any neighbour went from 0.0108 unguided
to 0.0207, against 0.0074 in this frame's ground truth, and isolated voxels
rose from 0.0293 to 0.0449.

The cause is a mismatch in what the loss was made of. Inside the completed
octree on 06:000930 the guidance objective covers 80,627 voxels, of which
15,652 are hits and 64,975 are observed-free: **81% of it was "be empty"**.
That is the wrong thing to ask here. The structure stage has already pinned
occupancy from the same observation, exactly, at 100% of hits. What guidance
uniquely supplies is the *labels*, and a single sweep's hits are a sparse
scatter across surfaces -- so inside one 4x4 patch the objective was saying
"voxel 3 is a car, the other 15 are free", which is a recipe for occupied
voxels with no same-label neighbour.

`--guidance-free-weight` scales the observed-free voxels down. One run, all
rows comparable, `weight 0.10` and `inner 3`:

| free weight | hits kept | label kept | occupied | incoh | isol |
|---|---|---|---|---|---|
| truth, this frame | - | - | 0.0570 | 0.0074 | 0.0281 |
| unguided | 0.643 | 0.570 | 0.0771 | 0.0108 | 0.0293 |
| 1.00 | 0.934 | 0.972 | 0.0672 | 0.0214 | 0.0447 |
| 0.10 | 0.999 | 0.982 | 0.0714 | 0.0192 | 0.0384 |
| 0.05 | 0.978 | 0.993 | 0.0649 | 0.0208 | 0.0429 |
| 0.02 | 0.986 | 0.995 | 0.0682 | 0.0192 | 0.0396 |
| **0.00** | **0.998** | **0.992** | 0.0789 | **0.0139** | **0.0266** |

Dropping the free-space term removes the coherence cost entirely -- 0.0139
incoherence against the unguided baseline's 0.0108, and 0.0266 isolated voxels
which is *below* both the unguided 0.0293 and the truth's 0.0281 -- while
fidelity goes up rather than down. On these two metrics there is no trade to
make -- but see the next section, which is where those metrics run out.

The response is close to binary. A free weight of 0.02 already sits at 0.0192
incoherence, most of the way to the unweighted 0.0214, so this is not a knob to
tune: either the observation is asked for labels or it is also asked to
re-derive emptiness. Occupancy moves the other way, 0.065 at free 0.05 against
0.079 at free 0.00, but the unguided baseline is itself 0.077, so free 0.00 is
not worse than doing nothing on that axis -- the over-occupancy relative to the
0.057 ground truth is a pre-existing property of the completion, not something
guidance introduces.

### Two frames, and how many inner steps

Tuned on one frame, a number tends to work on that frame. `free_weight 0`
checked on a second, with its own unguided row:

| frame | config | hits kept | label kept | occupied | incoh | isol | sec |
|---|---|---|---|---|---|---|---|
| 06:000930 | truth | - | - | 0.0570 | 0.0074 | 0.0281 | - |
| | unguided | 0.650 | 0.632 | 0.0780 | 0.0097 | 0.0271 | 11 |
| | free0 inner 1 | 0.978 | 0.936 | 0.0845 | 0.0131 | 0.0247 | 31 |
| | free0 inner 2 | 0.999 | 0.988 | 0.0856 | 0.0129 | 0.0242 | 52 |
| | free0 inner 3 | 1.000 | 0.993 | 0.0861 | 0.0120 | 0.0236 | 73 |
| 05:002100 | truth | - | - | 0.0548 | 0.0068 | 0.0255 | - |
| | unguided | 0.674 | 0.539 | 0.0803 | 0.0114 | 0.0345 | 11 |
| | free0 inner 1 | 0.991 | 0.980 | 0.0861 | 0.0129 | 0.0314 | 32 |
| | free0 inner 2 | 0.999 | 0.989 | 0.0867 | 0.0116 | 0.0309 | 54 |
| | free0 inner 3 | 0.990 | 0.997 | 0.0875 | 0.0108 | 0.0295 | 75 |

The pattern is the same on both. Worth noting what this table shows that the
single-frame ones could not: **isolated voxels are consistently *lower* under
guidance than without it** -- 0.0236 against 0.0271, and 0.0295 against 0.0345
-- and on 05:002100 at inner 3 even incoherence comes in below the unguided
baseline, 0.0108 against 0.0114. Guidance with no free-space term does not
degrade local structure; it slightly improves it, which is what should be
expected from supplying real labels instead of sampling them.

`inner 3` is better than `inner 2` on fidelity *and* coherence, so it is the
default; the only cost is 20s per scene, which does not matter at the scale
these are exported. `inner 1` is enough on one frame and not the other, so it
is not a safe default.

The one axis where guidance does cost something is total occupancy: about
+0.007, from 0.078-0.080 unguided to 0.085-0.088. Both are well above the
0.055-0.057 ground truth, so the over-density is a property of the completion
itself rather than of guidance, but guidance does add to it.

### Ring aliasing, and why the metrics missed it

With the free-space term gone, `incoherent` sat at 0.0120 against an unguided
0.0097 and `isolated` came in *below* the unguided baseline. On those numbers
guidance looked free. Rendered, it was not: the completion showed concentric
magenta arcs through the road and small off-class patches scattered where the
unguided version had clean regions.

The cause is spatial aliasing of the observation. A LiDAR sweep's returns lie
on rings. Guidance pins labels on the rings; the voxels *between* rings were
never observed and follow the prior; where the two disagree the result is
striping at the ring frequency.

`incoherent` asks whether an occupied voxel shares a label with any neighbour,
so a coherent 10-voxel blob of the wrong class passes it, and a stripe one ring
wide passes it along its whole length. The metric is blind at the scale the
artifact lives at. **This is worth remembering before trusting either number
again: they bound single-voxel noise and say nothing about regional structure.**
Judge a change here by rendering it.

### Tapering, which fixes the aliasing

Stopping guidance before `t=0` leaves unguided DDIM steps at the end, which is
a low-pass in diffusion time: the model resolves across the stripes. Measured
on 06:000930, with the rendering judged by eye:

| weight | inner | stop_t | hits kept | label kept | appearance |
|---|---|---|---|---|---|
| 0.10 | 3 | - | 1.000 | 0.991 | strong ring striping |
| 0.10 | 3 | 0.20 | 0.968 | 0.935 | cleaner, arcs remain mid-scene |
| 0.10 | 3 | 0.25 | 0.943 | 0.904 | cleaner again |
| **0.15** | **3** | **0.35** | **0.964** | **0.943** | **arcs largely gone; the default** |
| 0.15 | 4 | 0.40 | 0.955 | 0.941 | clean, one odd off-class band |
| 0.03 | 2 | 0.30 | 0.740 | 0.624 | cleanest guided, but no better than unguided |
| - | - | - | 0.650 | 0.632 | unguided; cleanest, labels ignore the scan |

Raising the weight pays for stopping earlier, which is what makes `0.15 / 0.35`
work: a third of the trajectory is left unguided for smoothing and fidelity
still lands at 0.943, against 0.904 for the weaker `0.10 / 0.25`. Guiding to
`t=0` maximises fidelity and looks worst, so the useful setting is not the one
that maximises the fidelity number.

An earlier revision of this document called tapering a negative result and
defaulted it off. That was decided on `incoherent` and `isolated` alone, before
anything had been rendered, and it was wrong.

## Where this landed

Post-fix, one frame (06:000930), all rows from reproducible runs:

| config | hits kept | label kept |
|---|---|---|
| unguided (mode 3) | 0.601 | 0.660 |
| weight 0.15, inner 3, no taper | 1.000 | 0.986 |
| **weight 0.15, inner 3, stop_t 0.20, ramp 0.15** | **0.995** | **0.987** |
| weight 0.15, inner 3, stop_t 0.35, ramp 0.20 | 0.959 | 0.933 |

Tapering to 0.20 now costs nothing measurable against no taper and keeps an
unguided tail as insurance against ring aliasing, so it is the default. Going
further to 0.35 does cost fidelity. With the observation carved correctly the
structure stage is far better constrained and the ring artifact is mild even
untapered -- the earlier finding that guidance made the rendering visibly worse
was largely an artifact of the under-carved observation.

The four exported frames, at those defaults:

| frame | volume observed | hits kept | label kept |
|---|---|---|---|
| 06:000465 | 0.128 | 0.998 | 0.994 |
| 06:000930 | 0.166 | 0.995 | 0.987 |
| 05:002100 | 0.114 | 0.998 | 0.991 |
| 02:000210 | 0.161 | 0.998 | 0.996 |

Mode 5 holds its kept half at 0.997-0.998 occupied and 0.995 label, against
0.70-0.75 and 0.78-0.80 when the semantic stage was unconditioned.

## If guidance is not enough: a learned partial encoder

Train `E(partial_labels, known_mask) -> z` against the frozen
`extract_code(full_labels)`. Because the target is the full-label latent, it
learns to undo exactly the contamination measured above. Training data is free:
take ground-truth frames and simulate sweep masks with the existing ray casting
in `observed_state`. The cost is that it *is* a conditioning branch, so the
paper's framing would have to change -- which is why guidance was tried first.

`--mode complete-semantic` is the guided path and `--mode complete-geometry` the
unguided one; they are separate modes rather than one `complete`, so a figure
can no longer be captioned as the first while being produced by the second. See
`octree_diff/inference/modes.py`.

## Reproducing

The ray casting in `observed_state` runs under `numba` with `prange`, so the
observed-set fractions move by a few parts in ten thousand between runs. Read
the coverage table to two significant figures.
