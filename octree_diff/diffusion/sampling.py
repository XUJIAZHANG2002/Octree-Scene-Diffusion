"""Samplers for the retrained (x0-prediction, continuous log-SNR) models.

Kept separate from `utils/util_sample_stuff.py` on purpose: the notebooks in
`notebooks/kitti/` depend on that module's current behaviour, and two of its
functions do not do what their names say.

Everything here assumes the parameterisation the training scripts actually use:
the model predicts x0, and noise is added as `z_t = alpha_t*x0 + sigma_t*eps`
with alpha/sigma from `log_snr_schedule_cosine`.
"""

import torch
import torch.nn.functional as F

from octree_diff.diffusion.util_sample_stuff import log_snr_schedule_cosine, log_snr_to_alpha_sigma


def alpha_sigma(t):
    """(alpha, sigma) for continuous time t in [0, 1]. t=0 is clean, t=1 is noise."""
    return log_snr_to_alpha_sigma(log_snr_schedule_cosine(t))


def _bcast(v, ref):
    """Reshape a per-sample scalar so it broadcasts against `ref`."""
    return v.view(-1, *([1] * (ref.dim() - 1)))


def time_pairs(steps, device):
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    return list(zip(ts[:-1], ts[1:]))


def ddim_step(z_t, x0_pred, a_t, s_t, a_next, s_next, eta=0.0):
    """One DDIM step from t to t_next. eta=0 is deterministic."""
    eps = (z_t - a_t * x0_pred) / s_t.clamp_min(1e-8)

    if eta > 0.0:
        # Standard DDIM stochasticity, in variance-preserving alpha/sigma form.
        ratio = (a_t / a_next.clamp_min(1e-8)).clamp(max=1.0)
        sigma_ddim = eta * (s_next / s_t.clamp_min(1e-8)) * (1.0 - ratio ** 2).clamp_min(0.0).sqrt()
        coeff = (s_next ** 2 - sigma_ddim ** 2).clamp_min(0.0).sqrt()
        return a_next * x0_pred + coeff * eps + sigma_ddim * torch.randn_like(z_t)

    return a_next * x0_pred + s_next * eps


def renoise(z_next, a_t, s_t, a_next, s_next):
    """Jump backwards from t_next to the noisier t (the RePaint 'resample' move)."""
    ratio = a_t / a_next.clamp_min(1e-8)
    var = (s_t ** 2 - (ratio ** 2) * (s_next ** 2)).clamp_min(0.0)
    return ratio * z_next + var.sqrt() * torch.randn_like(z_next)


# --------------------------------------------------------------------------
# dense latents  [B, C, D, H, W]  (structure)
# --------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample_dense(model, shape, steps=100, device="cuda", eta=0.0, z_T=None):
    z_t = torch.randn(shape, device=device) if z_T is None else z_T.to(device)
    for t, t_next in time_pairs(steps, device):
        tb = t.expand(z_t.shape[0])
        a_t, s_t = alpha_sigma(tb)
        a_next, s_next = alpha_sigma(t_next.expand(z_t.shape[0]))
        x0 = model(z_t, tb)
        z_t = ddim_step(z_t, x0,
                        _bcast(a_t, z_t), _bcast(s_t, z_t),
                        _bcast(a_next, z_t), _bcast(s_next, z_t), eta)
    return z_t


@torch.no_grad()
def ddim_inpaint_dense(model, z_known, mask, steps=100, device="cuda", eta=0.0,
                       resample=1, z_T=None):
    """RePaint-style inpainting. `mask` is 1 where `z_known` is trusted.

    `resample` is RePaint's jump count: at each step the pair (denoise, renoise)
    is repeated that many times, which lets the generated region reconcile with
    the known region instead of being overwritten by it. resample=1 disables it.
    """
    z_known = z_known.to(device)
    mask = mask.to(device).to(z_known.dtype)
    z_t = torch.randn_like(z_known) if z_T is None else z_T.to(device)

    for t, t_next in time_pairs(steps, device):
        tb = t.expand(z_t.shape[0])
        a_t, s_t = alpha_sigma(tb)
        a_next, s_next = alpha_sigma(t_next.expand(z_t.shape[0]))
        a_t, s_t = _bcast(a_t, z_t), _bcast(s_t, z_t)
        a_next, s_next = _bcast(a_next, z_t), _bcast(s_next, z_t)

        for r in range(resample):
            # known region carries the forward-noised ground truth at this level
            z_t = mask * (a_t * z_known + s_t * torch.randn_like(z_t)) + (1 - mask) * z_t

            x0 = model(z_t, tb)
            # blend the *clean* known latent into the x0 estimate
            x0 = mask * z_known + (1 - mask) * x0

            z_next = ddim_step(z_t, x0, a_t, s_t, a_next, s_next, eta)

            if r < resample - 1:
                z_t = renoise(z_next, a_t, s_t, a_next, s_next)
            else:
                z_t = z_next

    return mask * z_known + (1 - mask) * z_t


# --------------------------------------------------------------------------
# graph latents  [N_nodes, C]  (semantic, on a dual octree)
# --------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample_graph(model, doctree, n_nodes, latent_dim, steps=100,
                      device="cuda", eta=0.0, z_T=None):
    z_t = torch.randn(n_nodes, latent_dim, device=device) if z_T is None else z_T.to(device)
    for t, t_next in time_pairs(steps, device):
        tb = t.view(1)                       # 1-D, as the training loop passes it
        a_t, s_t = alpha_sigma(tb)
        a_next, s_next = alpha_sigma(t_next.view(1))
        x0 = model(z_t, doctree=doctree, timesteps=tb)
        z_t = ddim_step(z_t, x0, a_t, s_t, a_next, s_next, eta)
    return z_t


@torch.no_grad()
def ddim_inpaint_graph(model, doctree, z_known, mask, steps=100, device="cuda",
                       eta=0.0, resample=1, z_T=None):
    """As `ddim_inpaint_dense`, for per-node latents. `mask` is [N, 1] or [N, C]."""
    z_known = z_known.to(device)
    mask = mask.to(device).to(z_known.dtype)
    z_t = torch.randn_like(z_known) if z_T is None else z_T.to(device)

    for t, t_next in time_pairs(steps, device):
        tb = t.view(1)
        a_t, s_t = alpha_sigma(tb)
        a_next, s_next = alpha_sigma(t_next.view(1))

        for r in range(resample):
            z_t = mask * (a_t * z_known + s_t * torch.randn_like(z_t)) + (1 - mask) * z_t

            x0 = model(z_t, doctree=doctree, timesteps=tb)
            x0 = mask * z_known + (1 - mask) * x0

            z_next = ddim_step(z_t, x0, a_t, s_t, a_next, s_next, eta)

            if r < resample - 1:
                z_t = renoise(z_next, a_t, s_t, a_next, s_next)
            else:
                z_t = z_next

    return mask * z_known + (1 - mask) * z_t


# --------------------------------------------------------------------------
# decoder-space guidance (semantic completion where latent pinning cannot work)
# --------------------------------------------------------------------------

def guidance_weight_at(t, weight, stop_t=0.0, ramp=0.0, max_t=1.0):
    """Guidance strength at continuous time `t` (1 is noise, 0 is clean).

    Guidance pulls the latent toward satisfying a per-voxel objective, which is
    not the same as toward the model's manifold. Applied all the way to t=0 it
    leaves the final latent off-manifold, and the decoder then commits to it:
    measured on 06:000930, constant guidance doubles the fraction of occupied
    voxels sharing no label with any neighbour, 0.0108 to 0.0207. Stopping
    early leaves clean DDIM steps at the end to re-coherate what guidance
    forced, and costs nothing -- the skipped steps need no decoder pass.

    `stop_t` is where guidance ends, `ramp` how far above that it fades in, and
    `max_t` suppresses it at high noise where the x0 prediction is meaningless.
    """
    t = float(t)
    if t > max_t:
        return 0.0
    if ramp <= 0.0:
        return 0.0 if t < stop_t else weight
    return weight * min(1.0, max(0.0, (t - stop_t) / ramp))


def guidance_step(x0, decode_logits, target, voxel_weight, weight):
    """Nudge an x0 prediction so its decoding agrees with observed labels.

    Latent pinning needs a node's latent to be a function of its own patch. The
    outdoor semantic encoder is a graph network, so it is not: see
    `docs/outdoor-semantic-conditioning.md`. Guidance avoids the problem by
    never touching the encoder -- it measures the disagreement where the
    observation actually lives, in decoded label space, and backpropagates.

    `decode_logits(x0) -> [N, C, P, P]` is supplied by the caller so this module
    stays free of the VAE. `target` is [N, P, P] of class ids and `voxel_weight`
    is [N, P, P] of per-voxel loss weights, zero where nothing was observed.
    Weighting rather than masking is what lets the caller hold down the
    observed-free voxels, which outnumber the hits 4:1 and otherwise make the
    objective mostly "be empty".

    `weight` is the fraction of x0's norm moved per step: the raw gradient is
    renormalised, so the same weight means the same thing at every noise level
    and across scenes, which a bare learning rate would not.

    Uses `loss.backward()` rather than `torch.autograd.grad`, because the graph
    convolutions checkpoint themselves through `ckpt_conv_wrapper` with
    `use_reentrant=True`, and that is incompatible with `grad()`. Freeze the
    decoder's parameters before sampling so the backward pass only reaches `x`.
    """
    total = voxel_weight.sum()
    if weight == 0.0 or not bool(total > 0):
        return x0, float("nan")

    with torch.enable_grad():
        x = x0.detach().requires_grad_(True)
        logits = decode_logits(x)
        if logits.shape[0] != target.shape[0]:
            raise ValueError(
                f"decoder returned {logits.shape[0]} nodes but the target has "
                f"{target.shape[0]}; the node orders must match")
        per_voxel = F.cross_entropy(logits, target, reduction="none")
        loss = (per_voxel * voxel_weight).sum() / total
        loss.backward()
        grad = x.grad
    if grad is None:
        return x0, float(loss)

    gn = grad.norm()
    if not torch.isfinite(gn) or gn == 0:
        return x0, float(loss)
    return x0 - (weight * x0.norm() / gn) * grad, float(loss)


@torch.no_grad()
def ddim_guided_graph(model, doctree, decode_logits, target, voxel_weight,
                      n_nodes=None, latent_dim=None, steps=100, device="cuda",
                      eta=0.0, weight=0.1, inner=1, stop_t=0.0, ramp=0.0,
                      max_t=1.0, z_T=None, verbose=False):
    """DDIM on per-node latents, with the observation applied as guidance.

    Shape-compatible with `ddim_sample_graph`; pass `z_T` or give
    `n_nodes`/`latent_dim` so the initial noise can be drawn.

    `inner` repeats the guidance step within one DDIM step, which buys
    agreement without extra model evaluations -- only extra decoder passes, and
    recomputing the gradient after moving is worth much more than taking one
    larger step along a stale one. `stop_t`/`ramp`/`max_t` schedule the strength
    over the trajectory; see `guidance_weight_at`.
    """
    if z_T is None:
        if n_nodes is None or latent_dim is None:
            raise ValueError("pass z_T, or both n_nodes and latent_dim")
        z_t = torch.randn(n_nodes, latent_dim, device=device)
    else:
        z_t = z_T.to(device)

    target = target.to(device)
    voxel_weight = voxel_weight.to(device).float()
    guided_steps = 0

    for i, (t, t_next) in enumerate(time_pairs(steps, device)):
        tb = t.view(1)
        a_t, s_t = alpha_sigma(tb)
        a_next, s_next = alpha_sigma(t_next.view(1))

        x0 = model(z_t, doctree=doctree, timesteps=tb)

        w = guidance_weight_at(t, weight, stop_t, ramp, max_t)
        loss = float("nan")
        if w > 0.0:
            guided_steps += 1
            for _ in range(max(inner, 1)):
                x0, loss = guidance_step(x0, decode_logits, target,
                                         voxel_weight, w)

        z_t = ddim_step(z_t, x0, a_t, s_t, a_next, s_next, eta)

        if verbose and (i % max(steps // 10, 1) == 0 or i == steps - 1):
            print(f"    step {i + 1:3d}/{steps}  t={float(t):.3f}  w={w:.3f}  "
                  f"observed-voxel CE {loss:.4f}")

    if verbose:
        print(f"    guided {guided_steps}/{steps} steps")
    return z_t
