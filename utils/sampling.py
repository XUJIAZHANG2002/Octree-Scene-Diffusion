"""Samplers for the retrained (x0-prediction, continuous log-SNR) models.

Kept separate from `utils/util_sample_stuff.py` on purpose: the notebooks in
`notebooks/kitti/` depend on that module's current behaviour, and two of its
functions do not do what their names say.

Everything here assumes the parameterisation the training scripts actually use:
the model predicts x0, and noise is added as `z_t = alpha_t*x0 + sigma_t*eps`
with alpha/sigma from `log_snr_schedule_cosine`.
"""

import torch

from utils.util_sample_stuff import log_snr_schedule_cosine, log_snr_to_alpha_sigma


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
