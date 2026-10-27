"""Ancestral DDPM sampling for the 1000-step outdoor structure model.

`SemCityOcto/checkpoints/structure_ddpm_ep110.pt` is the structure model the
paper's outdoor FID/KID notebook (`vae_fid_kid.ipynb`) and the outpainting
notebooks sample from. Unlike every other model here it is an epsilon-predictor
on a linear beta schedule with integer timesteps, so it gets its own sampler,
lifted from `notebooks/kitti/02_outpaint_ddpm_library.ipynb`.
"""

import torch


def ddpm_schedule(T=1000, device="cpu", beta_start=1e-4, beta_end=2e-2):
    """The linear-beta buffers ancestral sampling needs, as a dict of [T] tensors."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    abar_prev = torch.cat([torch.ones(1, device=device), abar[:-1]])
    posterior_var = betas * (1.0 - abar_prev) / (1.0 - abar)
    return {
        "T": T,
        "sqrt_abar": abar.sqrt(),
        "sqrt_one_minus_abar": (1.0 - abar).sqrt(),
        # the t=0 posterior variance is 0; clip its log as OpenAI/LDM do
        "posterior_log_var": torch.log(posterior_var.clamp(min=1e-20)),
        "coef_x0": betas * abar_prev.sqrt() / (1.0 - abar),
        "coef_xt": (1.0 - abar_prev) * alphas.sqrt() / (1.0 - abar),
    }


def _at(a, t, x):
    return a.gather(0, t).view(-1, *([1] * (x.dim() - 1)))


@torch.no_grad()
def sample_ddpm(model, shape, sched, device="cuda", x_T=None):
    """Full ancestral sampling, T-1 down to 0. Returns the final x0 estimate."""
    x = torch.randn(shape, device=device) if x_T is None else x_T.to(device)
    for step in reversed(range(sched["T"])):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        eps = model(x, timesteps=t)
        x0 = (x - _at(sched["sqrt_one_minus_abar"], t, x) * eps) / (
            _at(sched["sqrt_abar"], t, x) + 1e-8)
        if step == 0:
            return x0
        mean = _at(sched["coef_x0"], t, x) * x0 + _at(sched["coef_xt"], t, x) * x
        x = mean + torch.exp(0.5 * _at(sched["posterior_log_var"], t, x)) * torch.randn_like(x)


@torch.no_grad()
def inpaint_ddpm(model, x_known, mask, sched, device="cuda", resample=1, x_T=None):
    """RePaint-style inpainting: the known region follows q(x_t | x_known).

    At every step the known region is replaced by a fresh noising of `x_known`
    to the same level, and the model fills the rest. `resample > 1` repeats each
    step that many times, jumping back one step in between, which lets the
    unknown region adjust to what the known region imposes (RePaint's
    harmonisation) at `resample` times the cost. The known region of the result
    is exactly `x_known`.

    `mask` is 1 where known, broadcastable to `x_known`.
    """
    x_known, mask = x_known.to(device), mask.to(device).float()
    x = torch.randn_like(x_known) if x_T is None else x_T.to(device)
    T = sched["T"]
    abar = sched["sqrt_abar"] ** 2
    for step in reversed(range(T)):
        t = torch.full((x.shape[0],), step, device=device, dtype=torch.long)
        for r in range(max(resample, 1)):
            # condition: the known region at the current noise level
            noised = (_at(sched["sqrt_abar"], t, x) * x_known
                      + _at(sched["sqrt_one_minus_abar"], t, x) * torch.randn_like(x))
            x = mask * noised + (1.0 - mask) * x
            eps = model(x, timesteps=t)
            x0 = (x - _at(sched["sqrt_one_minus_abar"], t, x) * eps) / (
                _at(sched["sqrt_abar"], t, x) + 1e-8)
            if step == 0:
                return mask * x_known + (1.0 - mask) * x0
            mean = _at(sched["coef_x0"], t, x) * x0 + _at(sched["coef_xt"], t, x) * x
            x_prev = mean + torch.exp(0.5 * _at(sched["posterior_log_var"], t, x)) * torch.randn_like(x)
            if r < resample - 1:
                # one forward step back up to t: x_t ~ q(x_t | x_{t-1})
                beta = 1.0 - abar[step] / abar[step - 1]
                x = (1.0 - beta).sqrt() * x_prev + beta.sqrt() * torch.randn_like(x)
            else:
                x = x_prev
