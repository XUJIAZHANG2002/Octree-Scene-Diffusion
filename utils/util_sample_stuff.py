import torch
from torch import nn
import math
@torch.no_grad()
def ddim_sample_structure_logsnr_cosine(z_T, model, S=50, predict_mode="x0"):
    """
    DDIM sampling using logSNR cosine schedule and x₀ prediction.

    Args:
        z_T (torch.Tensor): Starting noise [B, C, H, W, D]
        model (nn.Module): The trained diffusion model
        doctree: Optional structure for conditioning (can be None)
        S (int): Number of inference steps
        predict_mode (str): Must be "x0" for now
    Returns:
        z_t: final denoised sample [B, C, H, W, D]
    """
    assert predict_mode == "x0", "Only x0 mode is supported for now."

    # === Setup DDIM timestep schedule ===
    t_steps = torch.linspace(1., 0., S + 1, device=z_T.device)  # [S+1]
    t_pairs = list(zip(t_steps[:-1], t_steps[1:]))              # [(t_s, t_s-1)]

    z_t = z_T

    for t, t_next in t_pairs:
        # Shape [B], values = current time
        t_tensor = torch.full((z_t.shape[0],), t, device=z_t.device)
        t_next_tensor = torch.full((z_t.shape[0],), t_next, device=z_t.device)

        logsnr_t     = alpha_cosine_log_snr(t_tensor)
        logsnr_next = alpha_cosine_log_snr(t_next_tensor)

        alpha_t, sigma_t = log_snr_to_alpha_sigma(logsnr_t)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(logsnr_next)

        # Reshape for broadcasting: [B, 1, 1, 1, 1]
        alpha_t     = alpha_t[:, None, None, None, None]
        sigma_t     = sigma_t[:, None, None, None, None]
        alpha_next  = alpha_next[:, None, None, None, None]
        sigma_next  = sigma_next[:, None, None, None, None]

        # === Predict x₀ ===
        x0_pred = model(z_t, timesteps=t_tensor)

        # === DDIM deterministic update ===
        # No stochastic noise added → purely deterministic
        z_t = alpha_next * x0_pred + sigma_next * (z_t - alpha_t * x0_pred) / sigma_t

    return z_t


# @torch.no_grad()
# def ddim_sample_structure_logsnr_cosine_inpaint(z_T, model, mask, S=50, predict_mode="x0"):
#     """
#     DDIM sampling using logSNR cosine schedule and x₀ prediction.

#     Args:
#         z_T (torch.Tensor): Starting noise [B, C, H, W, D]
#         model (nn.Module): The trained diffusion model
#         doctree: Optional structure for conditioning (can be None)
#         S (int): Number of inference steps
#         predict_mode (str): Must be "x0" for now
#     Returns:
#         z_t: final denoised sample [B, C, H, W, D]
#     """
#     assert predict_mode == "x0", "Only x0 mode is supported for now."

#     # === Setup DDIM timestep schedule ===
#     t_steps = torch.linspace(1., 0., S + 1, device=z_T.device)  # [S+1]
#     t_pairs = list(zip(t_steps[:-1], t_steps[1:]))              # [(t_s, t_s-1)]

#     z_t = z_T
#     z_gt = mask.clone()

#     for t, t_next in t_pairs:
#         # Shape [B], values = current time
#         t_tensor = torch.full((z_t.shape[0],), t, device=z_t.device)
#         t_next_tensor = torch.full((z_t.shape[0],), t_next, device=z_t.device)

#         logsnr_t     = alpha_cosine_log_snr(t_tensor)
#         logsnr_next = alpha_cosine_log_snr(t_next_tensor)

#         alpha_t, sigma_t = log_snr_to_alpha_sigma(logsnr_t)
#         alpha_next, sigma_next = log_snr_to_alpha_sigma(logsnr_next)

#         # Reshape for broadcasting: [B, 1, 1, 1, 1]
#         alpha_t     = alpha_t[:, None, None, None, None]
#         sigma_t     = sigma_t[:, None, None, None, None]
#         alpha_next  = alpha_next[:, None, None, None, None]
#         sigma_next  = sigma_next[:, None, None, None, None]
        
#         # Inpaint
#         noised_z_gt = (z_gt * alpha_next + sigma_next * torch.randn_like(z_gt))

#         # === Predict x₀ ===
#         x0_pred = model(z_t, timesteps=t_tensor)
#         x0_pred = ( mask) * noised_z_gt + (1-mask) * x0_pred
    

#         # === DDIM deterministic update ===
#         # No stochastic noise added → purely deterministic
#         z_t = alpha_next * x0_pred + sigma_next * (z_t - alpha_t * x0_pred) / sigma_t

#     return z_t
@torch.no_grad()
def ddim_sample_logsnr_cosine(z_T, model, doctree, S=50, predict_mode="x0"):
    """
    DDIM sampling using cosine logSNR schedule and x₀ prediction.

    Args:
        z_T: Initial noise tensor [N, C]
        model: Trained diffusion model
        doctree: DualOctree structure
        S: Number of DDIM steps (e.g., 50)
        predict_mode: "x0" (only supported here)
    """
    assert predict_mode == "x0", "Only x0 mode is supported in this version."

    # === Step 1: Generate continuous t schedule [1.0, ..., 0.0]
    t_steps = torch.linspace(1., 0., S + 1, device=z_T.device)  # [S+1]
    t_pairs = list(zip(t_steps[:-1], t_steps[1:]))


    z_t = z_T
    for t, t_next in t_pairs:
        # Convert scalars to tensors
        t_tensor = t.view(1, 1)
        t_next_tensor = t_next.view(1, 1)

        logsnr_t = alpha_cosine_log_snr(t_tensor)
        logsnr_next = alpha_cosine_log_snr(t_next_tensor)

        alpha_t, sigma_t = log_snr_to_alpha_sigma(logsnr_t)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(logsnr_next)

        x0_pred = model(z_t, doctree=doctree, timesteps=t_tensor)  # Predict x0

        # DDIM update rule
        z_t = alpha_next * x0_pred + sigma_next * torch.randn_like(z_t)

    return z_t
def beta_linear_log_snr(t):
    return -torch.log(torch.special.expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s: float = 0.008, eps: float = 1e-5):
    return -torch.log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1 + eps)

# === LogSNR to alpha/sigma ===
def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

# === Sampling timesteps: for inference (not training) ===
def get_sampling_timesteps(batch, device, steps):
    times = torch.linspace(1., 0., steps + 1, device=device)
    times = times.repeat(batch, 1)  # shape: [B, steps + 1]
    times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)  # shape: [2, B, steps]
    times = times.unbind(dim=-1)  # tuple of length = steps, each [2, B]
    return times