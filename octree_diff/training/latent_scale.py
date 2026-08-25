import os

import torch
import yaml


def resolve_latent_scale(cfg_value, estimate_fn, meta_path, name):
    """Return the factor that maps VAE latents onto ~unit variance.

    The diffusion noise schedule assumes x0 has roughly unit variance; both VAEs
    here produce latents with std well below 1, so the latents are multiplied by
    this factor during training and must be divided by it again before decoding
    at inference time.

    `cfg_value` is either a number (use it verbatim) or "auto" (estimate it from
    a sample of the training set). The resolved value is written next to the
    checkpoint as a small sidecar so inference can pick it up.
    """
    if isinstance(cfg_value, str) and cfg_value.lower() == "auto":
        std = estimate_fn()
        scale = 1.0 / max(float(std), 1e-6)
        print(f"latent_scale ({name}): measured std {std:.4f} -> scale {scale:.4f}")
    else:
        scale = float(cfg_value)
        print(f"latent_scale ({name}): using configured value {scale:.4f}")

    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        yaml.safe_dump({"latent_scale": scale}, f)
    print(f"wrote {meta_path}")

    return scale


def load_latent_scale(meta_path, default=1.0):
    """Read a latent scale sidecar; used by inference."""
    if not os.path.exists(meta_path):
        return default
    with open(meta_path) as f:
        return float(yaml.safe_load(f)["latent_scale"])
