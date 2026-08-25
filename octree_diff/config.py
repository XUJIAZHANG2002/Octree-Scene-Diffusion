"""Config loading: alias normalisation, path anchoring, and consistency checks.

Three problems this solves, all of which bit during the indoor retrain:

1. The four config files spell the same concept differently -- `checkpoint_path`
   vs `save_path`, `epochs` vs `t_max`, `in_ch` vs `in_channels` vs `channel_in`.
   Aliases are normalised on load, so old configs keep working and code reads one
   canonical name.

2. Relative paths resolved against the process cwd, so everything only worked when
   run from the repo root. They are now anchored to the repo root itself.

3. A diffusion config's `vae_ref.checkpoint` silently duplicates the VAE config's
   own output path, and its UNet width silently depends on the VAE's latent size.
   Editing one without the other produced a checkpoint that loaded but was wrong.
   `check_consistency` turns that into a startup error.
"""

import os
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")

# section -> {alias: canonical}
ALIASES = {
    "training": {"save_path": "checkpoint_path", "t_max": "epochs"},
    "model": {"in_ch": "in_channels", "channel_in": "in_channels"},
    "unet": {"in_ch": "in_channels", "channel_in": "in_channels"},
}

# keys holding a filesystem path, anchored to REPO_ROOT when relative
PATH_KEYS = ("checkpoint_path", "data_dir", "class_weights_file", "checkpoint")

# stage -> (own config, VAE config it depends on or None)
STAGES = {
    "str_vae": ("structure_vae_config.yaml", None),
    "str_diff": ("structure_diffusion_config.yaml", "structure_vae_config.yaml"),
    "sem_vae": ("sem_vae_config.yaml", None),
    "sem_diff": ("sem_diffusion_config.yaml", "sem_vae_config.yaml"),
}


def _normalise(cfg, where):
    for section, amap in ALIASES.items():
        block = cfg.get(section)
        if not isinstance(block, dict):
            continue
        for alias, canon in amap.items():
            if alias not in block:
                continue
            if canon in block and block[canon] != block[alias]:
                raise ValueError(
                    f"{where}: {section}.{alias} and {section}.{canon} are aliases "
                    f"but disagree ({block[alias]!r} vs {block[canon]!r}) -- keep one")
            block[canon] = block.pop(alias)
    return cfg


def _anchor_paths(cfg):
    for block in cfg.values():
        if not isinstance(block, dict):
            continue
        for key, val in block.items():
            if key in PATH_KEYS and isinstance(val, str) and not os.path.isabs(val):
                block[key] = os.path.join(REPO_ROOT, val)
    return cfg


def load_config(path, anchor=True):
    """Load one YAML config, normalising aliases and anchoring relative paths."""
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(REPO_ROOT, path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _normalise(cfg, os.path.basename(path))
    return _anchor_paths(cfg) if anchor else cfg


def check_consistency(stage, cfg, vae_cfg):
    """Fail loudly on the cross-config couplings that are otherwise silent."""
    problems = []

    ref = cfg.get("vae_ref", {}).get("checkpoint")
    produced = vae_cfg["training"]["checkpoint_path"]
    if ref and os.path.normpath(ref) != os.path.normpath(produced):
        problems.append(
            f"vae_ref.checkpoint ({ref}) is not the checkpoint its VAE config "
            f"writes ({produced}) -- the diffusion would train against a different "
            f"VAE than the one you configured")

    if stage == "str_diff":
        unet_in = cfg["unet"]["in_channels"]
        z = vae_cfg["model"]["z_channels"]
        if unet_in != z:
            problems.append(
                f"unet.in_channels ({unet_in}) != structure VAE z_channels ({z}) -- "
                f"the UNet cannot consume the latents the VAE produces")

    if ref and not os.path.exists(ref):
        problems.append(
            f"vae_ref.checkpoint does not exist: {ref} -- train that stage first")

    if problems:
        raise SystemExit("config inconsistency:\n  - " + "\n  - ".join(problems))


def load_stage(stage, config_dir=None, check=True):
    """Load a training stage's config, plus the VAE config it depends on.

    Returns (cfg, vae_cfg); vae_cfg is None for the two VAE stages.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(STAGES)}")
    d = config_dir or CONFIG_DIR
    own, vae_name = STAGES[stage]
    cfg = load_config(os.path.join(d, own))
    vae_cfg = load_config(os.path.join(d, vae_name)) if vae_name else None
    if check and vae_cfg is not None:
        check_consistency(stage, cfg, vae_cfg)
    return cfg, vae_cfg
