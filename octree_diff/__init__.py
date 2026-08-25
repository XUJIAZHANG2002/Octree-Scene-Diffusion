"""Octree Diffusion for Semantic Scene Generation and Completion (ICRA 2026).

https://arxiv.org/abs/2509.16483

Top-level names are resolved lazily so that `import octree_diff` stays cheap and,
more importantly, does not drag in torch/ocnn (or Open3D, via `octree_diff.viz`)
for callers that only want the version string.
"""

__version__ = "0.1.0"

_LAZY = {
    "IndoorPipeline": ("octree_diff.inference.pipeline", "IndoorPipeline"),
    "simulate_sensor": ("octree_diff.inference.sensor_sim", "simulate_sensor"),
    "pick_viewpoint": ("octree_diff.inference.sensor_sim", "pick_viewpoint"),
    "observed_labels": ("octree_diff.inference.sensor_sim", "observed_labels"),
    "ddim_sample_dense": ("octree_diff.diffusion.sampling", "ddim_sample_dense"),
    "ddim_inpaint_dense": ("octree_diff.diffusion.sampling", "ddim_inpaint_dense"),
    "ddim_sample_graph": ("octree_diff.diffusion.sampling", "ddim_sample_graph"),
    "ddim_inpaint_graph": ("octree_diff.diffusion.sampling", "ddim_inpaint_graph"),
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
