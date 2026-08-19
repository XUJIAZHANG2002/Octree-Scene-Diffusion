"""Single entry point for training, inference and verification.

Replaces main.py, which shelled out to the training scripts with `--config` --
a flag none of them parsed. Everything here runs in-process.

    python -m octree_diff.cli train    --stage sem_vae --device cuda
    python -m octree_diff.cli generate --num-samples 8 --out out/gen
    python -m octree_diff.cli complete --input scene.pt --known-mask known.pt --best-of 3
    python -m octree_diff.cli extend   --input scene.pt --steps-out 4
    python -m octree_diff.cli verify

The four training stages must run in this order, because each diffusion stage
loads the VAE checkpoint named by its `vae_ref.checkpoint`:

    str_vae -> str_diff        sem_vae -> sem_diff
"""

import argparse
import os
import sys

STAGES = ["str_vae", "str_diff", "sem_vae", "sem_diff"]


def _add_common(p):
    p.add_argument("--config-dir", "--config_dir", dest="config_dir", default=None,
                   help="directory of *_config.yaml (default: the repo's configs/)")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")


def _add_sampling(p):
    p.add_argument("--out", default="out")
    p.add_argument("--steps", type=int, default=100, help="DDIM steps")
    p.add_argument("--eta", type=float, default=0.0, help="0 = deterministic DDIM")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repaint-resample", "--repaint_resample", dest="repaint_resample",
                   type=int, default=1, help="RePaint jumps per step")
    p.add_argument("--best-of", "--best_of", dest="best_of", type=int, default=1,
                   help="sample the layout N times, keep the one that best agrees with "
                        "the observed region; rejects the ~1-in-20 draw that degrades")


def build_parser():
    p = argparse.ArgumentParser(
        prog="octree_diff", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train one stage")
    t.add_argument("--stage", required=True, choices=STAGES)
    _add_common(t)

    g = sub.add_parser("generate", help="unconditional generation")
    g.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=1)
    g.add_argument("--stages", default="both", choices=["structure", "semantic", "both"])
    g.add_argument("--structure-from", dest="structure_from",
                   help="reuse a layout from this .pt instead of sampling one")
    _add_sampling(g); _add_common(g)

    c = sub.add_parser("complete", help="semantic scene completion (inpaint)")
    c.add_argument("--input", required=True, help="[128,128,64] int label tensor")
    c.add_argument("--known-mask", "--known_mask", dest="known_mask", required=True,
                   help="[128,128,64] bool tensor, True = observed")
    _add_sampling(c); _add_common(c)

    e = sub.add_parser("extend", help="scene extension along +x (outpaint)")
    e.add_argument("--input", required=True, help="[128,128,64] int label tensor")
    e.add_argument("--steps-out", "--steps_out", dest="steps_out", type=int, default=4)
    _add_sampling(e); _add_common(e)

    v = sub.add_parser("verify", help="run the indoor check suite")
    v.add_argument("--checks", nargs="+", default=None)
    v.add_argument("--ref-dir", "--ref_dir", dest="ref_dir", default=None)
    v.add_argument("--steps", type=int, default=50)
    v.add_argument("--repaint-resample", "--resample", dest="resample", type=int, default=1)
    v.add_argument("--n", type=int, default=10)
    v.add_argument("--n-gen", "--n_gen", dest="n_gen", type=int, default=8)
    v.add_argument("--n-ref", "--n_ref", dest="n_ref", type=int, default=100)
    v.add_argument("--n-out", "--n_out", dest="n_out", type=int, default=4)
    _add_common(v)
    return p


def cmd_train(args):
    import importlib
    module = {
        "str_vae": "train_structure_vae",
        "str_diff": "train_structure_diffusion",
        "sem_vae": "train_sem_vae",
        "sem_diff": "train_sem_diffusion",
    }[args.stage]
    mod = importlib.import_module(f"octree_diff.training.{module}")
    mod.train(config_dir=args.config_dir, device=args.device)


def cmd_sample(args):
    import torch
    import yaml
    from octree_diff.inference import pipeline as P

    mode = {"generate": "generate", "complete": "inpaint", "extend": "outpaint"}[args.command]
    # the run_* helpers read these off a namespace; fill in the ones this
    # subcommand does not expose so they see a uniform object
    for name, default in (("stages", "both"), ("structure_from", None), ("num_samples", 1),
                          ("input", None), ("known_mask", None), ("steps_out", 4)):
        if not hasattr(args, name):
            setattr(args, name, default)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("loading models...")
    pipe = P.IndoorPipeline(device, config_dir=args.config_dir)
    print(f"  latent scales: structure {pipe.str_scale:.4f}, semantic {pipe.sem_scale:.4f}")

    print(f"running {mode}...")
    {"generate": P.run_generate, "inpaint": P.run_inpaint, "outpaint": P.run_outpaint}[mode](
        pipe, args, args.out)

    with open(os.path.join(args.out, "manifest.yaml"), "w") as f:
        yaml.safe_dump({
            "mode": mode, "stages": args.stages, "steps": args.steps, "eta": args.eta,
            "repaint_resample": args.repaint_resample, "seed": args.seed,
            "num_samples": args.num_samples, "input": args.input,
            "known_mask": args.known_mask,
            "latent_scale": {"structure": pipe.str_scale, "semantic": pipe.sem_scale},
            "checkpoints": pipe.paths,
        }, f, sort_keys=False)
    print(f"wrote {args.out}/")


def cmd_verify(args):
    import torch
    from octree_diff.inference import verify_indoor as V

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    pipe = V.IndoorPipeline(args.device, config_dir=args.config_dir)
    print(f"latent scales: structure {pipe.str_scale:.4f}, semantic {pipe.sem_scale:.4f}")

    for name in (args.checks or list(V.CHECKS)):
        fn, needs = V.CHECKS[name]
        missing = [n for n in needs if not os.path.exists(pipe.paths[n])]
        if missing:
            print(f"\n[{name}] skipped — not trained yet: {', '.join(missing)}")
            continue
        fn(pipe, args)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "verify":
        return cmd_verify(args)
    return cmd_sample(args)


if __name__ == "__main__":
    sys.exit(main())
