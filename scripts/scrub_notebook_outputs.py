"""Replace machine-specific absolute paths in notebook *outputs* with placeholders.

Executing a notebook bakes the author's filesystem into stdout, warnings and
tracebacks. The cell sources are path-clean (see fix_notebook_imports.py), but the
stored outputs are not, and they ship in the repo.

Only `outputs` are touched -- never `source` -- so figures and printed results are
preserved. Re-run after any re-execution:

    python scripts/scrub_notebook_outputs.py --check
    python scripts/scrub_notebook_outputs.py
"""

import argparse
import glob
import json
import os
import re

HOME = os.path.expanduser("~")

# ordered: longest / most specific first
RULES = [
    (re.escape(HOME) + r"/Octree-Scene-Diffusion", "<repo>"),
    (re.escape(HOME) + r"/miniconda3/envs/([A-Za-z0-9_.-]+)", r"<env:\1>"),
    (re.escape(HOME) + r"/miniconda3", "<conda>"),
    (re.escape(HOME) + r"/SemCityOcto", "<kitti-weights>"),
    (re.escape(HOME) + r"/Replica-Dataset", "<replica>"),
    (r"/home/[A-Za-z0-9_.-]+", "<home>"),
]


def scrub(text):
    n = 0
    for pat, rep in RULES:
        text, k = re.subn(pat, rep, text)
        n += k
    return text, n


def walk(obj):
    """Scrub every string in a nested output structure, counting replacements."""
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, list):
        total = 0
        out = []
        for v in obj:
            v, k = walk(v)
            out.append(v); total += k
        return out, total
    if isinstance(obj, dict):
        total = 0
        out = {}
        for key, v in obj.items():
            # image payloads are base64 -- never touch them
            if key.startswith("image/") or key == "application/pdf":
                out[key] = v; continue
            v, k = walk(v)
            out[key] = v; total += k
        return out, total
    return obj, 0


def process(path, check):
    nb = json.load(open(path))
    total = 0
    for cell in nb["cells"]:
        if "outputs" not in cell:
            continue
        cell["outputs"], n = walk(cell["outputs"])
        total += n
    if total and not check:
        with open(path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="report without writing")
    p.add_argument("--glob", default="notebooks/**/*.ipynb")
    a = p.parse_args()
    grand = 0
    for path in sorted(glob.glob(a.glob, recursive=True)):
        n = process(path, a.check)
        grand += n
        if n:
            print(f"{'would scrub' if a.check else 'scrubbed':12s} {n:4d}  {path}")
    print(f"\n{grand} path references {'found' if a.check else 'scrubbed'}")


if __name__ == "__main__":
    main()
