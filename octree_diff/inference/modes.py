"""The five execution models, named once so the two exporters agree.

A mode is fixed by three things: what is observed, and which of the two
diffusion stages is conditioned on it. Keeping them in one table exists to make
one distinction unmissable:

    completing *geometry* is not completing *semantics*

An observation can pin the structure stage while the semantic stage still
samples from the prior. The result looks like inpainting in a figure -- the
layout follows the sensor -- but the labels are invented, and because label 0
means free, the semantic decode also deletes observed voxels the structure
stage had correctly kept. That was the outdoor `complete` path: measured on
SemanticKITTI 06:000930, the structure stage kept 100% of the scan's 15,652
hits and the unconditioned semantic decode then dropped 35% of them and
relabelled the survivors to 0.584 agreement.

`complete-geometry` and `complete-semantic` are therefore separate modes rather
than one `complete`, and a domain that cannot do the second says so instead of
quietly running the first.
"""

from collections import namedtuple

GENERATE = "generate"
SEMANTICS_FROM_STRUCTURE = "semantics-from-structure"
COMPLETE_GEOMETRY = "complete-geometry"
COMPLETE_SEMANTIC = "complete-semantic"
EXTEND = "extend"

Mode = namedtuple("Mode", "number observes structure semantic summary")

# structure/semantic: "sampled" = from the prior; "pinned" = conditioned by
# masked diffusion on a known latent; "guided" = conditioned by a gradient step
# in decoded label space. Both domains guide their semantic stage: a semantic
# latent is not a function of its own patch alone, so a pinned one encoded
# from partial labels is wrong even where the patch was fully observed.
# Indoors, on the four sample scenes, pinning kept 0.59-0.82 of the observed
# labels and guidance 0.89-1.00.
MODES = {
    GENERATE: Mode(
        1, "nothing", "sampled", "sampled",
        "full generation: both stages from noise"),
    SEMANTICS_FROM_STRUCTURE: Mode(
        2, "a complete layout, geometry only", "pinned", "sampled",
        "label a given layout: structure held fixed, semantics sampled"),
    COMPLETE_GEOMETRY: Mode(
        3, "a sensor reading, geometry only", "pinned", "sampled",
        "geometric completion: the observation pins the layout, labels are invented"),
    COMPLETE_SEMANTIC: Mode(
        4, "a labelled sensor reading", "pinned", "guided",
        "semantic scene completion: both stages follow the observation"),
    EXTEND: Mode(
        5, "a complete scene, as a half-volume", "pinned", "guided",
        "extension: outpaint beyond the given area"),
}

ORDER = [GENERATE, SEMANTICS_FROM_STRUCTURE, COMPLETE_GEOMETRY,
         COMPLETE_SEMANTIC, EXTEND]

# Where a domain's stages differ from the table above. None do at present;
# `--semantic-stage pinned` in pipeline.py keeps the indoor latent-pinning path
# for comparison.
_STAGE_OVERRIDE = {}


def stages(domain, name):
    """(structure, semantic) for this domain, honouring the overrides."""
    m = MODES[name]
    return _STAGE_OVERRIDE.get((domain, name), (m.structure, m.semantic))

# Modes each domain can actually run, and why not when it cannot. Absent from a
# domain's dict means supported.
_UNSUPPORTED = {
    "indoor": {},
    "outdoor": {},
}

# Modes whose semantic stage is weaker than the name suggests, per domain. The
# exporters print these so a figure is never captioned better than it is.
CAVEATS = {
    ("outdoor", COMPLETE_SEMANTIC):
        "the semantic stage is guided, not pinned. Latent pinning needs a "
        "node's latent to be a function of its own patch and the outdoor "
        "encoder is a graph network, so it is not: 11.7% of nodes have all 16 "
        "voxels of their patch observed, and even there the latent encoded from "
        "masked labels sits a median 61% away from the one encoded from full "
        "labels. Guidance sidesteps that by measuring the "
        "disagreement in decoded label space instead. Strength is "
        "--guidance-weight; 0 makes this mode 3. See "
        "docs/outdoor-semantic-conditioning.md",
    ("outdoor", EXTEND):
        "the kept half's semantics are held by the same guidance as mode 4, so "
        "--guidance-weight 0 reverts to relabelling roughly 20% of it",
}

# Historical spellings. `complete` meant different things in the two exporters,
# which is how the outdoor figures came to be captioned as semantic completion.
_ALIASES = {
    "full": GENERATE,
    "semantics": SEMANTICS_FROM_STRUCTURE,
    "semantics-only": SEMANTICS_FROM_STRUCTURE,
    "complete-structure": COMPLETE_GEOMETRY,
    "complete-semantics": COMPLETE_SEMANTIC,
    "outpaint": EXTEND,
    "inpaint": COMPLETE_SEMANTIC,
}

_LEGACY_COMPLETE = {"indoor": COMPLETE_SEMANTIC, "outdoor": COMPLETE_GEOMETRY}


def choices():
    """Accepted `--mode` values, for argparse help."""
    return ORDER + [str(MODES[m].number) for m in ORDER]


def describe(domain):
    """The table, as text, for `--mode list` and error messages."""
    out = []
    for name in ORDER:
        m = MODES[name]
        bad = _UNSUPPORTED.get(domain, {}).get(name)
        flag = "unsupported" if bad else "ok"
        out.append(f"  {m.number}. {name:<26} [{flag}]  {m.summary}")
        st, sem = stages(domain, name)
        out.append(f"     observes {m.observes}; structure {st}, semantic {sem}")
        if (domain, name) in CAVEATS:
            out.append(f"     caveat: {CAVEATS[(domain, name)]}")
        if bad:
            out.append(f"     unsupported: {bad}")
    return "\n".join(out)


def resolve(value, domain):
    """Canonical mode name for a CLI value, or raise with the domain's table."""
    if domain not in _UNSUPPORTED:
        raise ValueError(f"unknown domain {domain!r}")

    key = str(value).strip().lower()
    by_number = {str(MODES[m].number): m for m in ORDER}

    if key == "complete":
        name = _LEGACY_COMPLETE[domain]
        print(f"note: --mode complete is ambiguous and now means "
              f"--mode {name} for the {domain} domain; "
              f"pass it explicitly to silence this")
    else:
        name = by_number.get(key) or _ALIASES.get(key) or key

    if name not in MODES:
        raise ValueError(f"unknown --mode {value!r}. Modes for {domain}:\n"
                         + describe(domain))

    bad = _UNSUPPORTED[domain].get(name)
    if bad:
        raise ValueError(f"--mode {name} is not supported for the {domain} "
                         f"domain: {bad}")
    return name


def announce(domain, name):
    """One line naming what is pinned and what is sampled, plus any caveat."""
    m = MODES[name]
    st, sem = stages(domain, name)
    print(f"mode {m.number} ({name}): {m.summary}")
    print(f"  observes {m.observes}; structure {st}, semantic {sem}")
    caveat = CAVEATS.get((domain, name))
    if caveat:
        print(f"  caveat: {caveat}")
