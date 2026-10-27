"""SemanticKITTI scene-completion volumes, decoded on the fly.

The voxel ground truth is `sequences/<seq>/voxels/<frame>.label`: 256x256x32
uint16 raw SemanticKITTI ids, packed as one flat array. Decoding one is ~11 ms,
against training steps of ~1 s, so there is no reason to materialise a derived
dataset -- doing so would cost ~46 GB for no speedup.

Two details that matter:

`.invalid` marks voxels outside the sensor's traversal, where the ground truth
is not defined. The SSC convention is to exclude them from the loss rather than
treat them as free space; they are mapped to `ignore_index` (-1), which
`compute_semantic_loss` already drops.

Frames are near-duplicates -- consecutive ones differ in about 4% of voxels --
so `stride` subsamples. The SemanticKITTI SSC benchmark uses every 5th frame.

Sequence split follows the benchmark: 08 is validation, the rest train.
"""

import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

GRID = (256, 256, 32)
TRAIN_SEQUENCES = ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10")
VAL_SEQUENCES = ("08",)


def _unpack_bits(compressed):
    """`.invalid` and `.occluded` are bit-packed, one bit per voxel."""
    out = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = (compressed >> (7 - i)) & 1
    return out


class SSCVoxelDataset(Dataset):
    def __init__(self, root, sequences=TRAIN_SEQUENCES, stride=5, yaml_path=None,
                 ignore_invalid=True, ignore_index=-1):
        self.root = root
        self.ignore_invalid = ignore_invalid
        self.ignore_index = ignore_index

        yaml_path = yaml_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "semantic-kitti.yaml")
        lm = yaml.safe_load(open(yaml_path))["learning_map"]
        self.remap = np.zeros(max(lm) + 1, dtype=np.int64)
        for raw, learn in lm.items():
            self.remap[raw] = learn

        self.files = []
        for seq in sequences:
            d = os.path.join(root, "sequences", seq, "voxels")
            if not os.path.isdir(d):
                continue
            fs = sorted(f for f in os.listdir(d) if f.endswith(".label"))
            self.files += [os.path.join(d, f) for f in fs[::stride]]
        if not self.files:
            raise RuntimeError(f"no voxel labels under {root} for {sequences}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        labels = np.fromfile(path, dtype=np.uint16)
        labels = self.remap[labels].reshape(GRID)

        if self.ignore_invalid:
            inv = _unpack_bits(np.fromfile(path.replace(".label", ".invalid"),
                                           dtype=np.uint8))
            labels[inv.reshape(GRID) == 1] = self.ignore_index

        return torch.from_numpy(labels), os.path.basename(path)


def get_ssc_dataloader(root, batch_size=1, stride=5, sequences=TRAIN_SEQUENCES,
                       shuffle=True, num_workers=4, **kw):
    ds = SSCVoxelDataset(root, sequences=sequences, stride=stride, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)
