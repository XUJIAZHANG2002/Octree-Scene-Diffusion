import os
import random
import torch
from torch.utils.data import Dataset, DataLoader

class VoxelPatchDataset(Dataset):
    def __init__(self, patch_root, transform=True):
        """
        Args:
            patch_root (str): folder containing .pt patch files
            transform (bool): if True, apply random flips/rotations
        """
        self.files = []
        for root, _, fnames in os.walk(patch_root):
            for f in fnames:
                if f.endswith(".pt"):
                    self.files.append(os.path.join(root, f))
        if not self.files:
            raise RuntimeError(f"No .pt patches found under {patch_root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        blob = torch.load(self.files[idx], map_location="cpu")
        labels = blob["labels"].clone().long()  # (D=64, H=128, W=128)
        meta = blob.get("meta", {})
        
        if self.transform:
            labels = self._random_transform(labels)
            
        # permute to (H, W, D) = (128, 128, 64)
        labels = labels.permute(1, 2, 0).contiguous()
        return labels, meta

    def _random_transform(self, labels):
        # random flips
        if random.random() < 0.5:
            labels = torch.flip(labels, dims=[1]) # flip y
        if random.random() < 0.5:
            labels = torch.flip(labels, dims=[2]) # flip x
        # random rotation (k * 90 deg around z axis)
        k = random.randint(0, 3)
        labels = torch.rot90(labels, k, dims=[1, 2])
        return labels

def get_dataloader(patch_root, batch_size=1, shuffle=True):
    dataset = VoxelPatchDataset(patch_root, transform=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


