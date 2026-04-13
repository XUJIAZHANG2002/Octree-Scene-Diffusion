import os
import torch
from torch.utils.data import Dataset, DataLoader

class SplitDataset(Dataset):
    """
    Dataset for structural occupancy grids.
    Expects files to be saved as .pt tensors of shape [1, D, H, W] or [1, 1, D, H, W].
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
            # Adding weights_only=True stops the warning at the source
            split_tensor = torch.load(self.files[idx], map_location="cpu", weights_only=True)
            
            if split_tensor.dim() == 5: 
                split_tensor = split_tensor.squeeze(0)
            p = torch.randint(0, 4, (1,)).item()
            if p == 1:
                split_tensor = torch.flip(split_tensor, dims=[1])   # flip H
            elif p == 2:
                split_tensor = torch.flip(split_tensor, dims=[2])   # flip W
            elif p == 3:
                split_tensor = torch.flip(split_tensor, dims=[1, 2])  # flip both
            return split_tensor.float(), 0

def get_split_dataloader(data_dir, batch_size=1, shuffle=True):
    dataset = SplitDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)