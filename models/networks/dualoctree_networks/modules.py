# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import math
import torch
import torch.nn
import torch.nn.init
import torch.nn.functional as F
import torch.utils.checkpoint
# import torch_geometric.nn

from .utils.scatter import scatter_mean
from ocnn.octree import key2xyz, xyz2key

from ocnn.octree import Octree
from ocnn.utils import scatter_add
from models.networks.modules import (
    nonlinearity,
    ckpt_conv_wrapper,
    DualOctreeGroupNorm,
    Conv1x1,
    Conv1x1Gn,
    Conv1x1GnGelu,
    Conv1x1GnGeluSequential,
    Downsample,
    Upsample,
    GraphConv,
    GraphResBlock,
    GraphResBlocks,
    GraphDownsample,
    GraphUpsample,
)


class GraphDownsample(torch.nn.Module):

    def __init__(self, channels_in, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out or channels_in
        self.downsample = Downsample(channels_in)
        if self.channels_in != self.channels_out:
            self.conv1x1 = Conv1x1GnGelu(self.channels_in, self.channels_out)

    def forward(self, x, octree, d, leaf_mask, numd, lnumd):
        # downsample nodes at layer depth
        outd = x[-numd:]
        outd = self.downsample(outd)

        # get the nodes at layer (depth-1)
        out = torch.zeros(leaf_mask.shape[0], x.shape[1], device=x.device)
        out[leaf_mask] = x[-lnumd-numd:-numd]
        out[leaf_mask.logical_not()] = outd

        # construct the final output
        out = torch.cat([x[:-numd-lnumd], out], dim=0)

        if self.channels_in != self.channels_out:
            out = self.conv1x1(out, octree, d)
        return out

    def extra_repr(self):
        return 'channels_in={}, channels_out={}'.format(
            self.channels_in, self.channels_out)


class GraphUpsample(torch.nn.Module):

    def __init__(self, channels_in, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out or channels_in
        self.upsample = Upsample(channels_in)
        if self.channels_in != self.channels_out:
            self.conv1x1 = Conv1x1GnGelu(self.channels_in, self.channels_out)

    def forward(self, x, octree, d, leaf_mask, numd):
        # upsample nodes at layer (depth-1)
        outd = x[-numd:]
        out1 = outd[leaf_mask.logical_not()]
        out1 = self.upsample(out1)

        # construct the final output
        out = torch.cat([x[:-numd], outd[leaf_mask], out1], dim=0)
        if self.channels_in != self.channels_out:
            out = self.conv1x1(out, octree, d)
        return out

    def extra_repr(self):
        return 'channels_in={}, channels_out={}'.format(
            self.channels_in, self.channels_out)
import torch
import torch.nn as nn
import torch.nn.functional as F

# class SharedPatchEmbed(nn.Module):
#     def __init__(self, num_classes, embed_dim, hidden_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(num_classes, embed_dim)

#         self.conv_proj = nn.Sequential(
#             nn.Conv2d(embed_dim, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 4, 4]
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),        # [B, 128, 2, 2]
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # [B, 256, 1, 1]
#             nn.ReLU(),
#         )

#         self.to_latent = nn.Sequential(
#             nn.Flatten(),                   # [B, 256]
#             nn.Linear(256, hidden_dim),     # → final node embedding
#         )

#     def forward(self, x):
#         """
#         x: [B, 1, 4, 4], each element ∈ [0, num_classes)
#         """
#         B = x.shape[0]
#         x = self.embedding(x.long())               # [B, 1, 4, 4, embed_dim]
#         x = x.squeeze(1)                           # [B, 4, 4, embed_dim]
#         x = x.permute(0, 3, 1, 2)                  # [B, embed_dim, 4, 4]
#         x = self.conv_proj(x)                      # [B, 256, 1, 1]
#         x = self.to_latent(x)                      # [B, hidden_dim]
#         return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SharedPatchDecoder(nn.Module):
#     def __init__(self, hidden_dim, embed_dim, num_classes):
#         super().__init__()

#         self.proj = nn.Sequential(
#             nn.Linear(hidden_dim, 256),     # [B, 256]
#             nn.ReLU(),
#         )

#         self.upconv = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),     # → [B, 128, 2, 2]
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, embed_dim, kernel_size=2, stride=2),  # → [B, embed_dim, 4, 4]
#             nn.ReLU(),
#             nn.Conv2d(embed_dim, num_classes, kernel_size=1)           # → [B, num_classes, 4, 4]
#         )

#     def forward(self, x):
#         """
#         x: [B, hidden_dim] → from latent
#         return: [B, num_classes, 4, 4]
#         """
#         x = self.proj(x)                  # [B, 256]
#         x = x.view(-1, 256, 1, 1)         # [B, 256, 1, 1]
#         x = self.upconv(x)               # [B, num_classes, 4, 4]
#         return x
class SharedPatchEmbed(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

        self.conv_proj = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
        )

    def forward(self, x):
        """
        x: [N, 4, 4] with values ∈ [0, num_classes)
        """
        x = self.embedding(x.long())         # [N, 4, 4, embed_dim]
        x = x.permute(0, 3, 1, 2)            # [N, embed_dim, 4, 4]
        x = self.conv_proj(x)                # [N, 256, 1, 1]
        x = self.to_latent(x)                # [N, hidden_dim]
        return x

class SharedPatchDecoder(nn.Module):
    def __init__(self, hidden_dim, embed_dim, num_classes):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),     # [B, 256]
            nn.ReLU(),
        )

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),     # [B, 128, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(128, embed_dim, kernel_size=2, stride=2),  # [B, embed_dim, 4, 4]
            nn.ReLU(),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)           # [B, num_classes, 4, 4]
        )

    def forward(self, x):
        """
        x: [B, hidden_dim] → from latent
        return: [B, num_classes, 4, 4]
        """
        x = self.proj(x)                  # [B, 256]
        x = x.view(-1, 256, 1, 1)         # [B, 256, 1, 1]
        x = self.upconv(x)               # [B, num_classes, 4, 4]
        return x
