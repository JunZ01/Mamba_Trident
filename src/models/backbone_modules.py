#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backbone_modules.py

"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath
import sys
from pathlib import Path


VMAMBA_PATH = Path(__file__).parent.parent / "VMamba"
sys.path.insert(0, str(VMAMBA_PATH))

from classification.models.vmamba import VSSBlock, PatchMerging2D

print("✅ Successfully imported VMamba components for backbone.")


class PatchEmbed2D(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1) # -> (B, H, W, C)
        return self.norm(x)

class VMambaStage(nn.Module):
    def __init__(self, dim, depth, dpr_list, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim, drop_path=dpr_list[i], **kwargs) for i in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x#!/usr/bin/env python
