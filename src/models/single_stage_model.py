
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange
from typing import Dict, Any, List
from copy import deepcopy
from .backbone_modules import VSSBlock, PatchMerging2D, PatchEmbed2D, VMambaStage
from .decoder_modules import FinalUpsample_X4, CVSSDecoderBlock


class SingleStageUNet(nn.Module):

    def __init__(self, n_classes: int, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        dims = config['dims']
        depths = config['depths']
        patch_size = config.get('patch_size', 4)
        in_channels_dict = config['in_channels']
        drop_path_rate = config.get('drop_path_rate', 0.1)
        vssm_args = config.get('vssm_args', {})
        self.num_stages = len(dims)


        total_in_channels = sum(in_channels_dict.values())
        print(f"SingleStageUNet: Total input channels after concatenation = {total_in_channels}")

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=total_in_channels,
            embed_dim=dims[0]
        )

        self.encoder_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage_dim = dims[i]
            stage_depth = depths[i]
            stage_dpr = dpr[cur:cur + stage_depth]
            cur += stage_depth

            self.encoder_stages.append(
                VMambaStage(stage_dim, stage_depth, stage_dpr, **vssm_args)
            )
            if i < self.num_stages - 1:
                self.downsamplers.append(PatchMerging2D(dims[i], dims[i + 1]))

        self.decoder_stages = nn.ModuleList()
        self.skip_fusers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        decoder_depths = config.get('decoder_depths', [1] * (self.num_stages - 1))

        for i in range(self.num_stages - 1, 0, -1):
            # i = 3, 2, 1
            in_dim = dims[i]
            out_dim = dims[i - 1]

            self.upsamplers.append(nn.Linear(in_dim, 4 * out_dim, bias=False))

            self.skip_fusers.append(nn.Linear(2 * out_dim, out_dim))

            decoder_depth = decoder_depths[self.num_stages - 1 - i]
            self.decoder_stages.append(
                nn.Sequential(*[CVSSDecoderBlock(hidden_dim=out_dim, **vssm_args) for _ in range(decoder_depth)])
            )

        self.norm = nn.LayerNorm(dims[0])
        self.up = FinalUpsample_X4(dim=dims[0])
        self.head = nn.Conv2d(dims[0], n_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)
        print("✅ SingleStageUNet (Early-Fusion) initialized successfully.\n")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0);
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features):
        if isinstance(features, dict):
            s2 = features.get('s2') if features.get('s2') is not None else features.get('optical')
            s1 = features.get('s1') if features.get('s1') is not None else features.get('sar')
            dem = features.get('dem')

            if s2 is None or s1 is None or dem is None:
                raise ValueError(
                    f"SingleStageUNet is missing input modalities. "
                    f"S2: {s2 is not None}, S1: {s1 is not None}, DEM: {dem is not None}"
                )
            concatenated_input = torch.cat([s2, s1, dem], dim=1)

        elif isinstance(features, torch.Tensor):
            concatenated_input = features

        else:
            raise TypeError(
                f"SingleStageUNet expects a Dict or Tensor, got: {type(features)}"
            )


        x = self.patch_embed(concatenated_input)
        skips: List[torch.Tensor] = []
        for i in range(self.num_stages):
            x = self.encoder_stages[i](x)
            if i < self.num_stages - 1:
                skips.append(x)
                x = self.downsamplers[i](x)

        for i in range(self.num_stages - 1):
            
            x = self.upsamplers[i](x)
            x = rearrange(x, 'b h w (p1 p2 c_out) -> b (h p1) (w p2) c_out', p1=2, p2=2)

            skip_feat = skips.pop()
            x = torch.cat([x, skip_feat], dim=-1)
            x = self.skip_fusers[i](x)

            x = self.decoder_stages[i](x)

        x = self.norm(x)
        x = self.up(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.head(x)

        H_orig, W_orig = concatenated_input.shape[-2:]
        if x.shape[-2:] != (H_orig, W_orig):
            x = F.interpolate(x, size=(H_orig, W_orig), mode='bilinear', align_corners=False)

        return x