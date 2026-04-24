
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange
from typing import Dict, Any

from .decoder_modules import CVSSDecoderBlock, FinalUpsample_X4
from .backbone_modules import VSSBlock, PatchMerging2D, PatchEmbed2D, VMambaStage
from .fusion_modules import ConcatFusion, CrossMambaFusionBlock, SkipFusionBlock, DecoderFusionBlock


class TwoStageUNet(nn.Module):
    def __init__(self, n_classes: int, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        dims = config['dims']
        depths = config['depths']
        self.decoder_depths = config.get('decoder_depths', [1] * (len(dims) - 1))
        in_channels = config['in_channels']
        patch_size = config.get('patch_size', 4)
        drop_path_rate = config.get('drop_path_rate', 0.1)
        vssm_args = config.get('vssm_args', {})
        self.num_stages = len(dims)


        self.has_dem = 'dem' in in_channels
        self.has_s1 = 's1' in in_channels
        self.has_s2 = 's2' in in_channels

        if not (self.has_dem and self.has_s1 and self.has_s2):
            self.use_synergy_skip = False
            print(
                "⚠️ Warning: One or more modalities (DEM, S1, S2) are missing. `use_synergy_skip` is forced to False.")
        else:
            self.use_synergy_skip = config.get('use_synergy_skip', True)

        self.dem_injection_type = config.get('dem_injection_type', 'cross_mamba')
        print(f"🚀 Initializing TwoStageUNet (use_synergy_skip={self.use_synergy_skip}).")

        if self.has_dem: self.dem_embed = PatchEmbed2D(patch_size, in_channels['dem'], dims[0])
        if self.has_s1: self.s1_embed = PatchEmbed2D(patch_size, in_channels['s1'], dims[0])
        if self.has_s2: self.s2_embed = PatchEmbed2D(patch_size, in_channels['s2'], dims[0])

        self.dem_stages, self.s1_stages, self.s2_stages = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.s1_dem_injection_fusers, self.s2_dem_injection_fusers = nn.ModuleList(), nn.ModuleList()
        self.s1s2_cross_fusers = nn.ModuleList()
        self.skip_fusers = nn.ModuleList()
        self.synergy_fusers = nn.ModuleList()
        self.dem_downsamplers, self.s1_downsamplers, self.s2_downsamplers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(self.num_stages):
            stage_dim, stage_depth = dims[i], depths[i]
            stage_dpr = dpr[cur:cur + stage_depth];
            cur += stage_depth

            if self.has_dem: self.dem_stages.append(VMambaStage(stage_dim, stage_depth, stage_dpr, **vssm_args))
            if self.has_s1: self.s1_stages.append(VMambaStage(stage_dim, stage_depth, stage_dpr, **vssm_args))
            if self.has_s2: self.s2_stages.append(VMambaStage(stage_dim, stage_depth, stage_dpr, **vssm_args))

            cross_mamba_args = config.get('cross_mamba_args', {})
            if self.dem_injection_type == 'cross_mamba':
                if self.has_dem and self.has_s1: self.s1_dem_injection_fusers.append(
                    CrossMambaFusionBlock(dim=stage_dim, **cross_mamba_args))
                if self.has_dem and self.has_s2: self.s2_dem_injection_fusers.append(
                    CrossMambaFusionBlock(dim=stage_dim, **cross_mamba_args))
            else:
                if self.has_dem and self.has_s1: self.s1_dem_injection_fusers.append(ConcatFusion(dim=stage_dim))
                if self.has_dem and self.has_s2: self.s2_dem_injection_fusers.append(ConcatFusion(dim=stage_dim))

            if self.has_s1 and self.has_s2:
                self.s1s2_cross_fusers.append(CrossMambaFusionBlock(dim=stage_dim, **cross_mamba_args))
                if self.use_synergy_skip: self.synergy_fusers.append(ConcatFusion(dim=stage_dim))

            self.skip_fusers.append(
                SkipFusionBlock(dim=stage_dim, num_inputs=sum([self.has_dem, self.has_s1, self.has_s2])))

            if i < self.num_stages - 1:
                next_dim = dims[i + 1]
                if self.has_dem: self.dem_downsamplers.append(PatchMerging2D(stage_dim, next_dim))
                if self.has_s1: self.s1_downsamplers.append(PatchMerging2D(stage_dim, next_dim))
                if self.has_s2: self.s2_downsamplers.append(PatchMerging2D(stage_dim, next_dim))

        self.decoder = self._build_vmamba_decoder(dims, self.decoder_depths, vssm_args)
        self.norm = nn.LayerNorm(dims[0])
        self.up = FinalUpsample_X4(dim=dims[0])
        self.head = nn.Conv2d(dims[0], n_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)
        print("✅ TwoStageUNet initialized successfully.\n")

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        def get_input(x_dict, key1, key2=None):
            tensor = x_dict.get(key1)
            if tensor is None and key2: tensor = x_dict.get(key2)
            return tensor

        dem_feat = self.dem_embed(get_input(x_dict, 'dem')) if self.has_dem else None
        s1_feat = self.s1_embed(get_input(x_dict, 's1', 'sar')) if self.has_s1 else None
        s2_feat = self.s2_embed(get_input(x_dict, 's2', 'optical')) if self.has_s2 else None

        skips, synergy_skips = [], []

        for i in range(self.num_stages):
            dem_out = self.dem_stages[i](dem_feat) if self.has_dem else None

            s1_injected, dem_updated_by_s1 = s1_feat, None
            if self.has_s1 and self.has_dem and len(self.s1_dem_injection_fusers) > i:
                if self.dem_injection_type == 'cross_mamba':
                    s1_injected, dem_updated_by_s1 = self.s1_dem_injection_fusers[i](s1_feat, dem_out)
                else:
                    s1_injected = self.s1_dem_injection_fusers[i](s1_feat, dem_out)

            s2_injected, dem_updated_by_s2 = s2_feat, None
            if self.has_s2 and self.has_dem and len(self.s2_dem_injection_fusers) > i:
                if self.dem_injection_type == 'cross_mamba':
                    s2_injected, dem_updated_by_s2 = self.s2_dem_injection_fusers[i](s2_feat, dem_out)
                else:
                    s2_injected = self.s2_dem_injection_fusers[i](s2_feat, dem_out)

            synergy_feat = None
            if self.use_synergy_skip and dem_updated_by_s1 is not None and dem_updated_by_s2 is not None:
                synergy_feat = self.synergy_fusers[i](dem_updated_by_s1, dem_updated_by_s2)
            synergy_skips.append(synergy_feat)

            s1_processed = self.s1_stages[i](s1_injected) if self.has_s1 else None
            s2_processed = self.s2_stages[i](s2_injected) if self.has_s2 else None

            s1_final, s2_final = s1_processed, s2_processed
            if self.has_s1 and self.has_s2:
                s1_final, s2_final = self.s1s2_cross_fusers[i](s1_processed, s2_processed)

            active_skips = [s for s in [dem_out, s1_final, s2_final] if s is not None]
            fused_skip = self.skip_fusers[i](*active_skips)
            skips.append(fused_skip)

            if i < self.num_stages - 1:
                if self.has_dem: dem_feat = self.dem_downsamplers[i](dem_out)
                if self.has_s1: s1_feat = self.s1_downsamplers[i](s1_final)
                if self.has_s2: s2_feat = self.s2_downsamplers[i](s2_final)

        x = skips.pop();
        _ = synergy_skips.pop()
        for i in range(len(self.decoder)):
            skip_feat = skips.pop();
            synergy_skip_feat = synergy_skips.pop()
            x = self._patch_expand_forward(self.decoder[i]['expand'], x)

            if self.use_synergy_skip:
                if synergy_skip_feat is not None:
                    x = self.decoder[i]['skip_fusion'](torch.cat([x, skip_feat, synergy_skip_feat], dim=-1))
                else:
                    raise ValueError("Synergy skip was expected but is None.")
            else:
                x = self.decoder[i]['skip_fusion'](torch.cat([x, skip_feat], dim=-1))

            x = self.decoder[i]['block'](x)

        x = self.norm(x);
        x = self.up(x)
        x = self.head(x.permute(0, 3, 1, 2))

        ref_tensor = get_input(x_dict, 's2', 'optical')
        if ref_tensor is None: ref_tensor = get_input(x_dict, 's1', 'sar')
        if ref_tensor is None: ref_tensor = get_input(x_dict, 'dem')
        if ref_tensor is None: raise ValueError("No valid input tensor found for shape reference.")
        H_orig, W_orig = ref_tensor.shape[-2:]
        if x.shape[-2:] != (H_orig, W_orig):
            x = F.interpolate(x, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        return x

    def _build_vmamba_decoder(self, dims, decoder_depths, vssm_args):
        decoder = nn.ModuleList()
        num_stages = len(dims)
        for i in range(num_stages - 1):
            in_dim, out_dim, depth = dims[num_stages - 1 - i], dims[num_stages - 2 - i], decoder_depths[i]
            num_inputs = 3 if self.use_synergy_skip else 2
            fusion_in_dim = num_inputs * out_dim
            decoder.append(nn.ModuleDict({
                'expand': nn.Linear(in_dim, 4 * out_dim, bias=False),
                'skip_fusion': DecoderFusionBlock(in_dim=fusion_in_dim, out_dim=out_dim, vssm_args=vssm_args),
                'block': nn.Sequential(*[CVSSDecoderBlock(hidden_dim=out_dim, **vssm_args) for _ in range(depth)])
            }))
        return decoder

    def _patch_expand_forward(self, expand_layer, x):
        x = expand_layer(x);
        return rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0);
            nn.init.constant_(m.weight, 1.0)