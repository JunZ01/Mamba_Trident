#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fusion_modules.py

"""

import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath
from enum import Enum
from .backbone_modules import VSSBlock



class FusionStrategy(Enum):

    CONCAT = "concat"
    CROSS_ATTENTION = "cross_attention"
    CROSS_MAMBA = "cross_mamba"


class ConcatFusion(nn.Module):

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.projection = nn.Linear(dim * 2, dim)

    def forward(self, x1, x2):
        return self.projection(torch.cat([x1, x2], dim=-1))



class CrossAttentionFusion(nn.Module):

    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):  # x1 is query, x2 is context (key, value)
        B, H, W, C = x1.shape
        x1_flat, x2_flat = x1.view(B, H * W, C), x2.view(B, H * W, C)

        q = self.q_proj(x1_flat).reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(x2_flat).reshape(B, H * W, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        fused = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        return x1 + self.out_proj(fused).view(B, H, W, C)


class Cross_Mamba_Attention_SSM(nn.Module):

    def __init__(self, d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto", **kwargs):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model, self.d_state, self.ssm_ratio = d_model, d_state, ssm_ratio
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj_1 = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, **factory_kwargs)
        self.A_log_1 = self.A_log_init(d_state, self.d_inner)
        self.A_log_2 = self.A_log_init(d_state, self.d_inner)
        self.D_1 = nn.Parameter(torch.ones(self.d_inner))
        self.D_2 = nn.Parameter(torch.ones(self.d_inner))
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor)
        with torch.no_grad():
            dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner).contiguous()
        return nn.Parameter(torch.log(A))

    def forward(self, x1, x2):
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except ImportError:
            raise ImportError("Mamba-ssm is not installed.")

        B, L, _ = x1.shape
        x1_p, x2_p = x1.permute(0, 2, 1), x2.permute(0, 2, 1)

        x_dbl_1 = self.x_proj_1(rearrange(x1_p, "b d l -> (b l) d"))
        x_dbl_2 = self.x_proj_2(rearrange(x2_p, "b d l -> (b l) d"))

        dt_1, B_1, C_1 = torch.split(x_dbl_1, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_2, B_2, C_2 = torch.split(x_dbl_2, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt_1 = rearrange(self.dt_proj_1.weight @ dt_1.t(), "d (b l) -> b d l", l=L)
        dt_2 = rearrange(self.dt_proj_2.weight @ dt_2.t(), "d (b l) -> b d l", l=L)

        A_1, A_2 = -torch.exp(self.A_log_1.float()), -torch.exp(self.A_log_2.float())

        B_1, B_2 = [rearrange(B_x, "(b l) s -> b s l", l=L).contiguous() for B_x in [B_1, B_2]]
        C_1, C_2 = [rearrange(C_x, "(b l) s -> b s l", l=L).contiguous() for C_x in [C_1, C_2]]

        y1 = selective_scan_fn(x1_p, dt_1, A_1, B_1, C_2, self.D_1.float(), delta_bias=self.dt_proj_1.bias.float(),
                               delta_softplus=True)
        y2 = selective_scan_fn(x2_p, dt_2, A_2, B_2, C_1, self.D_2.float(), delta_bias=self.dt_proj_2.bias.float(),
                               delta_softplus=True)

        return self.out_norm_1(rearrange(y1, "b d l -> b l d")), self.out_norm_2(rearrange(y2, "b d l -> b l d"))


class CrossMambaFusionBlock(nn.Module):

    def __init__(self, dim, ssm_ratio=2.0, d_state=16, d_conv=3, conv_bias=True, drop_path=0., **kwargs):
        super().__init__()
        self.d_inner = int(dim * ssm_ratio)


        self.in_proj1 = nn.Linear(dim, self.d_inner)
        self.in_proj2 = nn.Linear(dim, self.d_inner)

        self.conv2d_1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()
        # ----------------------------------------------

        self.cross_ssm = Cross_Mamba_Attention_SSM(d_model=self.d_inner, d_state=d_state, ssm_ratio=1.0,
                                                   **kwargs)

        self.out_proj1 = nn.Linear(self.d_inner, dim)
        self.out_proj2 = nn.Linear(self.d_inner, dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1, x2):
        B, H, W, C = x1.shape
        x1_res, x2_res = x1, x2


        x1_proj = self.in_proj1(x1)
        x2_proj = self.in_proj2(x2)

        # (B, H, W, C) -> (B, C, H, W) for Conv2d
        x1_conv = self.conv2d_1(x1_proj.permute(0, 3, 1, 2))
        x2_conv = self.conv2d_2(x2_proj.permute(0, 3, 1, 2))

        # (B, C, H, W) -> (B, H*W, C) for SSM
        x1_flat = self.act(x1_conv).flatten(2).transpose(1, 2)
        x2_flat = self.act(x2_conv).flatten(2).transpose(1, 2)
        # -----------------------------------

        y1_flat, y2_flat = self.cross_ssm(x1_flat, x2_flat)

        y1 = self.out_proj1(y1_flat).view(B, H, W, C)
        y2 = self.out_proj2(y2_flat).view(B, H, W, C)

        return x1_res + self.drop_path1(y1), x2_res + self.drop_path2(y2)



class SkipFusionBlock(nn.Module):

    def __init__(self, dim, num_inputs: int = 3): 
        super().__init__()
        self.projection = nn.Linear(dim * num_inputs, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, *features):

        combined_features = torch.cat(features, dim=-1)
        return self.norm(self.projection(combined_features))

class DecoderFusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, vssm_args={}):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.vss_block = VSSBlock(hidden_dim=out_dim, **vssm_args)
        self.local_enhancer = ConvBlock(dim=out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_cat):

        projected_x = self.projection(x_cat)
        global_fused_x = self.vss_block(projected_x)
        local_enhanced_x = self.local_enhancer(global_fused_x)

        return self.norm(local_enhanced_x)


class ConvBlock(nn.Module):

    def __init__(self, dim, kernel_size=3, expand_ratio=2.0, dropout_rate=0.1):  
        super().__init__()
        hidden_dim = int(dim * expand_ratio)

        self.conv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.Conv2d(hidden_dim, hidden_dim,
                      kernel_size=kernel_size, padding=kernel_size // 2,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(p=dropout_rate),


            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        residual = x
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return residual + x


class SkipFusionNoDEM(nn.Module):
    """
    A special skip fusion block.

    It accepts three inputs (DEM, S1, S2) by convention, but intentionally
    ignores the first input (DEM) and only fuses S1_final and S2_final.
    """

    def __init__(self, dim, num_inputs: int = 3, **kwargs):
        super().__init__()
        self.projection = nn.Linear(dim * (num_inputs - 1), dim)
        self.norm = nn.LayerNorm(dim)
        print(f"✅ [Ablation] Initialized SkipFusionNoDEM: Will ignore the first input feature (raw DEM).")

    def forward(self, *features):

        combined_features = torch.cat(features[1:], dim=-1)

        return self.norm(self.projection(combined_features))

