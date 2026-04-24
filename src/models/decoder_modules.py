
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from .backbone_modules import VSSBlock



class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) style channel attention."""

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        return x * self.sigmoid(attn)


class ChannelAttentionBlock(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(ChannelAttentionBlock, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class CVSSDecoderBlock(nn.Module):


    def __init__(self, hidden_dim: int, drop_path: float = 0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()

        self.vss_block = VSSBlock(hidden_dim=hidden_dim, drop_path=drop_path, **kwargs)


        self.norm = norm_layer(hidden_dim)
        self.channel_attn_block = ChannelAttentionBlock(num_feat=hidden_dim)
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):

        x = self.vss_block(x)

        x_normed = self.norm(x)
        x_normed_permuted = x_normed.permute(0, 3, 1, 2).contiguous()  # -> (B, C, H, W)
        attn_out = self.channel_attn_block(x_normed_permuted)
        attn_out_permuted = attn_out.permute(0, 2, 3, 1).contiguous()  # -> (B, H, W, C)
        x = x + self.scale * attn_out_permuted

        return x


class FinalUpsample_X4(nn.Module):

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: (B, H, W, C)
        """

        x = self.linear1(x).permute(0, 3, 1, 2).contiguous()  # -> B, C, H, W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()  # -> B, 2H, 2W, C

        x = self.linear2(x).permute(0, 3, 1, 2).contiguous()  # -> B, C, 2H, 2W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()  # -> B, 4H, 4W, C

        x = self.norm(x)
        return x