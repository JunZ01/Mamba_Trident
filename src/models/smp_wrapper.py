#!/usr/bin/env python
# src/models/smp_wrapper.py

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Any, Union


class SMPWrapper(nn.Module):
    """
    SMP model wrapper that supports both dict inputs (multi-modal)
    and tensor inputs (single-modal).
    """
    def __init__(self, model_type: str, n_classes: int, config: Dict[str, Any]):
        super().__init__()

        in_chans_dict = config.get('in_channels', {})
        self.input_keys = ['s2', 's1', 'dem']
        self.total_in_channels = sum(in_chans_dict.get(k, 0) for k in self.input_keys)

        if self.total_in_channels == 0:
            raise ValueError("SMPWrapper cannot find 'in_channels' in the config.")

        print(f"✅ [SMPWrapper] Initializing '{model_type}' with {self.total_in_channels} input channels.")

        resize = config.get('resize', [1024, 1024])
        img_size = resize[0] if isinstance(resize, list) else resize

        if model_type == 'unet_resnet50':
            self.model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights=None,
                in_channels=self.total_in_channels,
                classes=n_classes
            )
        elif model_type == 'deeplabv3p_resnet50':
            self.model = smp.DeepLabV3Plus(
                encoder_name="resnet50",
                encoder_weights=None,
                in_channels=self.total_in_channels,
                classes=n_classes
            )
        elif model_type == 'segformer_mitb3':
            self.model = smp.Segformer(
                encoder_name="mit_b3",
                encoder_weights=None,
                in_channels=self.total_in_channels,
                classes=n_classes
            )
        elif model_type == 'unetplusplus_resnet50':
            self.model = smp.UnetPlusPlus(
                encoder_name="resnet50",
                encoder_weights=None,
                in_channels=self.total_in_channels,
                classes=n_classes
            )
        elif model_type == 'segformer_mitb0':
            self.model = smp.Segformer(
                encoder_name="mit_b0",
                encoder_weights=None,
                in_channels=self.total_in_channels,
                classes=n_classes
            )
        else:
            raise NotImplementedError(f"SMPWrapper does not support model_type: {model_type}")

    def forward(self, features: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:

        if isinstance(features, dict):

            s2 = features.get('s2') if features.get('s2') is not None else features.get('optical')
            s1 = features.get('s1') if features.get('s1') is not None else features.get('sar')
            dem = features.get('dem')

            if s2 is None or s1 is None or dem is None:
                raise ValueError(
                    f"SMPWrapper is missing input modalities. "
                    f"S2: {s2 is not None}, S1: {s1 is not None}, DEM: {dem is not None}"
                )

            x = torch.cat([s2, s1, dem], dim=1)

        elif isinstance(features, torch.Tensor):
            x = features

            if x.shape[1] != self.total_in_channels:
                print(
                    f"⚠️ Warning: Input has {x.shape[1]} channels, "
                    f"model expects {self.total_in_channels}"
                )

        else:
            raise TypeError(
                f"SMPWrapper expects a Dict or Tensor, got: {type(features)}"
            )
        return self.model(x)