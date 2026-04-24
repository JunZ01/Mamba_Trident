#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
factory_refactored.py

"""
from typing import Dict, Any
import torch.nn as nn

from .two_stage_model import TwoStageUNet
from .single_stage_model import SingleStageUNet
from .smp_wrapper import SMPWrapper
# =================================================================
from .umamba_bot import UMambaBot
from .swin_umamba import SwinUMamba

def create_model (config: Dict[str, Any]) -> nn.Module:
    """

    Args:
        config (Dict[str, Any])

    Returns:
        nn.Module
    """
    model_type = config.get('model_type')
    if not model_type:
        raise ValueError("config should be contain 'model_type' key")

    print(f"🏭 Creating model of type (refactored factory): {model_type}")
    #
    if model_type == 'single_stage':
        model_init_config = {
            'n_classes': config.get('n_classes'),
            'config': config
        }
        return SingleStageUNet(**model_init_config)

    elif model_type == 'two_stage':
        return TwoStageUNet(n_classes=config['n_classes'], config=config)

    elif model_type == 'umamba_bot':
        return UMambaBot(config=config)

    elif model_type == 'swin_umamba':
        return SwinUMamba(config=config)


    n_classes = config.get('n_classes')
    if not n_classes:
        raise ValueError(f"Factory: 'n_classes' not found in config for model_type '{model_type}'")
    smp_models = [
        'unet_resnet50',
        'deeplabv3p_resnet50',
        'unetplusplus_resnet50',
        'segformer_mitb3',
    ]

    if model_type in smp_models:

        return SMPWrapper(
            model_type=model_type,
            n_classes=n_classes,
            config=config  #
        )
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not registered in the factory.")
