#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
losses.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    High-performance Dice Loss, fully vectorized
    """

    def __init__(self, smooth=1e-5, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) logits, targets: (B, H, W) labels
        inputs = F.softmax(inputs, dim=1)
        n_classes = inputs.shape[1]

        targets_one_hot = F.one_hot(targets, num_classes=n_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)
            inputs = inputs * mask
            targets_one_hot = targets_one_hot * mask

        intersection = (inputs * targets_one_hot).sum(dim=[0, 2, 3])
        denominator = inputs.sum(dim=[0, 2, 3]) + targets_one_hot.sum(dim=[0, 2, 3])

        dice_per_class = (2. * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice_per_class.mean()


class BoundaryLoss(nn.Module):

    def __init__(self, idx_class=None):
        """
        Args:
            idx_class (int, optional):
                If specified, compute the boundary loss only for the given class
                (e.g., LIA=2). If None, average over all foreground classes.
        """
        super().__init__()
        self.idx_class = idx_class

    def compute_edts_forPenalizedLoss(self, GT):
        """
        GT: numpy array, shape (B, H, W), boolean or 0/1
        """
        res = np.zeros(GT.shape)
        for i in range(GT.shape[0]):
            posmask = GT[i]
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        return res

    def forward(self, preds, targets):
        """
        preds: (B, C, H, W) logits
        targets: (B, H, W) labels
        """
        probs = torch.softmax(preds, dim=1)

        if self.idx_class is not None:
            classes = [self.idx_class]
        else:

            classes = range(1, probs.shape[1])

        total_loss = 0.0
        count = 0

        for cls in classes:
            pred_cls = probs[:, cls, :, :]
            gt_cls = (targets == cls).cpu().numpy().astype(bool)
            if not np.any(gt_cls):
                continue

            with torch.no_grad():
                dist_maps = self.compute_edts_forPenalizedLoss(gt_cls)
                dist_maps = torch.from_numpy(dist_maps).float().to(preds.device)

            loss = dist_maps * pred_cls
            total_loss += loss.mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        return total_loss / count

class CombinedLoss(nn.Module):


    def __init__(self, components, device='cuda'):
        super().__init__()
        self.components = []
        self.device = device

        for comp_config in components:
            config_copy = comp_config.copy()
            weight = config_copy.pop('weight', 1.0)
            loss_fn = get_loss_function(config_copy, device)
            self.components.append({'loss': loss_fn, 'weight': weight})

    def forward(self, inputs, targets):
        total_loss = 0
        for comp in self.components:
            total_loss += comp['weight'] * comp['loss'](inputs, targets)
        return total_loss

def get_loss_function(config, device='cuda'):

    loss_type = config.get('loss_type') or config.get('type')

    if loss_type == 'ce':
        return nn.CrossEntropyLoss()

    elif loss_type == 'weighted_ce':
        weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=weights)

    elif loss_type == 'focal':
        alpha = None
        if 'focal_alpha' in config:
            alpha = torch.tensor(config['focal_alpha'], dtype=torch.float32).to(device)
        gamma = config.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == 'dice':
        return DiceLoss(
            smooth=config.get('dice_smooth', 1e-5),
            ignore_index=config.get('ignore_index', None)
        )

    elif loss_type == 'boundary':
        return BoundaryLoss(
            idx_class=config.get('target_class', None)  # If None, compute over all foreground classes
        )

    elif loss_type == 'combined':
        return CombinedLoss(config['loss_components'], device)

    elif loss_type == 'ce_dice':
        return CombinedLoss([
            {'type': 'ce', 'weight': 0.5},
            {'type': 'dice', 'weight': 0.5}
        ], device)

    else:
        available_types = ['ce', 'weighted_ce', 'focal', 'dice', 'boundary', 'combined', 'ce_dice']
        raise ValueError(f"Unknown loss type: {loss_type}. Available types: {available_types}")