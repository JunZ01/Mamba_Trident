#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics.py

"""
import numpy as np
import torch
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')

def _fast_hist(true, pred, n_classes):

    mask = (true >= 0) & (true < n_classes)
    hist = np.bincount(
        n_classes * true[mask].astype(int) + pred[mask],
        minlength=n_classes ** 2,
    ).reshape(n_classes, n_classes)
    return hist


class MetricsManager:
    """
    1. metrics = MetricsManager(n_classes=3)
    2. for preds, targets in val_loader:
    3.     metrics.update(preds, targets)
    4. results = metrics.get_results()
    5. print(results['mIoU'], results['LIA_F1'])
    """

    def __init__(self, n_classes, lia_class_id=2, device='cpu'):
        self.n_classes = n_classes
        self.lia_class_id = lia_class_id
        self.device = device
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self.boundary_f1s = []
        self.connectivity_scores = []
        self.area_errors = []

    def update(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Raw model output (B, C, H, W)
            targets (torch.Tensor): Ground-truth labels (B, H, W)
        """

        pred_labels = preds.argmax(1).cpu().numpy()
        true_labels = targets.cpu().numpy()

        self.confusion_matrix += _fast_hist(true_labels.flatten(), pred_labels.flatten(), self.n_classes)

        for i in range(true_labels.shape[0]):
            true_sample = true_labels[i]
            pred_sample = pred_labels[i]

            if self.lia_class_id is not None and np.any(true_sample == self.lia_class_id):
                self.boundary_f1s.append(self._calculate_boundary_f1(pred_sample, true_sample, self.lia_class_id))
                self.connectivity_scores.append(
                    self._calculate_connectivity_score(pred_sample, true_sample, self.lia_class_id))
                self.area_errors.append(self._calculate_area_error(pred_sample, true_sample, self.lia_class_id))

    def get_results(self, full_report=True):
        """
        Args:
            full_report (bool): Whether to return all detailed metrics.

        Returns:
            dict: A dictionary containing all computed results.
        """
        ious = self._calculate_iou()
        precisions, recalls, f1s = self._calculate_prf1()
        overall_acc, per_class_acc = self._calculate_accuracy()

        results = {
            'mIoU': np.nanmean(ious),
            'IoUs': ious,
            'mPrecision': np.nanmean(precisions),
            'Precisions': precisions,
            'mRecall': np.nanmean(recalls),
            'Recalls': recalls,
            'mF1': np.nanmean(f1s),
            'F1s': f1s,
            'Overall_Accuracy': overall_acc,
            'mAccuracy': np.nanmean(per_class_acc),
            'Accuracies': per_class_acc,
        }

        if full_report and self.lia_class_id is not None:
            results.update({
                f'LIA_IoU': ious[self.lia_class_id],
                f'LIA_F1': f1s[self.lia_class_id],
                f'LIA_Accuracy': per_class_acc[self.lia_class_id],
                f'LIA_Boundary_F1': np.nanmean(self.boundary_f1s) if self.boundary_f1s else 0.0,
                f'LIA_Connectivity': np.nanmean(self.connectivity_scores) if self.connectivity_scores else 0.0,
                f'LIA_Area_Error': np.nanmean(self.area_errors) if self.area_errors else 0.0,
            })

        return results

    def _calculate_iou(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        union = tp + fp + fn
        return (tp / (union + 1e-8)).tolist()

    def _calculate_prf1(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return precision.tolist(), recall.tolist(), f1.tolist()

    def _calculate_accuracy(self):
        """
        Returns:
            tuple: (overall_accuracy, per_class_accuracy)
                - overall_accuracy
                - per_class_accuracy
        """

        total_correct = np.diag(self.confusion_matrix).sum()
        total_samples = self.confusion_matrix.sum()
        overall_accuracy = total_correct / (total_samples + 1e-8)


        per_class_accuracy = []
        for i in range(self.n_classes):
            class_total = self.confusion_matrix[i].sum()
            class_correct = self.confusion_matrix[i, i]
            per_class_accuracy.append(class_correct / (class_total + 1e-8))

        return overall_accuracy, per_class_accuracy


    @staticmethod
    def _calculate_boundary_f1(pred, target, class_id, tolerance=2):
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        if not np.any(pred_mask) and not np.any(target_mask): return 1.0
        if not np.any(pred_mask) or not np.any(target_mask): return 0.0

        pred_boundary = pred_mask ^ ndimage.binary_erosion(pred_mask)
        target_boundary = target_mask ^ ndimage.binary_erosion(target_mask)

        pred_dilated = ndimage.binary_dilation(pred_boundary, iterations=tolerance)
        target_dilated = ndimage.binary_dilation(target_boundary, iterations=tolerance)

        precision = np.sum(target_boundary & pred_dilated) / (np.sum(pred_boundary) + 1e-8)
        recall = np.sum(pred_boundary & target_dilated) / (np.sum(target_boundary) + 1e-8)

        return 2 * precision * recall / (precision + recall + 1e-8)

    @staticmethod
    def _calculate_connectivity_score(pred, target, class_id):
        pred_mask, target_mask = (pred == class_id), (target == class_id)
        if not np.any(pred_mask) and not np.any(target_mask): return 1.0
        if not np.any(pred_mask) or not np.any(target_mask): return 0.0

        _, pred_num = ndimage.label(pred_mask)
        _, target_num = ndimage.label(target_mask)
        return min(pred_num, target_num) / (max(pred_num, target_num) + 1e-8)

    @staticmethod
    def _calculate_area_error(pred, target, class_id):
        pred_area, target_area = np.sum(pred == class_id), np.sum(target == class_id)
        if target_area == 0: return 1.0 if pred_area > 0 else 0.0
        return min(abs(pred_area - target_area) / target_area, 1.0)