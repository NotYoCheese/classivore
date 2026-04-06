#!/usr/bin/env python3
"""Weighted focal loss for multi-label classification.

Combines class weighting with focal loss to handle extreme class imbalance
(698 categories, 220:1 ratio). Supports per-sample confidence weighting
so uncertain labels contribute less to the loss.

Key parameters (validated over 25+ training iterations):
- alpha=0.75 (not the paper default of 0.25)
- gamma=3.5 (aggressive hard-example focus)
- class_weight_cap=7.0 (NEVER normalize after capping)
"""

import math

import torch
import torch.nn as nn


class WeightedFocalLoss(nn.Module):
    """Binary cross-entropy focal loss with class weights and confidence weighting.

    Focal loss formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Combined with per-class weights and optional per-sample confidence weights
    for soft-label training.
    """

    def __init__(self, alpha=0.75, gamma=3.5, class_weights=None):
        """Initialize focal loss.

        Args:
            alpha: Scaling factor for positive class. Higher values give more
                weight to positive examples. Default 0.75 (validated optimal
                for 698-category IAB taxonomy).
            gamma: Focusing parameter. Higher values increase focus on hard
                examples. Default 3.5.
            class_weights: Optional 1D tensor of per-class weights. Applied
                multiplicatively to the loss for each class.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )

    def forward(self, logits, targets, confidence_weights=None):
        """Compute weighted focal loss.

        Args:
            logits: Raw model output, shape (batch, num_classes).
            targets: Binary labels, shape (batch, num_classes).
            confidence_weights: Optional per-sample confidence weights,
                shape (batch, num_classes). Values in [0, 1]. When None,
                all samples get equal weight (1.0).

        Returns:
            Scalar loss value.
        """
        probs = torch.sigmoid(logits)
        # Clamp to avoid log(0)
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)

        # Binary cross-entropy components
        bce_pos = -targets * torch.log(probs)
        bce_neg = -(1 - targets) * torch.log(1 - probs)

        # Focal modulation
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting (alpha for positive, 1-alpha for negative)
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Combine
        loss = alpha_weight * focal_weight * (bce_pos + bce_neg)

        # Apply per-class weights
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        # Apply per-sample confidence weights
        if confidence_weights is not None:
            loss = loss * confidence_weights

        return loss.mean()


def compute_class_weights(label_counts, total_samples, cap=7.0):
    """Compute class weights from label frequency.

    Formula: weight = log(total_samples / (positive_count + 1.0))
    Capped at `cap`. NEVER normalize after capping — normalization
    reduces all weights to ~1.0 and negates the effect entirely.
    (This was the single biggest bug fix in previous training work,
    improving F1 by 82%.)

    Args:
        label_counts: 1D tensor or list of positive sample counts per class.
        total_samples: Total number of training samples.
        cap: Maximum weight value. Default 7.0.

    Returns:
        1D tensor of class weights.
    """
    if not isinstance(label_counts, torch.Tensor):
        label_counts = torch.tensor(label_counts, dtype=torch.float32)

    weights = torch.log(total_samples / (label_counts + 1.0))
    weights = torch.clamp(weights, max=cap)
    # DO NOT NORMALIZE. See docstring.
    return weights
