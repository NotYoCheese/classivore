#!/usr/bin/env python3
"""Tests for weighted focal loss."""

import torch
import pytest

from classivore.training.loss import WeightedFocalLoss, compute_class_weights


class TestWeightedFocalLoss:
    """Test focal loss computation."""

    def test_basic_loss_is_positive(self):
        loss_fn = WeightedFocalLoss()
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[0, 0] = 1.0
        targets[1, 3] = 1.0

        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_perfect_predictions_low_loss(self):
        loss_fn = WeightedFocalLoss()
        # High logits where targets are 1, low where 0
        logits = torch.full((4, 10), -5.0)
        targets = torch.zeros(4, 10)
        targets[0, 0] = 1.0
        logits[0, 0] = 5.0

        loss = loss_fn(logits, targets)
        assert loss.item() < 0.1

    def test_wrong_predictions_high_loss(self):
        loss_fn = WeightedFocalLoss()
        # Low logits where targets are 1
        logits = torch.full((4, 10), -5.0)
        targets = torch.zeros(4, 10)
        targets[0, 0] = 1.0

        loss = loss_fn(logits, targets)
        assert loss.item() > 0.01

    def test_focal_reduces_easy_example_loss(self):
        """Focal term should reduce loss for well-classified examples."""
        # Compare gamma=0 (standard BCE) vs gamma=3.5
        bce_fn = WeightedFocalLoss(gamma=0.0)
        focal_fn = WeightedFocalLoss(gamma=3.5)

        logits = torch.tensor([[3.0]])  # Easy positive (high confidence)
        targets = torch.tensor([[1.0]])

        bce_loss = bce_fn(logits, targets)
        focal_loss = focal_fn(logits, targets)

        assert focal_loss.item() < bce_loss.item()

    def test_focal_preserves_hard_example_loss(self):
        """Focal term should have less effect on hard examples."""
        bce_fn = WeightedFocalLoss(gamma=0.0)
        focal_fn = WeightedFocalLoss(gamma=3.5)

        logits = torch.tensor([[-3.0]])  # Hard positive (low confidence)
        targets = torch.tensor([[1.0]])

        bce_loss = bce_fn(logits, targets)
        focal_loss = focal_fn(logits, targets)

        # Focal loss should be closer to BCE for hard examples
        ratio = focal_loss.item() / bce_loss.item()
        assert ratio > 0.5  # Not reduced as much as easy examples

    def test_class_weights_applied(self):
        weights = torch.tensor([2.0, 1.0, 0.5])
        loss_fn = WeightedFocalLoss(class_weights=weights)

        logits = torch.randn(2, 3)
        targets = torch.zeros(2, 3)
        targets[0, 0] = 1.0  # Class 0, weight 2.0
        targets[1, 2] = 1.0  # Class 2, weight 0.5

        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_confidence_weights_reduce_loss(self):
        loss_fn = WeightedFocalLoss()
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[0, 0] = 1.0

        full_loss = loss_fn(logits, targets)

        # Half confidence should reduce loss
        conf = torch.full((4, 10), 0.5)
        reduced_loss = loss_fn(logits, targets, confidence_weights=conf)

        assert reduced_loss.item() < full_loss.item()
        assert reduced_loss.item() == pytest.approx(full_loss.item() * 0.5, rel=0.01)

    def test_confidence_weights_none_is_no_op(self):
        loss_fn = WeightedFocalLoss()
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[0, 0] = 1.0

        loss_none = loss_fn(logits, targets, confidence_weights=None)
        loss_ones = loss_fn(logits, targets, confidence_weights=torch.ones(4, 10))

        assert loss_none.item() == pytest.approx(loss_ones.item(), rel=1e-5)

    def test_output_is_scalar(self):
        loss_fn = WeightedFocalLoss()
        logits = torch.randn(8, 698)
        targets = torch.zeros(8, 698)

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        loss_fn = WeightedFocalLoss()
        logits = torch.randn(4, 10, requires_grad=True)
        targets = torch.zeros(4, 10)
        targets[0, 0] = 1.0

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


class TestComputeClassWeights:
    """Test class weight computation."""

    def test_rare_classes_get_higher_weight(self):
        counts = torch.tensor([100.0, 10.0, 1.0])
        weights = compute_class_weights(counts, total_samples=1000)

        assert weights[2] > weights[1] > weights[0]

    def test_cap_applied(self):
        counts = torch.tensor([1.0])  # Very rare
        weights = compute_class_weights(counts, total_samples=100000, cap=7.0)

        assert weights[0].item() == 7.0

    def test_no_normalization(self):
        """Weights should NOT be normalized after capping.

        This was the #1 bug in previous training work. Normalizing
        after capping reduces all weights to ~1.0, negating the effect.
        """
        counts = torch.tensor([1000.0, 100.0, 10.0, 1.0])
        weights = compute_class_weights(counts, total_samples=10000, cap=7.0)

        # If normalized, mean would be ~1.0. It should NOT be.
        assert weights.mean().item() > 1.5

    def test_accepts_list_input(self):
        weights = compute_class_weights([100, 10, 1], total_samples=1000)
        assert weights.shape == (3,)

    def test_zero_count_handled(self):
        counts = torch.tensor([0.0, 100.0])
        weights = compute_class_weights(counts, total_samples=1000)

        # log(1000 / (0 + 1)) = log(1000) ≈ 6.9, should be capped at 7.0
        assert weights[0].item() <= 7.0
        assert weights[0].item() > weights[1].item()

    def test_custom_cap(self):
        counts = torch.tensor([1.0])
        weights_5 = compute_class_weights(counts, total_samples=100000, cap=5.0)
        weights_10 = compute_class_weights(counts, total_samples=100000, cap=10.0)

        assert weights_5[0].item() == 5.0
        assert weights_10[0].item() > 5.0
