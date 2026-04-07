#!/usr/bin/env python3
"""Tests for per-category threshold optimization."""

import json

import numpy as np
import pytest

from classivore.training.thresholds import (
    load_thresholds,
    optimize_global_threshold,
    optimize_per_category_thresholds,
    save_thresholds,
)


class TestGlobalThreshold:
    def test_finds_optimal(self):
        # Probabilities that are clearly above/below threshold
        probs = np.array([[0.8, 0.1], [0.9, 0.2], [0.1, 0.7], [0.2, 0.8]])
        labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        threshold, f1 = optimize_global_threshold(probs, labels)

        assert 0.3 <= threshold <= 0.7
        assert f1 > 0.8

    def test_returns_float(self):
        probs = np.random.rand(10, 5)
        labels = (probs > 0.5).astype(int)

        threshold, f1 = optimize_global_threshold(probs, labels)

        assert isinstance(threshold, float)
        assert isinstance(f1, float)


class TestPerCategoryThresholds:
    def test_optimizes_each_category(self):
        np.random.seed(42)
        probs = np.random.rand(100, 5)
        labels = (probs > 0.5).astype(int)
        names = ["A", "B", "C", "D", "E"]

        thresholds = optimize_per_category_thresholds(probs, labels, names)

        assert len(thresholds) == 5
        assert all(0.3 <= t <= 0.7 for t in thresholds.values())

    def test_rare_categories_use_global(self):
        probs = np.zeros((100, 3))
        labels = np.zeros((100, 3), dtype=int)
        # Only 2 positive samples for category 2 (below threshold of 5)
        labels[0, 2] = 1
        labels[1, 2] = 1
        probs[0, 2] = 0.9
        probs[1, 2] = 0.8
        # Plenty of samples for categories 0 and 1
        for i in range(50):
            labels[i, 0] = 1
            probs[i, 0] = 0.7
            labels[50 + i, 1] = 1
            probs[50 + i, 1] = 0.6
        names = ["Common1", "Common2", "Rare"]

        thresholds = optimize_per_category_thresholds(
            probs, labels, names, global_threshold=0.55,
        )

        assert thresholds["Rare"] == 0.55  # Fallback to global

    def test_returns_dict(self):
        probs = np.random.rand(50, 3)
        labels = (probs > 0.5).astype(int)
        names = ["X", "Y", "Z"]

        thresholds = optimize_per_category_thresholds(probs, labels, names)

        assert isinstance(thresholds, dict)
        assert set(thresholds.keys()) == {"X", "Y", "Z"}


class TestSaveLoadThresholds:
    def test_roundtrip(self, tmp_path):
        thresholds = {"Sedan": 0.45, "SUV": 0.55, "Python": 0.50}
        save_thresholds(thresholds, tmp_path)
        loaded = load_thresholds(tmp_path)

        assert loaded == thresholds

    def test_load_missing_returns_none(self, tmp_path):
        assert load_thresholds(tmp_path) is None
