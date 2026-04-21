#!/usr/bin/env python3
"""Tests for the self-contained Classifier inference engine."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from classivore.inference.classifier import Classifier, _run_inference


# --- _run_inference activation dispatch ---

def _mock_tokenizer(num_texts):
    tok = MagicMock()
    tok.return_value = MagicMock()
    tok.return_value.to = MagicMock(return_value={})
    return tok


def _mock_model(logits):
    model = MagicMock()
    output = MagicMock()
    output.logits = logits
    model.return_value = output
    return model


class TestRunInferenceActivation:
    def test_multi_label_applies_sigmoid(self):
        logits = torch.tensor([[0.0, 1.0, -1.0]])
        model = _mock_model(logits)
        probs = _run_inference(
            model, _mock_tokenizer(1), ["hello"],
            batch_size=1, max_length=512, device="cpu",
            problem_type="multi_label_classification",
        )
        expected = torch.sigmoid(logits).numpy()
        assert np.allclose(probs, expected)
        # Sigmoid rows do NOT generally sum to 1
        assert not np.isclose(probs.sum(axis=1)[0], 1.0)

    def test_single_label_applies_softmax(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        model = _mock_model(logits)
        probs = _run_inference(
            model, _mock_tokenizer(2), ["a", "b"],
            batch_size=2, max_length=512, device="cpu",
            problem_type="single_label_classification",
        )
        # Each row sums to ~1.0 under softmax
        assert np.allclose(probs.sum(axis=1), 1.0)
        # Uniform logits row → uniform probabilities
        assert np.allclose(probs[1], [1 / 3, 1 / 3, 1 / 3])

    def test_regression_raises(self):
        with pytest.raises(NotImplementedError, match="regression"):
            _run_inference(
                _mock_model(torch.tensor([[0.0]])), _mock_tokenizer(1), ["a"],
                batch_size=1, max_length=512, device="cpu",
                problem_type="regression",
            )

    def test_default_is_multi_label(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        model = _mock_model(logits)
        probs = _run_inference(
            model, _mock_tokenizer(1), ["a"],
            batch_size=1, max_length=512, device="cpu",
        )
        expected = torch.sigmoid(logits).numpy()
        assert np.allclose(probs, expected)

    def test_unknown_problem_type_falls_back_to_sigmoid(self):
        logits = torch.tensor([[1.0, 2.0]])
        model = _mock_model(logits)
        probs = _run_inference(
            model, _mock_tokenizer(1), ["a"],
            batch_size=1, max_length=512, device="cpu",
            problem_type="future_problem_type",
        )
        expected = torch.sigmoid(logits).numpy()
        assert np.allclose(probs, expected)


# --- Classifier init reads problem_type from config.json ---

def _write_model_dir(tmp_path, problem_type=None, num_classes=3, include_thresholds=False):
    """Write a minimal model-directory scaffold for Classifier init."""
    config = {"max_position_embeddings": 512}
    if problem_type is not None:
        config["problem_type"] = problem_type
    (tmp_path / "config.json").write_text(json.dumps(config))

    mappings = {
        "index_to_name": {str(i): f"cat{i}" for i in range(num_classes)},
        "index_to_id": {str(i): str(i + 100) for i in range(num_classes)},
    }
    (tmp_path / "label_mappings.json").write_text(json.dumps(mappings))

    if include_thresholds:
        thresholds = {f"cat{i}": 0.3 for i in range(num_classes)}
        (tmp_path / "per_category_thresholds.json").write_text(json.dumps(thresholds))

    return tmp_path


def _stub_classifier_loaders():
    """Patch model + tokenizer loaders so Classifier.__init__ doesn't hit disk/GPU."""
    return (
        patch(
            "classivore.inference.classifier.AutoModelForSequenceClassification.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "classivore.inference.classifier.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "classivore.inference.classifier._select_device",
            return_value=("cpu", False),
        ),
    )


class TestClassifierProblemType:
    def test_reads_multi_label_from_config(self, tmp_path):
        _write_model_dir(tmp_path, problem_type="multi_label_classification")
        m1, m2, m3 = _stub_classifier_loaders()
        with m1, m2, m3:
            clf = Classifier(tmp_path)
        assert clf.problem_type == "multi_label_classification"

    def test_reads_single_label_from_config(self, tmp_path):
        _write_model_dir(tmp_path, problem_type="single_label_classification")
        m1, m2, m3 = _stub_classifier_loaders()
        with m1, m2, m3:
            clf = Classifier(tmp_path)
        assert clf.problem_type == "single_label_classification"

    def test_defaults_to_multi_label_when_missing(self, tmp_path):
        _write_model_dir(tmp_path, problem_type=None)
        m1, m2, m3 = _stub_classifier_loaders()
        with m1, m2, m3:
            clf = Classifier(tmp_path)
        assert clf.problem_type == "multi_label_classification"

    def test_default_threshold_is_0_5_for_multi_label(self, tmp_path):
        _write_model_dir(tmp_path, problem_type="multi_label_classification")
        m1, m2, m3 = _stub_classifier_loaders()
        with m1, m2, m3:
            clf = Classifier(tmp_path)
        assert np.allclose(clf.threshold_vec, 0.5)

    def test_default_threshold_is_0_0_for_single_label(self, tmp_path):
        _write_model_dir(tmp_path, problem_type="single_label_classification")
        m1, m2, m3 = _stub_classifier_loaders()
        with m1, m2, m3:
            clf = Classifier(tmp_path)
        assert np.allclose(clf.threshold_vec, 0.0)

    def test_per_category_thresholds_override_default(self, tmp_path):
        _write_model_dir(tmp_path, problem_type="single_label_classification",
                         include_thresholds=True)
        m1, m2, m3 = _stub_classifier_loaders()
        with m1, m2, m3:
            clf = Classifier(tmp_path)
        assert np.allclose(clf.threshold_vec, 0.3)


# --- Classifier.predict full-path tests for each problem_type ---

class TestClassifierPredict:
    def _build(self, tmp_path, problem_type, logits):
        """Build a Classifier where model forward returns the given logits."""
        _write_model_dir(tmp_path, problem_type=problem_type, num_classes=logits.shape[-1])

        model = MagicMock()
        output = MagicMock()
        output.logits = logits
        model.return_value = output
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock(return_value=model)

        # Tokenizer returns encoding + token ids for chunking
        encoding = MagicMock()
        encoding.to = MagicMock(return_value={})

        def tok_call(*args, **kwargs):
            if kwargs.get("add_special_tokens") is False:
                return {"input_ids": [0, 1, 2, 3]}  # short — no chunking
            return encoding

        tokenizer = MagicMock(side_effect=tok_call)

        with patch(
            "classivore.inference.classifier.AutoModelForSequenceClassification.from_pretrained",
            return_value=model,
        ), patch(
            "classivore.inference.classifier.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ), patch(
            "classivore.inference.classifier._select_device", return_value=("cpu", False),
        ):
            return Classifier(tmp_path)

    def test_multi_label_sigmoid_thresholding(self, tmp_path):
        # Logits: class 0 low, class 1 high, class 2 mid
        # Sigmoid: ~[0.27, 0.88, 0.62] → class 1 and 2 exceed 0.5
        logits = torch.tensor([[-1.0, 2.0, 0.5]])
        clf = self._build(tmp_path, "multi_label_classification", logits)
        results = clf.predict("some text")

        names = [r["name"] for r in results]
        assert "cat1" in names
        assert "cat2" in names
        assert "cat0" not in names
        # Results sorted descending by confidence
        assert results[0]["confidence"] >= results[1]["confidence"]

    def test_single_label_returns_all_classes_sorted(self, tmp_path):
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        clf = self._build(tmp_path, "single_label_classification", logits)
        results = clf.predict("some text")

        # All 3 classes returned (default threshold 0.0 for softmax)
        assert len(results) == 3
        # Sorted descending
        assert results[0]["name"] == "cat1"
        assert results[1]["name"] == "cat2"
        assert results[2]["name"] == "cat0"
        # Softmax sum ~= 1.0
        total = sum(r["confidence"] for r in results)
        assert abs(total - 1.0) < 0.01

    def test_regression_raises_on_predict(self, tmp_path):
        logits = torch.tensor([[0.5]])
        clf = self._build(tmp_path, "regression", logits)
        with pytest.raises(NotImplementedError, match="regression"):
            clf.predict("some text")
