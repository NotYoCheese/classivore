#!/usr/bin/env python3
"""Self-contained inference engine for trained classivore models.

No classivore internal dependencies — only torch, transformers, numpy, json.
This module can be imported by classivore-api without pulling in the training stack.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_OVERLAP = 128
DEFAULT_BATCH_SIZE = 32


def _select_device(override=None):
    """Auto-detect best available device.

    Returns:
        Tuple of (device string, use_fp16 bool).
    """
    if override:
        return override, override == "cuda"
    if torch.cuda.is_available():
        return "cuda", True
    if torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def _run_inference(model, tokenizer, texts, batch_size, max_length, device,
                   problem_type="multi_label_classification"):
    """Run batched inference and return activation-appropriate probabilities.

    Args:
        model: HuggingFace model in eval mode.
        tokenizer: Tokenizer instance.
        texts: List of text strings.
        batch_size: Inference batch size.
        max_length: Max token length.
        device: Device string.
        problem_type: One of:
            - "multi_label_classification" → sigmoid (independent per-class probs)
            - "single_label_classification" → softmax (probs sum to 1 per row)
            - "regression" → raises NotImplementedError

    Returns:
        Numpy array of shape (num_texts, num_classes) with probabilities.
    """
    if problem_type == "regression":
        raise NotImplementedError(
            "regression models are not supported by the inference pipeline"
        )

    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoding = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoding).logits
            if problem_type == "single_label_classification":
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


class Classifier:
    """Self-contained inference engine for a trained classivore model.

    All metadata (label mappings, thresholds, taxonomy paths) is loaded
    from the model directory. No taxonomy CSV or config is needed.

    Usage:
        classifier = Classifier("models/iab-2.2/20260406_193556")
        results = classifier.predict("Article about electric vehicles...")
        # [{"name": "Electric Vehicles", "id": "42", "path": [...], "confidence": 0.93}]
    """

    def __init__(self, model_dir, device=None):
        """Load model and all inference artifacts from a model directory.

        Args:
            model_dir: Path to trained model directory.
            device: Override device ("cuda", "mps", "cpu"). Auto-detects if None.
        """
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Device selection
        self.device, use_fp16 = _select_device(device)

        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
        )
        self.model.to(self.device)
        if use_fp16:
            self.model.half()
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        # Max length and problem type from model config
        with open(self.model_dir / "config.json") as f:
            model_config = json.load(f)
        self.max_length = model_config.get("max_position_embeddings", 512)
        # HuggingFace writes problem_type when the training script sets it; when missing
        # (or null), default to multi_label — the historical classivore behavior.
        self.problem_type = model_config.get("problem_type") or "multi_label_classification"

        # Label mappings
        with open(self.model_dir / "label_mappings.json") as f:
            mappings = json.load(f)
        self.index_to_name = mappings["index_to_name"]
        self.index_to_id = mappings.get("index_to_id", {})
        self.num_classes = len(self.index_to_name)

        # Per-category thresholds
        thresholds_path = self.model_dir / "per_category_thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                thresholds = json.load(f)
        else:
            thresholds = {}

        # Default threshold: 0.5 for sigmoid (multi-label), 0.0 for softmax (single-label).
        # Softmax probabilities for many-class problems are often <0.5 even for the winning
        # class, so the multi-label default would suppress valid predictions.
        default_threshold = 0.0 if self.problem_type == "single_label_classification" else 0.5
        self.threshold_vec = np.array([
            thresholds.get(self.index_to_name[str(i)], default_threshold)
            for i in range(self.num_classes)
        ])

        # Taxonomy metadata (optional — graceful degradation)
        meta_path = self.model_dir / "taxonomy_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.taxonomy_metadata = json.load(f)
        else:
            logger.warning("taxonomy_metadata.json not found in %s — path info unavailable", self.model_dir)
            self.taxonomy_metadata = None

        logger.info("Classifier loaded: %d categories, device=%s", self.num_classes, self.device)

    def predict(self, text, aggregation="max"):
        """Classify a single text.

        Args:
            text: Input text string.
            aggregation: How to aggregate chunk probabilities ("max" or "mean").

        Returns:
            List of dicts with name, id, confidence, and optionally path.
            Sorted by confidence descending. Only includes categories above threshold.
        """
        return self.predict_batch([text], aggregation=aggregation)[0]

    def predict_batch(self, texts, batch_size=None, aggregation="max"):
        """Classify multiple texts.

        Args:
            texts: List of input text strings.
            batch_size: Inference batch size (default: 32).
            aggregation: How to aggregate chunk probabilities ("max" or "mean").

        Returns:
            List of lists — one result list per input text.
        """
        batch_size = batch_size or DEFAULT_BATCH_SIZE

        # Chunk texts that exceed max_length
        all_chunks = []
        chunk_map = []  # (text_index, num_chunks) pairs
        needs_chunking = False

        for i, text in enumerate(texts):
            chunks = self._chunk_text(text)
            all_chunks.extend(chunks)
            chunk_map.append((i, len(chunks)))
            if len(chunks) > 1:
                needs_chunking = True

        # Run inference on all chunks
        if not needs_chunking:
            # Fast path: no chunking needed, texts == chunks
            all_probs = _run_inference(
                self.model, self.tokenizer, texts,
                batch_size, self.max_length, self.device,
                problem_type=self.problem_type,
            )
        else:
            all_probs = _run_inference(
                self.model, self.tokenizer, all_chunks,
                batch_size, self.max_length, self.device,
                problem_type=self.problem_type,
            )

        # Regroup and aggregate per text
        results = []
        offset = 0
        for text_idx, num_chunks in chunk_map:
            chunk_probs = all_probs[offset:offset + num_chunks]
            offset += num_chunks

            if num_chunks == 1:
                text_probs = chunk_probs[0]
            else:
                text_probs = self._aggregate_probs(chunk_probs, aggregation)

            results.append(self._apply_thresholds(text_probs))

        return results

    def _chunk_text(self, text):
        """Split text into overlapping chunks if it exceeds max_length.

        Args:
            text: Input text string.

        Returns:
            List of text chunk strings (length 1 if no chunking needed).
        """
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        usable_length = self.max_length - 2  # Reserve for [CLS] and [SEP]

        if len(token_ids) <= usable_length:
            return [text]

        stride = usable_length - DEFAULT_OVERLAP
        chunks = []
        for start in range(0, len(token_ids), stride):
            chunk_ids = token_ids[start:start + usable_length]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            if start + usable_length >= len(token_ids):
                break

        return chunks

    def _aggregate_probs(self, chunk_probs, aggregation):
        """Aggregate probabilities across chunks.

        Args:
            chunk_probs: Array of shape (num_chunks, num_classes).
            aggregation: "max" or "mean".

        Returns:
            Array of shape (num_classes,).
        """
        if aggregation == "mean":
            return np.mean(chunk_probs, axis=0)
        return np.max(chunk_probs, axis=0)

    def _apply_thresholds(self, probs):
        """Apply per-category thresholds and build result dicts.

        Args:
            probs: Array of shape (num_classes,).

        Returns:
            List of dicts sorted by confidence descending.
        """
        above = probs > self.threshold_vec
        indices = np.where(above)[0]

        categories = self.taxonomy_metadata.get("categories", {}) if self.taxonomy_metadata else {}

        results = []
        for idx in indices:
            name = self.index_to_name[str(idx)]
            entry = {
                "name": name,
                "id": self.index_to_id.get(str(idx), ""),
                "confidence": round(float(probs[idx]), 4),
            }
            meta = categories.get(name)
            if meta:
                entry["path"] = meta["path"]
            results.append(entry)

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
