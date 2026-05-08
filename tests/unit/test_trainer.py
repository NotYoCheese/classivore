#!/usr/bin/env python3
"""End-to-end trainer test on a toy taxonomy.

Verifies that train_model composes its components correctly and writes the
full inference artifact set to disk. Component-level tests live in
test_training_loss.py, test_training_dataset.py, and test_training_thresholds.py;
this test exercises their composition.

Uses microsoft/deberta-v3-xsmall (the smallest DeBERTa variant) so the test
stays under a minute on CPU once the model is cached. The first run will
download about 44 MB. The test skips cleanly if the model cannot be loaded
locally and no network is available.
"""

import json

import pytest
from transformers import AutoTokenizer

from classivore.config.settings import TaxonomyConfig
from classivore.publishing.artifact import REQUIRED
from classivore.training.trainer import train_model

TINY_MODEL = "microsoft/deberta-v3-xsmall"


@pytest.fixture(scope="module")
def tiny_model_available():
    """Skip if the tiny DeBERTa cannot be loaded (no cache, no network)."""
    try:
        AutoTokenizer.from_pretrained(TINY_MODEL)
    except Exception as e:
        pytest.skip(
            f"{TINY_MODEL} not loadable (no cache and no network): {e}"
        )


def _write_toy_taxonomy(tax_dir):
    """Write a config.yaml + taxonomy.csv with one tier-1 and 5 leaves."""
    tax_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = [
        "id,parent_id,name,display_name,path,depth,is_leaf,children_count",
        "1,,Automotive,Automotive,Automotive,1,False,5",
        "2,1,Sedan,Automotive: Sedan,Automotive > Sedan,2,True,0",
        "3,1,SUV,Automotive: SUV,Automotive > SUV,2,True,0",
        "4,1,Truck,Automotive: Truck,Automotive > Truck,2,True,0",
        "5,1,Coupe,Automotive: Coupe,Automotive > Coupe,2,True,0",
        "6,1,Convertible,Automotive: Convertible,Automotive > Convertible,2,True,0",
    ]
    (tax_dir / "taxonomy.csv").write_text("\n".join(csv_rows) + "\n")

    config_yaml = (
        "name: Toy Taxonomy\n"
        "version: '0.1'\n"
        "slug: toy\n"
        "taxonomy_file: taxonomy.csv\n"
        "id_column: id\n"
        "name_column: name\n"
        "parent_column: parent_id\n"
        f"model_base: {TINY_MODEL}\n"
        "batch_size: 4\n"
        "max_length: 64\n"
        "num_epochs: 1\n"
    )
    (tax_dir / "config.yaml").write_text(config_yaml)


def _write_toy_corpus_and_labels(data_dir, leaf_names, num_pages=50):
    """Write 50 corpus pages plus a matching label_state.json.

    Round-robin assigns one leaf label per page. Each leaf gets ~10 examples
    so MultilabelStratifiedShuffleSplit can split without warnings.
    """
    corpus_dir = data_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = data_dir / "labels" / "toy"
    labels_dir.mkdir(parents=True, exist_ok=True)

    pages_dict = {}
    with open(corpus_dir / "pages.json", "w") as f:
        for i in range(num_pages):
            content_hash = f"hash{i:03d}"
            label = leaf_names[i % len(leaf_names)]
            text = (
                f"This is article {i} about {label}. It discusses the topic "
                f"with details, examples, and analysis spanning several "
                f"sentences so the tokenizer has enough to work with."
            )
            f.write(json.dumps({
                "content_hash": content_hash,
                "url": f"https://example.com/{i}",
                "text": text,
            }) + "\n")
            pages_dict[content_hash] = {
                "url": f"https://example.com/{i}",
                "status": "stage2_complete",
                "tier1_categories": ["Automotive"],
                "labels": [{"name": label, "confidence": 0.95}],
                "reasoning": "toy fixture",
                "error": None,
            }

    with open(labels_dir / "label_state.json", "w") as f:
        json.dump({
            "started_at": "2026-05-01T00:00:00Z",
            "last_checkpoint_at": "2026-05-01T00:00:00Z",
            "stage1_batch_ids": [],
            "stage2_batch_ids": [],
            "stats": {},
            "pages": pages_dict,
        }, f)


def test_train_model_writes_full_artifact_set(tmp_path, tiny_model_available):
    """train_model on a toy taxonomy produces every required inference artifact."""
    tax_dir = tmp_path / "taxonomies" / "toy"
    _write_toy_taxonomy(tax_dir)

    leaf_names = ["Sedan", "SUV", "Truck", "Coupe", "Convertible"]
    data_dir = tmp_path / "data"
    _write_toy_corpus_and_labels(data_dir, leaf_names, num_pages=50)

    config = TaxonomyConfig(tax_dir / "config.yaml")
    output_dir = tmp_path / "models" / "toy" / "run1"

    result = train_model(
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
        device="cpu",
        epochs=1,
        batch_size=4,
    )

    assert result["model_path"] == str(output_dir)

    for name in REQUIRED:
        assert (output_dir / name).exists(), f"missing required artifact: {name}"

    # Tokenizer files. spm.model is the canonical SentencePiece vocab for DeBERTa-v3.
    assert (output_dir / "tokenizer_config.json").exists()
    assert (output_dir / "spm.model").exists()

    # Taxonomy metadata round-trips paths for the toy categories.
    metadata = json.loads((output_dir / "taxonomy_metadata.json").read_text())
    assert metadata["taxonomy_slug"] == "toy"
    assert metadata["taxonomy_version"] == "0.1"
    cats = metadata["categories"]
    for leaf in leaf_names:
        assert leaf in cats
        assert cats[leaf]["path"] == ["Automotive", leaf]

    # Label mappings index the same leaves the dataset built.
    mappings = json.loads((output_dir / "label_mappings.json").read_text())
    assert set(mappings["index_to_name"].values()) == set(leaf_names)

    # Per-category thresholds JSON has an entry per leaf.
    thresholds = json.loads((output_dir / "per_category_thresholds.json").read_text())
    assert set(thresholds.keys()) == set(leaf_names)
