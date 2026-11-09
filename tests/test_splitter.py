"""Testes da divisão treino/validação/teste (decisões, seção 4.1).

O teste central é ``test_no_anomaly_in_train``: se alguma anomalia vazar para o treino,
os três modelos passam a aprender anomalias como normalidade e todo resultado do
trabalho fica comprometido sem aviso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ANOMALY_VALIDATION_FRACTION,
    EVENT_OCCURRENCE_MATRIX_PATH,
    NORMAL_TRAIN_FRACTION,
    NORMAL_VALIDATION_FRACTION,
)
from src.data.splitter import split_blocks


def _blocks(n_normal: int, anomalies_per_type: dict[int, int]) -> pd.DataFrame:
    labels = [0] * n_normal
    types: list[int | None] = [None] * n_normal
    for type_, n in anomalies_per_type.items():
        labels += [1] * n
        types += [type_] * n
    # Embaralha para que normais e anomalias não fiquem em blocos contíguos, como no
    # arquivo real.
    order = np.random.default_rng(0).permutation(len(labels))
    return pd.DataFrame(
        {
            "label": np.array(labels, dtype=np.int8)[order],
            "type": pd.array(types, dtype="Int64")[order],
        },
        index=[f"blk_{i}" for i in range(len(labels))],
    )


@pytest.fixture
def blocks() -> pd.DataFrame:
    # Inclui um tipo com um único bloco e um com contagem ímpar: os casos-limite do
    # arredondamento.
    return _blocks(n_normal=1000, anomalies_per_type={0: 41, 5: 120, 13: 1, 21: 7})


def _assert_no_anomaly_in_train(blocks: pd.DataFrame, seed: int) -> None:
    split = split_blocks(blocks, seed=seed)
    leaked = blocks.iloc[split.train]
    leaked = leaked[leaked["label"] == 1]
    assert leaked.empty, (
        f"{len(leaked)} anomalias vazaram para o treino (seed={seed}): "
        f"{leaked.index[:5].tolist()}"
    )


@pytest.mark.parametrize("seed", [0, 1, 42, 123, 2024])
def test_no_anomaly_in_train(blocks, seed):
    _assert_no_anomaly_in_train(blocks, seed)


def test_partitions_are_disjoint_and_cover_all_blocks(blocks):
    split = split_blocks(blocks)
    all_indices = np.concatenate([split.train, split.validation, split.test])
    assert len(all_indices) == len(blocks)
    assert len(np.unique(all_indices)) == len(blocks)


def test_normal_proportions(blocks):
    split = split_blocks(blocks)
    labels = blocks["label"].to_numpy()
    n_normal = int((labels == 0).sum())
    assert len(split.train) == int(NORMAL_TRAIN_FRACTION * n_normal)
    n_val_normal = int((labels[split.validation] == 0).sum())
    assert n_val_normal == int(
        (NORMAL_TRAIN_FRACTION + NORMAL_VALIDATION_FRACTION) * n_normal
    ) - int(NORMAL_TRAIN_FRACTION * n_normal)


def test_anomalies_split_within_each_type(blocks):
    split = split_blocks(blocks)
    for type_, group in blocks[blocks["label"] == 1].groupby("type"):
        positions = blocks.index.get_indexer(group.index)
        n_val = np.isin(positions, split.validation).sum()
        assert n_val == int(ANOMALY_VALIDATION_FRACTION * len(group)), type_


def test_same_seed_reproduces_split(blocks):
    a, b = split_blocks(blocks, seed=7), split_blocks(blocks, seed=7)
    for part in ("train", "validation", "test"):
        np.testing.assert_array_equal(getattr(a, part), getattr(b, part))


def test_different_seeds_give_different_splits(blocks):
    a, b = split_blocks(blocks, seed=1), split_blocks(blocks, seed=2)
    assert not np.array_equal(a.train, b.train)


def test_anomaly_without_type_is_rejected(blocks):
    broken = blocks.copy()
    first_anomaly = broken.index[broken["label"] == 1][0]
    broken.loc[first_anomaly, "type"] = pd.NA
    with pytest.raises(ValueError, match="type"):
        split_blocks(broken)


@pytest.mark.skipif(
    not EVENT_OCCURRENCE_MATRIX_PATH.exists(),
    reason="dados do HDFS_v1 ausentes em data/",
)
def test_no_anomaly_in_train_real_data():
    from src.data.loader import load_occurrence_matrix

    _assert_no_anomaly_in_train(load_occurrence_matrix().blocks, seed=42)
