"""Testes do carregamento das duas representações.

Os testes sintéticos fixam o contrato do loader sem depender dos dados (que não vão
para o git). O teste com os dados reais roda só quando ``data/`` existe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import EVENT_IDS, EVENT_OCCURRENCE_MATRIX_PATH, EVENT_TRACES_PATH
from src.data.loader import load_event_traces, load_occurrence_matrix

# Três blocos: dois normais e uma anomalia de Type 21. O segundo normal começa com
# eventos simultâneos (intervalo 0), o caso que a etapa 3 precisa enxergar.
_LABELS = "BlockId,Label\nblk_a,Normal\nblk_b,Normal\nblk_c,Anomaly\n"
_TRACES = (
    "BlockId,Label,Type,Features,TimeInterval,Latency\n"
    'blk_a,Success,,"[E5,E22,E5]","[1.0, 2.0]",3\n'
    'blk_b,Success,,"[E5,E5,E11,E9]","[0.0, 0.0, 7.0]",7\n'
    'blk_c,Fail,21,"[E5,E23,E21]","[4.0, 0.0]",4\n'
)


def _matrix_csv(rows: list[tuple[str, str, str, dict[str, int]]]) -> str:
    header = "BlockId,Label,Type," + ",".join(EVENT_IDS)
    lines = [header]
    for block, label, type_, counts in rows:
        values = ",".join(str(counts.get(e, 0)) for e in EVENT_IDS)
        lines.append(f"{block},{label},{type_},{values}")
    return "\n".join(lines) + "\n"


_MATRIX = _matrix_csv(
    [
        ("blk_a", "Success", "", {"E5": 2, "E22": 1}),
        ("blk_b", "Success", "", {"E5": 2, "E11": 1, "E9": 1}),
        ("blk_c", "Fail", "21", {"E5": 1, "E23": 1, "E21": 1}),
    ]
)


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "labels": tmp_path / "anomaly_label.csv",
        "traces": tmp_path / "Event_traces.csv",
        "matrix": tmp_path / "Event_occurrence_matrix.csv",
    }
    paths["labels"].write_text(_LABELS)
    paths["traces"].write_text(_TRACES)
    paths["matrix"].write_text(_MATRIX)
    return paths


def _traces(files: dict[str, Path]):
    return load_event_traces(files["traces"], files["labels"], expected_counts=None)


def _matrix(files: dict[str, Path]):
    return load_occurrence_matrix(
        files["matrix"], files["labels"], expected_counts=None
    )


# --- Sequências -------------------------------------------------------------


def test_traces_preserve_order_and_encode_by_template_number(files):
    traces = _traces(files)
    assert [e.tolist() for e in traces.events] == [
        [5, 22, 5],
        [5, 5, 11, 9],
        [5, 23, 21],
    ]
    assert all(e.dtype == np.int8 for e in traces.events)


def test_traces_timestamps_are_cumulative_from_first_event(files):
    traces = _traces(files)
    assert [t.tolist() for t in traces.timestamps] == [
        [0, 1, 3],
        [0, 0, 0, 7],
        [0, 4, 4],
    ]


def test_traces_reject_interval_length_mismatch(files):
    files["traces"].write_text(_TRACES.replace('"[1.0, 2.0]",3', '"[1.0]",1'))
    with pytest.raises(ValueError, match="um elemento a menos"):
        _traces(files)


def test_traces_reject_latency_mismatch(files):
    files["traces"].write_text(_TRACES.replace('"[1.0, 2.0]",3', '"[1.0, 2.0]",9'))
    with pytest.raises(ValueError, match="Latency"):
        _traces(files)


def test_traces_reject_unknown_event(files):
    files["traces"].write_text(_TRACES.replace("[E5,E22,E5]", "[E5,E30,E5]"))
    with pytest.raises(ValueError, match="E1..E29"):
        _traces(files)


# --- Matriz -----------------------------------------------------------------


def test_matrix_columns_follow_numeric_event_order(files):
    matrix = _matrix(files)
    assert matrix.event_ids == EVENT_IDS
    assert matrix.counts.shape == (3, len(EVENT_IDS))
    assert matrix.counts[1, EVENT_IDS.index("E11")] == 1


def test_matrix_column_order_in_csv_does_not_matter(files):
    df = pd.read_csv(files["matrix"])
    shuffled = ["BlockId", "Label", "Type", *reversed(EVENT_IDS)]
    df[shuffled].to_csv(files["matrix"], index=False)
    assert _matrix(files).counts[0, EVENT_IDS.index("E22")] == 1


# --- Rótulos e metadados ----------------------------------------------------


def test_blocks_labels_and_types(files):
    for rep in (_traces(files), _matrix(files)):
        assert rep.blocks.index.tolist() == ["blk_a", "blk_b", "blk_c"]
        assert rep.blocks["label"].tolist() == [0, 0, 1]
        assert rep.blocks["type"].isna().tolist() == [True, True, False]
        assert rep.blocks["type"].iloc[2] == 21


def test_inline_label_must_agree_with_label_file(files):
    files["labels"].write_text(_LABELS.replace("blk_a,Normal", "blk_a,Anomaly"))
    with pytest.raises(ValueError):
        _traces(files)


def test_row_order_must_match_label_file(files):
    lines = _LABELS.splitlines()
    files["labels"].write_text(
        "\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n"
    )
    with pytest.raises(ValueError, match="ordem"):
        _matrix(files)


def test_type_only_on_anomalies(files):
    files["traces"].write_text(_TRACES.replace("blk_a,Success,,", "blk_a,Success,3,"))
    with pytest.raises(ValueError, match="Type"):
        _traces(files)


def test_expected_counts_are_enforced(files):
    with pytest.raises(ValueError, match="esperados"):
        load_occurrence_matrix(files["matrix"], files["labels"], expected_counts=(3, 2))


# --- Dados reais ------------------------------------------------------------

_HAS_DATA = EVENT_TRACES_PATH.exists() and EVENT_OCCURRENCE_MATRIX_PATH.exists()


@pytest.mark.skipif(not _HAS_DATA, reason="dados do HDFS_v1 ausentes em data/")
def test_real_representations_agree():
    """As duas representações descrevem os mesmos blocos com os mesmos eventos."""
    matrix = load_occurrence_matrix()
    traces = load_event_traces()

    assert matrix.blocks.equals(traces.blocks)
    recounted = np.stack(
        [np.bincount(e, minlength=len(EVENT_IDS) + 1)[1:] for e in traces.events]
    )
    np.testing.assert_array_equal(recounted, matrix.counts)