"""Testes das etapas 1 e 2 da classificação sequencial × não-sequencial (decisões, seção 5).

O teste central é ``test_precedence_ignores_blocks_missing_one_event``: se blocos sem
um dos eventos do par entrassem na estimativa, a ausência de um evento (sinal
não-sequencial) apareceria como violação de ordem, e tipos não-sequenciais seriam
classificados como sequenciais sem aviso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import EVENT_IDS, EVENT_TRACES_PATH, MIN_PAIR_SUPPORT
from src.data.loader import EventTraces, load_event_traces
from src.labeling.sequentiality import (
    BOTH_SIGNALS,
    NON_SEQUENTIAL,
    SEQUENTIAL,
    UNDETERMINED,
    classify_types,
    distinctive_events,
    length_signal,
    precedence_test,
    simultaneity_check,
)


def _traces(blocks: list[tuple]) -> EventTraces:
    """Monta ``EventTraces`` a partir de (eventos, type ou None = normal[, carimbos]).

    Sem carimbos, cada evento fica 1 s depois do anterior — nenhum par simultâneo.
    """
    events = [
        np.array([EVENT_IDS.index(e) + 1 for e in b[0]], dtype=np.int8) for b in blocks
    ]
    timestamps = [
        np.array(b[2] if len(b) > 2 else range(len(b[0])), dtype=np.int64)
        for b in blocks
    ]
    return EventTraces(
        events=events,
        timestamps=timestamps,
        blocks=pd.DataFrame(
            {
                "label": np.array([b[1] is not None for b in blocks], dtype=np.int8),
                "type": pd.array([b[1] for b in blocks], dtype="Int64"),
            },
            index=[f"blk_{i}" for i in range(len(blocks))],
        ),
    )


def _row(df: pd.DataFrame, **keys) -> pd.Series:
    mask = np.logical_and.reduce([df[k] == v for k, v in keys.items()])
    match = df[mask]
    assert len(match) == 1
    return match.iloc[0]


# --- Etapa 1 ----------------------------------------------------------------


def test_distinctive_event_requires_both_thresholds():
    # 20 normais: E7 em 0 (< 5%), E8 em 1 (= 5%, não passa: desigualdade estrita).
    normals = [(["E5"], None)] * 19 + [(["E5", "E8"], None)]
    # Type 1: E7 e E8 em 3 de 4 blocos; E9 em exatamente metade (não passa).
    type1 = [
        (["E5", "E7", "E8", "E9"], 1),
        (["E5", "E7", "E8", "E9"], 1),
        (["E5", "E7", "E8"], 1),
        (["E5"], 1),
    ]
    df = distinctive_events(_traces(normals + type1))

    assert set(df.loc[df["is_distinctive"], "event"]) == {"E7"}
    e8 = _row(df, type=1, event="E8")
    assert e8["type_freq"] == pytest.approx(0.75)
    assert e8["normal_freq"] == pytest.approx(0.05)
    assert _row(df, type=1, event="E9")["type_freq"] == pytest.approx(0.5)


def test_distinctive_uses_presence_not_count():
    normals = [(["E5"], None)] * 10
    # E7 repetido num único bloco de quatro: presença 25%, não 4/7 das ocorrências.
    type1 = [(["E7", "E7", "E7", "E7"], 1), (["E5"], 1), (["E5"], 1), (["E5"], 1)]
    df = distinctive_events(_traces(normals + type1))
    row = _row(df, type=1, event="E7")
    assert row["type_freq"] == pytest.approx(0.25)
    assert not row["is_distinctive"]


def test_distinctive_reports_every_type_and_event():
    blocks = [(["E5"], None), (["E7"], 1), (["E7"], 3), (["E7"], 3)]
    df = distinctive_events(_traces(blocks))
    assert len(df) == 2 * len(EVENT_IDS)
    assert df.drop_duplicates("type").set_index("type")["n_blocks"].to_dict() == {
        1: 1,
        3: 2,
    }


def test_length_signal_marks_type_outside_normal_range():
    # Normais com comprimento de 3 a 5; Type 3 truncado em 2 eventos, como no HDFS.
    normals = [(["E5", "E22", "E5"], None)] * 5 + [(["E5", "E22", "E5", "E11", "E9"], None)]
    type3 = [(["E5", "E22"], 3)] * 9 + [(["E22", "E5", "E11"], 3)]
    # Type 4: 8 de 10 fora da faixa, abaixo de 0,90.
    type4 = [(["E5"] * 6, 4)] * 8 + [(["E5", "E22", "E5"], 4)] * 2
    df = length_signal(_traces(normals + type3 + type4)).set_index("type")

    assert df.at[3, "normal_min_length"] == 3 and df.at[3, "normal_max_length"] == 5
    assert df.at[3, "out_of_range_fraction"] == pytest.approx(0.9)
    assert df.at[3, "has_length_signal"]
    assert df.at[4, "out_of_range_fraction"] == pytest.approx(0.8)
    assert not df.at[4, "has_length_signal"]


# --- Etapa 2 ----------------------------------------------------------------


def test_precedence_ignores_blocks_missing_one_event():
    # Normais: E23 sempre antes de E21 → P(E21 antes de E23) = 0, ordem fixa.
    normals = [(["E5", "E23", "E21"], None)] * 10
    # Type 1 preserva a ordem quando tem os dois eventos; nos demais blocos falta E21.
    type1 = [(["E5", "E23", "E21"], 1)] * 2 + [(["E5", "E23"], 1)] * 8
    df = precedence_test(_traces(normals + type1))

    row = _row(df, type=1, event_a="E21", event_b="E23")
    assert row["n_type_both"] == 2
    assert row["p_type"] == 0.0
    assert not row["is_violation"]


def test_precedence_detects_inversion_on_reference_pair():
    normals = [(["E5", "E23", "E21"], None)] * 10
    type1 = [(["E5", "E21", "E23"], 1)] * 6 + [(["E5", "E23", "E21"], 1)] * 2
    df = precedence_test(_traces(normals + type1))

    row = _row(df, type=1, event_a="E21", event_b="E23")
    assert row["is_reference"]
    assert row["p_normal"] == 0.0
    assert row["p_type"] == pytest.approx(0.75)
    assert row["is_violation"]


def test_precedence_requires_min_pair_support():
    # Inversão total, mas sobre um bloco a menos que o mínimo: fica na tabela, sem
    # marcar violação.
    normals = [(["E5", "E23", "E21"], None)] * 10
    few = [(["E5", "E21", "E23"], 1)] * (MIN_PAIR_SUPPORT - 1)
    enough = [(["E5", "E21", "E23"], 2)] * MIN_PAIR_SUPPORT
    df = precedence_test(_traces(normals + few + enough))

    row = _row(df, type=1, event_a="E21", event_b="E23")
    assert row["n_type_both"] == MIN_PAIR_SUPPORT - 1
    assert row["is_reference"] and row["divergence"] == pytest.approx(1.0)
    assert not row["has_min_support"]
    assert not row["is_violation"]
    assert _row(df, type=2, event_a="E21", event_b="E23")["is_violation"]


def test_precedence_uses_first_occurrence():
    # E22 aparece depois de E9 e também antes: conta a primeira ocorrência.
    normals = [(["E22", "E9", "E22"], None)] * 10
    df = precedence_test(_traces(normals + [(["E22", "E9"], 1)]))
    assert _row(df, type=1, event_a="E9", event_b="E22")["p_normal"] == 0.0


def test_precedence_threshold_boundaries():
    # Normais com P = 0,9 exatamente: entra na referência (inclusiva).
    normals = [(["E5", "E9"], None)] * 9 + [(["E9", "E5"], None)]
    # Type 1 com P = 0,65: divergência 0,25, conta como violação (inclusiva).
    type1 = [(["E5", "E9"], 1)] * 13 + [(["E9", "E5"], 1)] * 7
    # Type 2 com P = 0,7: divergência 0,2, não conta.
    type2 = [(["E5", "E9"], 2)] * 7 + [(["E9", "E5"], 2)] * 3
    df = precedence_test(_traces(normals + type1 + type2))

    t1 = _row(df, type=1, event_a="E5", event_b="E9")
    t2 = _row(df, type=2, event_a="E5", event_b="E9")
    assert t1["is_reference"] and t2["is_reference"]
    assert t1["divergence"] == pytest.approx(0.25)
    assert t1["is_violation"]
    assert not t2["is_violation"]


def test_precedence_unstable_normal_order_is_not_reference():
    normals = [(["E5", "E22"], None)] * 3 + [(["E22", "E5"], None)]
    type1 = [(["E22", "E5"], 1)] * 5
    row = _row(
        precedence_test(_traces(normals + type1)), type=1, event_a="E5", event_b="E22"
    )
    assert row["p_normal"] == pytest.approx(0.75)
    assert not row["is_reference"]
    assert not row["is_violation"]


def test_precedence_pair_absent_in_type_is_nan_and_not_violation():
    normals = [(["E5", "E9"], None)] * 5
    df = precedence_test(_traces(normals + [(["E5"], 1)]))
    row = _row(df, type=1, event_a="E5", event_b="E9")
    assert row["n_type_both"] == 0
    assert np.isnan(row["p_type"])
    assert not row["is_violation"]


# --- Etapa 3 ----------------------------------------------------------------

_NORMALS_E23_E21 = [(["E5", "E23", "E21"], None, [0, 10, 150])] * 10


def _check(blocks, max_gap=0, min_fraction=0.9) -> pd.DataFrame:
    return simultaneity_check(
        _traces(blocks), max_gap_seconds=max_gap, min_fraction=min_fraction
    )


def test_simultaneity_refuses_undefined_thresholds():
    traces = _traces(_NORMALS_E23_E21 + [(["E5", "E21", "E23"], 1)])
    with pytest.raises(ValueError, match="config.py"):
        simultaneity_check(traces, max_gap_seconds=None, min_fraction=0.9)
    with pytest.raises(ValueError, match="config.py"):
        simultaneity_check(traces, max_gap_seconds=0, min_fraction=None)


def test_zero_gap_inversion_is_artifact():
    type1 = [(["E5", "E21", "E23"], 1, [0, 7, 7])] * 5
    row = _row(_check(_NORMALS_E23_E21 + type1), type=1)
    assert row["n_inverted"] == 5
    assert row["simultaneous_fraction"] == 1.0
    assert row["is_artifact"]


def test_separated_inversion_is_genuine():
    type1 = [(["E5", "E21", "E23"], 1, [0, 10, 18_000])] * 5
    row = _row(_check(_NORMALS_E23_E21 + type1), type=1)
    assert row["median_gap_seconds"] == 17_990
    assert not row["is_artifact"]


def test_only_inverted_blocks_are_checked():
    # Os blocos do type na ordem normal têm intervalo nulo, mas não produzem a
    # violação; contá-los transformaria em artefato uma inversão bem separada no tempo.
    inverted = [(["E5", "E21", "E23"], 1, [0, 10, 500])] * 5
    in_order = [(["E5", "E23", "E21"], 1, [0, 3, 3])]
    row = _row(_check(_NORMALS_E23_E21 + inverted + in_order), type=1)
    assert row["n_inverted"] == 5
    assert row["n_simultaneous"] == 0
    assert not row["is_artifact"]


def test_gap_uses_first_occurrences():
    # Segunda E23 simultânea à E21, mas a primeira E23 está 100 s depois dela.
    type1 = [(["E5", "E21", "E23", "E23"], 1, [0, 5, 105, 105])] * 5
    row = _row(_check(_NORMALS_E23_E21 + type1), type=1)
    assert row["min_gap_seconds"] == 100


def test_simultaneity_thresholds_are_inclusive():
    # 9 de 10 blocos com intervalo de 1 s, um com 60 s.
    type1 = [(["E5", "E21", "E23"], 1, [0, 4, 5])] * 9 + [
        (["E5", "E21", "E23"], 1, [0, 4, 64])
    ]
    blocks = _NORMALS_E23_E21 + type1
    assert _row(_check(blocks, max_gap=1, min_fraction=0.9), type=1)["is_artifact"]
    assert not _row(_check(blocks, max_gap=0, min_fraction=0.9), type=1)["is_artifact"]
    assert not _row(_check(blocks, max_gap=1, min_fraction=0.95), type=1)["is_artifact"]


def test_simultaneity_handles_reference_with_low_p_normal():
    # Par listado como (E21, E23): nos normais P(E21 antes) = 0. Já em (E5, E9) o
    # normal é P = 1. Os dois sentidos da referência precisam achar os invertidos.
    normals = [(["E5", "E9", "E23", "E21"], None, [0, 1, 2, 3])] * 10
    type1 = [(["E9", "E5", "E21", "E23"], 1, [0, 0, 50, 90])] * 5
    df = _check(normals + type1)
    assert not _row(df, type=1, event_a="E21", event_b="E23")["is_artifact"]
    assert _row(df, type=1, event_a="E5", event_b="E9")["is_artifact"]


# --- Classificação final ----------------------------------------------------


def test_classify_types_four_outcomes():
    normals = [(["E5", "E23", "E21"], None, [0, 10, 150])] * 20
    # Type 1: evento distintivo (E7) E inversão E21/E23 bem separada no tempo.
    both = [(["E5", "E7", "E21", "E23"], 1, [0, 1, 10, 5000])] * 10
    # Type 2: só evento distintivo.
    non_seq = [(["E5", "E7", "E23", "E21"], 2)] * 10
    # Type 3: só inversão.
    seq = [(["E5", "E21", "E23"], 3, [0, 10, 5000])] * 10
    # Type 4: só inversão, mas com intervalo nulo — artefato, logo sem sinal algum.
    artifact = [(["E5", "E21", "E23"], 4, [0, 10, 10])] * 10
    # Type 5: dois sinais, mas abaixo de MIN_BLOCKS_PER_TYPE.
    small = [(["E5", "E7", "E21", "E23"], 5, [0, 1, 10, 5000])] * 9
    df = classify_types(
        _traces(normals + both + non_seq + seq + artifact + small)
    ).set_index("type")

    assert df.at[1, "classification"] == BOTH_SIGNALS
    assert df.at[1, "distinctive_events"] == ["E7"]
    assert df.at[1, "confirmed_violations"] == ["E21/E23"]
    assert df.at[2, "classification"] == NON_SEQUENTIAL
    assert df.at[3, "classification"] == SEQUENTIAL
    assert df.at[4, "n_artifacts"] == 1
    assert not df.at[4, "has_confirmed_violation"]
    assert df.at[4, "classification"] == UNDETERMINED
    assert df.at[5, "classification"] == UNDETERMINED


def test_length_signal_alone_is_non_sequential():
    normals = [(["E5", "E22", "E5"], None)] * 10
    df = classify_types(_traces(normals + [(["E5"], 3)] * 10)).set_index("type")
    assert df.at[3, "has_length_signal"] and not df.at[3, "has_distinctive_event"]
    assert df.at[3, "classification"] == NON_SEQUENTIAL


# --- Dados reais ------------------------------------------------------------


@pytest.mark.skipif(not EVENT_TRACES_PATH.exists(), reason="dados do HDFS_v1 ausentes em data/")
def test_real_type0_inverts_e21_e23():
    """Reproduz o resultado registrado (decisões, seção 5): inversão E21/E23 no Type 0."""
    df = precedence_test(load_event_traces())
    row = _row(df, type=0, event_a="E21", event_b="E23")
    assert row["is_reference"]
    assert row["p_normal"] < 0.01
    assert row["p_type"] == pytest.approx(0.87, abs=0.01)
    assert row["is_violation"]


@pytest.mark.skipif(not EVENT_TRACES_PATH.exists(), reason="dados do HDFS_v1 ausentes em data/")
def test_real_type0_inversion_is_not_simultaneous():
    """A inversão E21/E23 do Type 0 tem separação temporal: é reordenação genuína."""
    df = simultaneity_check(load_event_traces())
    row = _row(df, type=0, event_a="E21", event_b="E23")
    assert row["n_simultaneous"] == 0
    assert not row["is_artifact"]


@pytest.mark.skipif(not EVENT_TRACES_PATH.exists(), reason="dados do HDFS_v1 ausentes em data/")
def test_real_classification():
    """Type 13 sem violação (suporte), Type 3 pelo comprimento e a contagem final."""
    traces = load_event_traces()

    row = _row(precedence_test(traces), type=13, event_a="E21", event_b="E23")
    assert row["n_type_both"] < MIN_PAIR_SUPPORT
    assert row["divergence"] >= 0.25 and not row["is_violation"]

    df = classify_types(traces).set_index("type")
    assert df.at[3, "has_length_signal"] and not df.at[3, "has_distinctive_event"]
    assert df.at[3, "classification"] == NON_SEQUENTIAL
    assert df.at[13, "classification"] == NON_SEQUENTIAL
    assert set(df.index[df["classification"] == BOTH_SIGNALS]) == {0, 8}

    summary = df.groupby("classification")["n_blocks"].agg(["size", "sum"])
    assert summary.loc[NON_SEQUENTIAL].tolist() == [11, 13_931]
    assert summary.loc[BOTH_SIGNALS].tolist() == [2, 2_854]
    assert summary.loc[UNDETERMINED].tolist() == [13, 53]
    assert SEQUENTIAL not in summary.index
