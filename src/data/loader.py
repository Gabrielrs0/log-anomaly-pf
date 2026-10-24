"""Carregamento das duas representações do HDFS_v1.

Única porta de entrada dos arquivos de dados: nenhum outro módulo lê CSV diretamente.
Concentrar a leitura aqui garante que toda a análise parte das mesmas verificações de
integridade e da mesma ordem de linhas.

As duas representações ficam em estruturas separadas de propósito (decisões, seção 3):
``OccurrenceMatrix`` não carrega informação de ordem e ``EventTraces`` não carrega a
matriz de contagem. Misturá-las destruiria o contraste que o trabalho investiga.

Convenções comuns às duas estruturas:

- As linhas seguem a ordem de ``anomaly_label.csv``; a linha ``i`` de uma representação
  e a linha ``i`` da outra referem-se ao mesmo bloco.
- ``blocks`` é um DataFrame indexado por ``BlockId`` com as colunas ``label``
  (1 = anomalia, 0 = normal, a mesma convenção de ``predict`` dos modelos) e ``type``
  (subcategoria de falha; ``<NA>`` nos blocos normais).
- Eventos são codificados pelo número do template: E1 → 1, ..., E29 → 29. O código 0
  fica livre para preenchimento (padding) no LSTM.

Este módulo descreve os dados; não decide nada sobre eles. Classificar tipos, dividir
treino/teste ou ponderar contagens pertence às camadas acima.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ANOMALY_LABEL_PATH,
    EVENT_IDS,
    EVENT_OCCURRENCE_MATRIX_PATH,
    EVENT_TRACES_PATH,
    N_ANOMALOUS_BLOCKS,
    N_BLOCKS,
)

# --- Formato dos arquivos ---------------------------------------------------
# Conhecimento sobre o formato do Loghub, não parâmetro do experimento; por isso fica
# aqui e não em config.py.

_BLOCK_ID_COL = "BlockId"
_LABEL_COL = "Label"
_TYPE_COL = "Type"
_FEATURES_COL = "Features"
_TIME_INTERVAL_COL = "TimeInterval"
_LATENCY_COL = "Latency"

# anomaly_label.csv usa Normal/Anomaly; Event_traces e Event_occurrence_matrix usam
# Success/Fail para o mesmo rótulo.
_LABEL_FILE_VALUES = {"Normal": 0, "Anomaly": 1}
_INLINE_LABEL_VALUES = {"Success": 0, "Fail": 1}

_FEATURES_PATTERN = re.compile(r"\[E\d+(?:,E\d+)*\]")

_EVENT_CODES = {event_id: code for code, event_id in enumerate(EVENT_IDS, start=1)}

# Contagens publicadas; passar None desliga a conferência (útil em arquivos sintéticos).
_PUBLISHED_COUNTS = (N_BLOCKS, N_ANOMALOUS_BLOCKS)


# --- Estruturas devolvidas --------------------------------------------------


@dataclass(frozen=True, slots=True)
class OccurrenceMatrix:
    """Representação bag-of-events: contagem de cada template por bloco, sem ordem.

    Atributos:
        counts: matriz ``(n_blocos, 29)`` de inteiros; coluna ``j`` = ``event_ids[j]``.
        event_ids: nomes das colunas de ``counts``, em ordem numérica (E1..E29).
        blocks: rótulo e tipo por bloco (ver docstring do módulo).
    """

    counts: np.ndarray
    event_ids: tuple[str, ...]
    blocks: pd.DataFrame


@dataclass(frozen=True, slots=True)
class EventTraces:
    """Representação sequencial: eventos de cada bloco na ordem em que foram registrados.

    Atributos:
        events: um array ``int8`` por bloco com os códigos dos eventos (E5 → 5).
        timestamps: um array ``int64`` por bloco, alinhado a ``events``, com o instante
            de cada evento em segundos desde o primeiro evento do bloco. É a informação
            exigida pela verificação de simultaneidade (decisões, seção 5, etapa 3).
        blocks: rótulo e tipo por bloco (ver docstring do módulo).

    A resolução dos carimbos é de 1 s, a do HDFS.log bruto. Dois eventos com o mesmo
    carimbo têm ordem relativa NÃO confiável: Landauer et al. (2024) mostram que eventos
    simultâneos do HDFS são registrados em ordem aleatória.
    """

    events: list[np.ndarray]
    timestamps: list[np.ndarray]
    blocks: pd.DataFrame


# --- API pública ------------------------------------------------------------


def load_occurrence_matrix(
    path: Path = EVENT_OCCURRENCE_MATRIX_PATH,
    label_path: Path = ANOMALY_LABEL_PATH,
    expected_counts: tuple[int, int] | None = _PUBLISHED_COUNTS,
) -> OccurrenceMatrix:
    """Carrega ``Event_occurrence_matrix.csv`` como matriz de contagem.

    Args:
        path: caminho da matriz de ocorrência.
        label_path: caminho de ``anomaly_label.csv``, fonte canônica dos rótulos.
        expected_counts: ``(n_blocos, n_anomalias)`` esperados; ``None`` não confere.

    Raises:
        ValueError: se o arquivo divergir do formato ou das contagens esperadas.
    """
    df = pd.read_csv(path)

    missing = [e for e in EVENT_IDS if e not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: colunas de evento ausentes: {missing}")
    extra = [c for c in df.columns if re.fullmatch(r"E\d+", c) and c not in EVENT_IDS]
    if extra:
        raise ValueError(f"{path.name}: eventos fora de E1..E29: {extra}")

    # Seleção explícita fixa a ordem E1..E29, independentemente da ordem no CSV.
    counts = df.loc[:, list(EVENT_IDS)].to_numpy()
    if not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any():
        raise ValueError(f"{path.name}: contagens devem ser inteiros não negativos.")

    blocks = _build_blocks(df, path, label_path, expected_counts)
    return OccurrenceMatrix(
        counts=counts.astype(np.int32), event_ids=EVENT_IDS, blocks=blocks
    )


def load_event_traces(
    path: Path = EVENT_TRACES_PATH,
    label_path: Path = ANOMALY_LABEL_PATH,
    expected_counts: tuple[int, int] | None = _PUBLISHED_COUNTS,
) -> EventTraces:
    """Carrega ``Event_traces.csv`` como sequências ordenadas com carimbos de tempo.

    Args:
        path: caminho de ``Event_traces.csv``.
        label_path: caminho de ``anomaly_label.csv``, fonte canônica dos rótulos.
        expected_counts: ``(n_blocos, n_anomalias)`` esperados; ``None`` não confere.

    Raises:
        ValueError: se o arquivo divergir do formato ou das contagens esperadas.
    """
    df = pd.read_csv(path)

    features = df[_FEATURES_COL]
    malformed = ~features.str.fullmatch(_FEATURES_PATTERN)
    if malformed.any():
        raise ValueError(
            f"{path.name}: campo {_FEATURES_COL} malformado em "
            f"{int(malformed.sum())} blocos (ex.: {df.loc[malformed, _BLOCK_ID_COL].iloc[0]})."
        )

    # Parse vetorizado: concatenar todos os blocos e separar uma única vez é ordens de
    # grandeza mais rápido que interpretar 575 mil listas uma a uma.
    lengths = features.str.count(",").to_numpy() + 1
    tokens = ",".join(features.str.slice(1, -1)).split(",")
    unknown = set(tokens) - _EVENT_CODES.keys()
    if unknown:
        raise ValueError(f"{path.name}: eventos fora de E1..E29: {sorted(unknown)}")
    flat_events = np.fromiter(
        (_EVENT_CODES[t] for t in tokens), dtype=np.int8, count=len(tokens)
    )

    # TimeInterval[k] é o intervalo, em segundos, entre o evento k e o evento k+1 do
    # bloco — por isso tem um elemento a menos que Features. Conferido contra os
    # carimbos do HDFS.log bruto; Latency é a soma dos intervalos.
    intervals_str = df[_TIME_INTERVAL_COL].str.slice(1, -1)
    n_intervals = np.where(intervals_str == "", 0, intervals_str.str.count(",") + 1)
    if not np.array_equal(n_intervals, lengths - 1):
        bad = df.loc[n_intervals != lengths - 1, _BLOCK_ID_COL].iloc[0]
        raise ValueError(
            f"{path.name}: {_TIME_INTERVAL_COL} deve ter um elemento a menos que "
            f"{_FEATURES_COL} (ex.: {bad})."
        )
    joined = ",".join(s for s in intervals_str if s)
    flat_intervals = (
        np.array(joined.split(","), dtype=np.float64) if joined else np.empty(0)
    )
    if (flat_intervals < 0).any() or not np.all(
        flat_intervals == np.round(flat_intervals)
    ):
        raise ValueError(
            f"{path.name}: {_TIME_INTERVAL_COL} deve conter segundos inteiros não negativos."
        )
    flat_intervals = flat_intervals.astype(np.int64)

    # Carimbos por bloco sem laço em Python: põe 0 na posição do primeiro evento de cada
    # bloco, acumula globalmente e subtrai o acumulado no início do próprio bloco.
    ends = np.cumsum(lengths)
    starts = ends - lengths
    steps = np.zeros(len(flat_events), dtype=np.int64)
    steps[np.setdiff1d(np.arange(len(flat_events)), starts)] = flat_intervals
    running = np.cumsum(steps)
    flat_timestamps = running - np.repeat(running[starts], lengths)

    events = np.split(flat_events, ends[:-1])
    timestamps = np.split(flat_timestamps, ends[:-1])

    latency = df[_LATENCY_COL].to_numpy()
    last_ts = flat_timestamps[ends - 1]
    if not np.array_equal(last_ts, latency):
        bad = df.loc[last_ts != latency, _BLOCK_ID_COL].iloc[0]
        raise ValueError(
            f"{path.name}: soma de {_TIME_INTERVAL_COL} difere de {_LATENCY_COL} (ex.: {bad})."
        )

    blocks = _build_blocks(df, path, label_path, expected_counts)
    return EventTraces(events=events, timestamps=timestamps, blocks=blocks)


# --- Auxiliares -------------------------------------------------------------


def _build_blocks(
    df: pd.DataFrame,
    path: Path,
    label_path: Path,
    expected_counts: tuple[int, int] | None,
) -> pd.DataFrame:
    """Monta o DataFrame ``blocks`` e confere a coerência entre rótulos e tipos.

    ``anomaly_label.csv`` é a fonte canônica do rótulo. O rótulo embutido em cada
    arquivo (Success/Fail) serve só de conferência: divergência indica arquivos de
    versões diferentes, e seguir adiante misturaria duas rotulagens sem aviso.
    """
    block_ids = df[_BLOCK_ID_COL]
    if block_ids.duplicated().any():
        raise ValueError(f"{path.name}: BlockId duplicado.")

    labels = _load_labels(label_path)
    if not np.array_equal(block_ids.to_numpy(), labels.index.to_numpy()):
        # Exigir a mesma ordem, em vez de reordenar, mantém a linha i alinhada entre as
        # duas representações sem que as camadas de cima precisem se preocupar.
        raise ValueError(f"{path.name}: blocos ou ordem diferem de {label_path.name}.")

    inline = df[_LABEL_COL].map(_INLINE_LABEL_VALUES)
    if inline.isna().any():
        raise ValueError(f"{path.name}: valores de {_LABEL_COL} fora de Success/Fail.")
    if not np.array_equal(inline.to_numpy(), labels.to_numpy()):
        raise ValueError(f"{path.name}: rótulos divergem de {label_path.name}.")

    # O CSV traz Type como float (21.0) porque os normais têm célula vazia; Int64
    # preserva o inteiro e representa a ausência como <NA>.
    raw_type = df[_TYPE_COL]
    present = raw_type.notna()
    if not np.all(raw_type[present] == np.round(raw_type[present])):
        raise ValueError(f"{path.name}: {_TYPE_COL} contém valor não inteiro.")
    types = raw_type.astype("Int64")
    # Type só existe para anomalias (decisões, seção 2); um normal com tipo, ou uma
    # anomalia sem tipo, comprometeria a avaliação segmentada (seção 4.7).
    if not np.array_equal(present.to_numpy(), labels.to_numpy() == 1):
        raise ValueError(
            f"{path.name}: {_TYPE_COL} deve existir só nos blocos anômalos."
        )

    if expected_counts is not None:
        n_blocks, n_anomalies = expected_counts
        got = (len(labels), int(labels.sum()))
        if got != (n_blocks, n_anomalies):
            raise ValueError(
                f"{path.name}: esperados {n_blocks} blocos e {n_anomalies} anomalias, "
                f"encontrados {got[0]} e {got[1]}."
            )

    return pd.DataFrame(
        {"label": labels.to_numpy(), "type": types.array},
        index=pd.Index(block_ids.to_numpy(), name=_BLOCK_ID_COL),
    )


def _load_labels(path: Path) -> pd.Series:
    """Lê ``anomaly_label.csv`` como série 0/1 indexada por ``BlockId``."""
    df = pd.read_csv(path)
    labels = df[_LABEL_COL].map(_LABEL_FILE_VALUES)
    if labels.isna().any():
        raise ValueError(
            f"{path.name}: valores de {_LABEL_COL} fora de Normal/Anomaly."
        )
    return pd.Series(
        labels.to_numpy(dtype=np.int8),
        index=pd.Index(df[_BLOCK_ID_COL].to_numpy(), name=_BLOCK_ID_COL),
    )