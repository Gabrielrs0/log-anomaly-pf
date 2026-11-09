"""Classificação sequencial × não-sequencial dos tipos de anomalia (decisões, seção 5).

- Etapa 1, sinais NÃO-sequenciais, detectáveis por contagem, sem ordem:
  ``distinctive_events`` (eventos cuja PRESENÇA distingue um ``type`` dos normais) e
  ``length_signal`` (COMPRIMENTO de sequência fora da faixa dos normais).
- Etapa 2, ``precedence_test``: pares de eventos-núcleo cuja ORDEM é quase fixa nos
  normais e diverge na categoria. Sinal de anomalia sequencial.
- Etapa 3, ``simultaneity_check``: confronta cada violação da etapa 2 com os carimbos
  de tempo do par. Eventos simultâneos do HDFS são registrados em ordem aleatória
  (Landauer et al., 2024); uma inversão entre eventos simultâneos é artefato de
  registro, não anomalia sequencial.

- ``classify_types`` combina os sinais num rótulo por tipo, sem tratá-los como
  excludentes.

As três etapas leem ``EventTraces``, a única representação com ordem e com carimbos de
tempo (decisões, seção 3). A etapa 1 usa apenas presença e comprimento e ignora a
ordem.

As tabelas devolvidas trazem todas as medidas intermediárias, e não só as linhas que
passam nos limiares, para que a monografia possa mostrar quão perto do corte cada
caso ficou. Todos os tipos são incluídos, com ``n_blocks``; descartar os tipos com
menos de ``MIN_BLOCKS_PER_TYPE`` blocos cabe a ``classify_types``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.config import (
    CORE_EVENTS,
    DISTINCTIVE_MAX_NORMAL_FREQ,
    DISTINCTIVE_MIN_TYPE_FREQ,
    EVENT_IDS,
    LENGTH_SIGNAL_MIN_FRACTION,
    MIN_BLOCKS_PER_TYPE,
    MIN_PAIR_SUPPORT,
    PRECEDENCE_FIXED_ORDER_HIGH,
    PRECEDENCE_FIXED_ORDER_LOW,
    PRECEDENCE_MIN_DIVERGENCE,
    SIMULTANEITY_MAX_GAP_SECONDS,
    SIMULTANEITY_MIN_FRACTION,
)
from src.data.loader import EventTraces

# Sentinela de "evento ausente no bloco" na matriz de primeiras posições.
_ABSENT = -1


# --- Etapa 1 ----------------------------------------------------------------


def distinctive_events(traces: EventTraces) -> pd.DataFrame:
    """Frequência de presença de cada evento por ``type`` e nos normais.

    Um evento é distintivo de um ``type`` quando aparece em mais de
    ``DISTINCTIVE_MIN_TYPE_FREQ`` dos blocos da categoria e em menos de
    ``DISTINCTIVE_MAX_NORMAL_FREQ`` dos normais (desigualdades estritas).

    A frequência é de PRESENÇA (fração de blocos em que o evento aparece ao menos uma
    vez), não de contagem: o critério pergunta se o evento ocorre, e um bloco com o
    evento repetido não é "mais anômalo" para esse fim.

    Returns:
        Uma linha por par (``type``, evento), ordenada por ``type`` e pela ordem
        numérica dos eventos, com as colunas ``type``, ``event``, ``n_blocks``,
        ``type_freq``, ``normal_freq`` e ``is_distinctive``.
    """
    presence = _presence_matrix(traces)
    labels = traces.blocks["label"].to_numpy()
    types = traces.blocks["type"]

    normal = labels == 0
    if not normal.any():
        raise ValueError("Sem blocos normais: impossível estimar a frequência de referência.")
    normal_freq = presence[normal].mean(axis=0)

    rows = []
    for type_, mask in _type_masks(labels, types):
        type_freq = presence[mask].mean(axis=0)
        for j, event in enumerate(EVENT_IDS):
            rows.append(
                {
                    "type": type_,
                    "event": event,
                    "n_blocks": int(mask.sum()),
                    "type_freq": float(type_freq[j]),
                    "normal_freq": float(normal_freq[j]),
                }
            )

    df = pd.DataFrame(rows)
    df["is_distinctive"] = (df["type_freq"] > DISTINCTIVE_MIN_TYPE_FREQ) & (
        df["normal_freq"] < DISTINCTIVE_MAX_NORMAL_FREQ
    )
    return df


def length_signal(traces: EventTraces) -> pd.DataFrame:
    """Fração de blocos de cada ``type`` com comprimento fora da faixa dos normais.

    Landauer et al. (2024) apontam três manifestações de anomalia no HDFS: tipos de
    evento novos, comprimento de sequência e contagens. ``distinctive_events`` cobre só
    a primeira; sem esta medida, um tipo de truncamento puro — como o Type 3, cujos
    blocos têm 2 eventos, contra mínimo de 13 nos normais — fica sem sinal algum.
    Comprimento é a soma das contagens, então é sinal NÃO-sequencial.

    A faixa é ``[min, max]`` do comprimento nos blocos normais, inclusiva. O tipo é
    marcado quando a fração fora da faixa é ``>= LENGTH_SIGNAL_MIN_FRACTION``.

    Returns:
        Uma linha por ``type``, com as colunas ``type``, ``n_blocks``,
        ``normal_min_length``, ``normal_max_length``, ``out_of_range_fraction`` e
        ``has_length_signal``.
    """
    lengths = np.fromiter((len(e) for e in traces.events), dtype=np.int64)
    labels = traces.blocks["label"].to_numpy()

    normal = labels == 0
    if not normal.any():
        raise ValueError("Sem blocos normais: impossível estimar a faixa de comprimento.")
    low, high = int(lengths[normal].min()), int(lengths[normal].max())
    outside = (lengths < low) | (lengths > high)

    rows = [
        {
            "type": type_,
            "n_blocks": int(mask.sum()),
            "normal_min_length": low,
            "normal_max_length": high,
            "out_of_range_fraction": float(outside[mask].mean()),
        }
        for type_, mask in _type_masks(labels, traces.blocks["type"])
    ]
    df = pd.DataFrame(rows)
    df["has_length_signal"] = df["out_of_range_fraction"] >= LENGTH_SIGNAL_MIN_FRACTION
    return df


# --- Etapa 2 ----------------------------------------------------------------


def precedence_test(traces: EventTraces) -> pd.DataFrame:
    """Teste de precedência entre os pares de ``CORE_EVENTS``.

    Para cada par (A, B), P(A antes de B) é a fração de blocos em que a PRIMEIRA
    ocorrência de A vem antes da primeira ocorrência de B. Os eventos-núcleo se repetem
    num mesmo bloco (um por réplica), e a primeira ocorrência marca quando aquela fase
    do ciclo de vida do bloco começou.

    A probabilidade é estimada só sobre blocos em que AMBOS os eventos aparecem
    (decisões, seção 5, etapa 2). Num bloco sem B, "A antes de B" não é nem verdadeiro
    nem falso; contá-lo de qualquer lado faria a ausência de um evento — sinal
    não-sequencial, já capturado pela etapa 1 — se passar por violação de ordem.

    Um par entra na referência quando, nos normais, P >= ``PRECEDENCE_FIXED_ORDER_HIGH``
    ou P <= ``PRECEDENCE_FIXED_ORDER_LOW``. Num par de referência, divergência
    ``|P_tipo - P_normal| >= PRECEDENCE_MIN_DIVERGENCE`` conta como violação, desde
    que estimada sobre pelo menos ``MIN_PAIR_SUPPORT`` blocos do tipo com ambos os
    eventos. Abaixo disso a divergência é indistinguível de coincidência; a linha fica
    na tabela, com ``has_min_support = False``, para que o descarte seja auditável.

    Returns:
        Uma linha por par (``type``, (A, B)), com A antes de B em ``CORE_EVENTS``, e as
        colunas ``type``, ``event_a``, ``event_b``, ``n_blocks`` (blocos do tipo),
        ``n_normal_both`` e ``n_type_both`` (blocos com ambos os eventos),
        ``p_normal``, ``p_type``, ``is_reference``, ``divergence``,
        ``has_min_support`` e ``is_violation``. Probabilidades ficam ``NaN`` quando nenhum bloco do grupo
        contém o par; um par assim nunca é referência nem violação.
    """
    first = _first_positions(traces, CORE_EVENTS)
    labels = traces.blocks["label"].to_numpy()
    types = traces.blocks["type"]

    normal = labels == 0
    if not normal.any():
        raise ValueError("Sem blocos normais: impossível estabelecer a ordem de referência.")

    rows = []
    for (ia, a), (ib, b) in combinations(enumerate(CORE_EVENTS), 2):
        both = (first[:, ia] != _ABSENT) & (first[:, ib] != _ABSENT)
        a_first = first[:, ia] < first[:, ib]
        n_normal_both, p_normal = _precedence(a_first, both & normal)
        for type_, mask in _type_masks(labels, types):
            n_type_both, p_type = _precedence(a_first, both & mask)
            rows.append(
                {
                    "type": type_,
                    "event_a": a,
                    "event_b": b,
                    "n_blocks": int(mask.sum()),
                    "n_normal_both": n_normal_both,
                    "n_type_both": n_type_both,
                    "p_normal": p_normal,
                    "p_type": p_type,
                }
            )

    df = pd.DataFrame(rows)
    # Comparações com NaN dão False: par sem dados nunca vira referência nem violação.
    df["is_reference"] = (df["p_normal"] >= PRECEDENCE_FIXED_ORDER_HIGH) | (
        df["p_normal"] <= PRECEDENCE_FIXED_ORDER_LOW
    )
    df["divergence"] = (df["p_type"] - df["p_normal"]).abs()
    df["has_min_support"] = df["n_type_both"] >= MIN_PAIR_SUPPORT
    df["is_violation"] = (
        df["is_reference"]
        & (df["divergence"] >= PRECEDENCE_MIN_DIVERGENCE)
        & df["has_min_support"]
    )
    return df


# --- Etapa 3 ----------------------------------------------------------------


def simultaneity_check(
    traces: EventTraces,
    precedence: pd.DataFrame | None = None,
    max_gap_seconds: float | None = SIMULTANEITY_MAX_GAP_SECONDS,
    min_fraction: float | None = SIMULTANEITY_MIN_FRACTION,
) -> pd.DataFrame:
    """Verificação de simultaneidade das violações de ordem (decisões, seção 5, etapa 3).

    Para cada violação da etapa 2, toma os blocos do ``type`` que contêm ambos os
    eventos na ordem INVERTIDA em relação aos normais — os blocos que de fato produzem
    a divergência — e mede o intervalo ``|t(B) - t(A)|`` entre as primeiras ocorrências
    do par, mesma definição de ordem usada na etapa 2. Os carimbos vêm do campo
    ``TimeInterval`` de ``Event_traces.csv``, acumulados pelo loader.

    Se a fração de blocos invertidos com intervalo ``<= max_gap_seconds`` for
    ``>= min_fraction``, o intervalo é consistentemente nulo e a violação é
    reclassificada como artefato de registro. Caso contrário, há separação temporal e
    a reordenação é genuína.

    Args:
        traces: sequências com carimbos de tempo.
        precedence: saída de ``precedence_test``; ``None`` a recalcula.
        max_gap_seconds: maior intervalo ainda tratado como simultâneo.
        min_fraction: fração mínima de blocos simultâneos para caracterizar artefato.

    Raises:
        ValueError: se algum limiar for ``None``. Os limiares são decisão metodológica
            (decisões, seção 5); sem eles registrados em ``config.py`` a etapa não roda,
            em vez de assumir um padrão silencioso.

    Returns:
        Uma linha por violação, com as colunas ``type``, ``event_a``, ``event_b``,
        ``n_inverted`` (blocos com a ordem invertida), ``n_simultaneous``,
        ``simultaneous_fraction``, ``median_gap_seconds``, ``min_gap_seconds``,
        ``max_gap_seconds`` e ``is_artifact``.
    """
    if max_gap_seconds is None or min_fraction is None:
        raise ValueError(
            "Limiares de simultaneidade não definidos: registre "
            "SIMULTANEITY_MAX_GAP_SECONDS e SIMULTANEITY_MIN_FRACTION em config.py "
            "(decisões, seção 5, etapa 3)."
        )
    if precedence is None:
        precedence = precedence_test(traces)

    first = _first_positions(traces, CORE_EVENTS)
    first_ts = _timestamps_at(traces, first)
    labels = traces.blocks["label"].to_numpy()
    masks = dict(_type_masks(labels, traces.blocks["type"]))

    rows = []
    for row in precedence[precedence["is_violation"]].itertuples(index=False):
        ia, ib = CORE_EVENTS.index(row.event_a), CORE_EVENTS.index(row.event_b)
        both = (first[:, ia] != _ABSENT) & (first[:, ib] != _ABSENT)
        a_first = first[:, ia] < first[:, ib]
        # Violação só existe em par de referência, logo P_normal está num dos extremos
        # e a ordem normal é a da maioria.
        normal_a_first = row.p_normal >= PRECEDENCE_FIXED_ORDER_HIGH
        inverted = masks[row.type] & both & (a_first != normal_a_first)
        gaps = np.abs(first_ts[inverted, ib] - first_ts[inverted, ia])

        n_simultaneous = int((gaps <= max_gap_seconds).sum())
        fraction = n_simultaneous / len(gaps)
        rows.append(
            {
                "type": row.type,
                "event_a": row.event_a,
                "event_b": row.event_b,
                "n_inverted": len(gaps),
                "n_simultaneous": n_simultaneous,
                "simultaneous_fraction": fraction,
                "median_gap_seconds": float(np.median(gaps)),
                "min_gap_seconds": int(gaps.min()),
                "max_gap_seconds": int(gaps.max()),
                "is_artifact": fraction >= min_fraction,
            }
        )

    columns = [
        "type", "event_a", "event_b", "n_inverted", "n_simultaneous",
        "simultaneous_fraction", "median_gap_seconds", "min_gap_seconds",
        "max_gap_seconds", "is_artifact",
    ]  # fmt: skip
    # Sem violações a tabela sai vazia com dtype object; fixar bool evita que
    # ``~df["is_artifact"]`` deixe de ser máscara e vire seleção de colunas.
    return pd.DataFrame(rows, columns=columns).astype({"is_artifact": bool})


# --- Classificação final ----------------------------------------------------

NON_SEQUENTIAL = "nao-sequencial"
BOTH_SIGNALS = "ambos os sinais"
SEQUENTIAL = "sequencial"
UNDETERMINED = "indeterminado"


def classify_types(traces: EventTraces) -> pd.DataFrame:
    """Rótulo de cada ``type`` a partir dos sinais das três etapas.

    Os sinais não são excludentes: os Types 0 e 8 têm evento distintivo E violação de
    ordem confirmada. Uma regra de precedência de um sinal sobre o outro esvaziaria uma
    das categorias, então a classificação tem quatro saídas:

    - ``"nao-sequencial"``: sinal não-sequencial (evento distintivo ou comprimento) e
      nenhuma violação confirmada;
    - ``"ambos os sinais"``: sinal não-sequencial e violação confirmada;
    - ``"sequencial"``: violação confirmada sem sinal não-sequencial;
    - ``"indeterminado"``: nenhum sinal, ou menos de ``MIN_BLOCKS_PER_TYPE`` blocos.

    Violação confirmada é a da etapa 2 que a etapa 3 não reclassificou como artefato.

    Returns:
        Uma linha por ``type``, com as colunas ``type``, ``n_blocks``,
        ``distinctive_events`` (lista), ``has_distinctive_event``,
        ``has_length_signal``, ``out_of_range_fraction``, ``n_violations`` (etapa 2),
        ``n_artifacts`` (etapa 3), ``confirmed_violations`` (lista de pares "A/B"),
        ``has_confirmed_violation`` e ``classification``.
    """
    distinctive = distinctive_events(traces)
    length = length_signal(traces).set_index("type")
    precedence = precedence_test(traces)
    simultaneity = simultaneity_check(traces, precedence)

    rows = []
    for type_, n_blocks in length["n_blocks"].items():
        events = distinctive.loc[
            (distinctive["type"] == type_) & distinctive["is_distinctive"], "event"
        ].tolist()
        checked = simultaneity[simultaneity["type"] == type_]
        confirmed = checked[~checked["is_artifact"]]
        rows.append(
            {
                "type": type_,
                "n_blocks": n_blocks,
                "distinctive_events": events,
                "has_distinctive_event": bool(events),
                "has_length_signal": bool(length.at[type_, "has_length_signal"]),
                "out_of_range_fraction": length.at[type_, "out_of_range_fraction"],
                "n_violations": len(checked),
                "n_artifacts": int(checked["is_artifact"].sum()),
                "confirmed_violations": [
                    f"{a}/{b}" for a, b in zip(confirmed["event_a"], confirmed["event_b"])
                ],
                "has_confirmed_violation": not confirmed.empty,
            }
        )

    df = pd.DataFrame(rows)
    df["classification"] = [
        _label(r.n_blocks, r.has_distinctive_event or r.has_length_signal,
               r.has_confirmed_violation)
        for r in df.itertuples(index=False)
    ]  # fmt: skip
    return df


def _label(n_blocks: int, non_sequential: bool, sequential: bool) -> str:
    if n_blocks < MIN_BLOCKS_PER_TYPE:
        return UNDETERMINED
    if non_sequential and sequential:
        return BOTH_SIGNALS
    if non_sequential:
        return NON_SEQUENTIAL
    if sequential:
        return SEQUENTIAL
    return UNDETERMINED


# --- Auxiliares -------------------------------------------------------------


def _type_masks(labels: np.ndarray, types: pd.Series):
    """Gera (``type``, máscara booleana) para cada tipo de anomalia, em ordem crescente."""
    anomaly = labels == 1
    type_values = types.to_numpy(dtype="float64", na_value=np.nan)
    for type_ in sorted(types[anomaly].dropna().unique()):
        yield int(type_), anomaly & (type_values == type_)


def _flatten(traces: EventTraces) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatena as sequências: (eventos, bloco de cada evento, posição no bloco)."""
    lengths = np.fromiter((len(e) for e in traces.events), dtype=np.int64)
    flat = np.concatenate(traces.events) if len(lengths) else np.empty(0, np.int8)
    block = np.repeat(np.arange(len(lengths)), lengths)
    starts = np.cumsum(lengths) - lengths
    position = np.arange(len(flat)) - np.repeat(starts, lengths)
    return flat, block, position


def _presence_matrix(traces: EventTraces) -> np.ndarray:
    """Matriz booleana ``(n_blocos, 29)``: o evento ``EVENT_IDS[j]`` ocorre no bloco."""
    flat, block, _ = _flatten(traces)
    presence = np.zeros((len(traces.events), len(EVENT_IDS)), dtype=bool)
    # O loader codifica E1 → 1, ..., E29 → 29; a coluna é o código menos um.
    presence[block, flat.astype(np.int64) - 1] = True
    return presence


def _first_positions(traces: EventTraces, events: tuple[str, ...]) -> np.ndarray:
    """Posição da primeira ocorrência de cada evento em cada bloco; ``_ABSENT`` se não há."""
    flat, block, position = _flatten(traces)
    first = np.full((len(traces.events), len(events)), _ABSENT, dtype=np.int64)
    for k, event in enumerate(events):
        hits = np.flatnonzero(flat == EVENT_IDS.index(event) + 1)
        # ``flat`` está ordenado por bloco e, dentro do bloco, por posição; o primeiro
        # acerto de cada bloco é, portanto, a primeira ocorrência.
        blocks_hit, first_hit = np.unique(block[hits], return_index=True)
        first[blocks_hit, k] = position[hits[first_hit]]
    return first


def _timestamps_at(traces: EventTraces, positions: np.ndarray) -> np.ndarray:
    """Carimbo, em segundos, do evento em cada posição; ``_ABSENT`` onde não há evento."""
    lengths = np.fromiter((len(t) for t in traces.timestamps), dtype=np.int64)
    flat_ts = np.concatenate(traces.timestamps) if len(lengths) else np.empty(0, np.int64)
    starts = np.cumsum(lengths) - lengths
    present = positions != _ABSENT
    out = np.full(positions.shape, _ABSENT, dtype=np.int64)
    out[present] = flat_ts[(starts[:, None] + positions)[present]]
    return out


def _precedence(a_first: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    """(n, P(A antes de B)) sobre os blocos de ``mask``; P = NaN se n = 0."""
    n = int(mask.sum())
    return n, (float(a_first[mask].mean()) if n else float("nan"))
