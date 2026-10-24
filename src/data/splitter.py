"""Divisão treino/validação/teste (decisões, seção 4.1).

Os três métodos são não supervisionados e aprendem o padrão de normalidade; uma
anomalia no treino contamina exatamente o que precisam aprender. Por isso:

- normais: 60% treino, 20% validação, 20% teste;
- anomalias: 50% validação, 50% teste — nenhuma no treino.

A divisão devolve índices posicionais. Como o loader garante que a linha ``i`` é o
mesmo bloco nas duas representações, os mesmos índices servem para a matriz de
ocorrência e para as sequências — os três modelos são avaliados nos mesmos blocos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    ANOMALY_TEST_FRACTION,
    ANOMALY_TRAIN_FRACTION,
    ANOMALY_VALIDATION_FRACTION,
    NORMAL_TEST_FRACTION,
    NORMAL_TRAIN_FRACTION,
    NORMAL_VALIDATION_FRACTION,
    RANDOM_SEED,
)

_NORMAL_FRACTIONS = (
    NORMAL_TRAIN_FRACTION,
    NORMAL_VALIDATION_FRACTION,
    NORMAL_TEST_FRACTION,
)
_ANOMALY_FRACTIONS = (
    ANOMALY_TRAIN_FRACTION,
    ANOMALY_VALIDATION_FRACTION,
    ANOMALY_TEST_FRACTION,
)


@dataclass(frozen=True, slots=True)
class Split:
    """Índices posicionais (ordenados) de cada partição."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def split_blocks(blocks: pd.DataFrame, seed: int = RANDOM_SEED) -> Split:
    """Divide os blocos em treino, validação e teste.

    As anomalias são divididas separadamente dentro de cada ``type``. Sem isso, uma
    categoria pequena pode cair inteira na validação ou inteira no teste, e a avaliação
    segmentada (seção 4.7) ficaria sem positivos daquela categoria num dos lados — ou
    com um limiar ajustado na validação sobre uma mistura de categorias diferente da do
    teste. Os normais não têm subcategoria e são divididos como um único grupo.

    Args:
        blocks: o DataFrame ``blocks`` devolvido pelo loader (colunas ``label`` e
            ``type``).
        seed: semente da amostragem. Execuções repetidas (seção 4.8) variam este valor;
            o padrão é ``RANDOM_SEED``.

    Raises:
        ValueError: se houver anomalia sem ``type`` ou se, por qualquer motivo, uma
            anomalia acabar no treino.
    """
    labels = blocks["label"].to_numpy()
    types = blocks["type"]
    is_anomaly = labels == 1
    if types[is_anomaly].isna().any():
        raise ValueError("Anomalia sem type: impossível estratificar a divisão.")

    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[], [], []]

    groups = [np.flatnonzero(~is_anomaly)]
    # Ordem fixa dos grupos: o rng é consumido sempre na mesma sequência, o que mantém
    # a divisão reproduzível para uma dada semente.
    anomaly_types = types[is_anomaly]
    for type_ in sorted(anomaly_types.unique()):
        groups.append(np.flatnonzero(is_anomaly & (types == type_).to_numpy()))
    fractions = [_NORMAL_FRACTIONS] + [_ANOMALY_FRACTIONS] * (len(groups) - 1)

    for indices, group_fractions in zip(groups, fractions):
        for part, chunk in zip(parts, _partition(indices, group_fractions, rng)):
            part.append(chunk)

    train, validation, test = (np.sort(np.concatenate(p)) for p in parts)

    # Defesa redundante de propósito: a regra mais importante da divisão não pode
    # depender só da fração 0 em config.py.
    if is_anomaly[train].any():
        raise ValueError("Anomalia no treino (decisões, seção 4.1).")

    return Split(train=train, validation=validation, test=test)


def _partition(
    indices: np.ndarray,
    fractions: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embaralha ``indices`` e corta nas frações dadas.

    Os cortes usam piso das frações acumuladas; a sobra do arredondamento (no máximo um
    bloco por corte) vai para a última partição, o teste. Isso garante que uma fração 0
    produza partição vazia, e não um bloco perdido por arredondamento.
    """
    shuffled = rng.permutation(indices)
    n = len(shuffled)
    cut_train = int(np.floor(fractions[0] * n))
    cut_validation = int(np.floor((fractions[0] + fractions[1]) * n))
    return (
        shuffled[:cut_train],
        shuffled[cut_train:cut_validation],
        shuffled[cut_validation:],
    )