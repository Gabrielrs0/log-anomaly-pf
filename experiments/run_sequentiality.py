"""Classificação sequencial × não-sequencial dos tipos de anomalia (decisões, seção 5).

Roda as três etapas e a regra de classificação final sobre o HDFS_v1 completo e grava:

- ``results/sequentiality_types.csv``: uma linha por ``type``, com cada sinal em coluna
  própria e o rótulo final. É a tabela que a monografia cita.
- ``results/block_categories.csv``: ``BlockId``, ``type`` e ``category`` de cada bloco
  ANÔMALO — a entrada da avaliação segmentada (decisões, seção 4.7).

Só os anômalos vão para ``block_categories.csv``: os normais não têm categoria, e
incluí-los levaria o arquivo de ~0,7 MB a ~20 MB, contra a premissa de que
``results/`` é pequeno o bastante para o git (decisões, seção 7). Quem consome o
arquivo NÃO pode tomá-lo como universo da avaliação: ao avaliar uma categoria, os
negativos continuam sendo todos os blocos normais (decisões, seção 4.7).

Determinístico: nenhuma etapa amostra, então não há semente envolvida.

Uso, a partir da raiz do repositório::

    python -m experiments.run_sequentiality
"""

from __future__ import annotations

import time

import pandas as pd

from src.config import RESULTS_DIR
from src.data.loader import load_event_traces
from src.labeling.sequentiality import classify_types

TYPES_PATH = RESULTS_DIR / "sequentiality_types.csv"
BLOCKS_PATH = RESULTS_DIR / "block_categories.csv"

# Separador dos campos com lista (eventos distintivos, violações confirmadas). Vírgula
# colidiria com o separador do CSV e forçaria aspas.
_LIST_SEP = ";"


def main() -> None:
    start = time.perf_counter()
    traces = load_event_traces()
    types = classify_types(traces)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _types_table(types).to_csv(TYPES_PATH, index=False)
    blocks = _block_categories(traces.blocks, types)
    blocks.to_csv(BLOCKS_PATH)

    summary = types.groupby("classification")["n_blocks"].agg(tipos="size", blocos="sum")
    print(summary.to_string())
    print(f"\n{len(types)} tipos → {TYPES_PATH.relative_to(RESULTS_DIR.parent)}")
    print(f"{len(blocks)} blocos anômalos → {BLOCKS_PATH.relative_to(RESULTS_DIR.parent)}")
    print(f"{time.perf_counter() - start:.1f} s")


def _types_table(types: pd.DataFrame) -> pd.DataFrame:
    """Converte as colunas de lista em texto para o CSV."""
    out = types.copy()
    for col in ("distinctive_events", "confirmed_violations"):
        out[col] = out[col].map(_LIST_SEP.join)
    return out


def _block_categories(blocks: pd.DataFrame, types: pd.DataFrame) -> pd.DataFrame:
    """Categoria de cada bloco anômalo, herdada do seu ``type``."""
    anomalous = blocks.loc[blocks["label"] == 1, ["type"]]
    category = types.set_index("type")["classification"]
    out = anomalous.assign(category=anomalous["type"].map(category))
    # Todo type vem de classify_types sobre os mesmos blocos; uma categoria ausente
    # indicaria blocos e tipos de execuções diferentes.
    if out["category"].isna().any():
        raise ValueError("Bloco anômalo sem categoria.")
    return out


if __name__ == "__main__":
    main()
