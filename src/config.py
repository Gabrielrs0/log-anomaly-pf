"""Configuração central do projeto: caminhos, constantes do conjunto e parâmetros.

Todo número que influencia um resultado vive aqui, com a fonte da decisão ao lado.
Os demais módulos importam daqui em vez de repetir literais, para que a monografia
possa citar um único lugar como registro dos parâmetros usados (decisões, seção 9).

Valores marcados como ``None`` correspondem a decisões metodológicas ainda NÃO
fechadas. Ficam nulos de propósito: o código que os consome deve falhar em vez de
assumir um padrão silencioso (decisões, seção 1).

As referências "decisões, seção X" no código apontam para
``docs/decisoes-metodologicas.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

RANDOM_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

# Resolvido a partir deste arquivo, e não do diretório corrente, para que os scripts
# de experiments/ funcionem igual localmente e no Colab, onde data/ é um conjunto de
# symlinks para o Drive criado pelo notebooks/colab_runner.ipynb.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"

# Sequência ORDENADA de eventos por bloco — única fonte de informação de ordem.
# Só pode alimentar o LSTM Autoencoder e o teste de sequencialidade (decisões, seção 3).
EVENT_TRACES_PATH: Final[Path] = DATA_DIR / "Event_traces.csv"

# Contagem por bloco, sem ordem — alimenta PCA e Isolation Forest (decisões, seção 3).
EVENT_OCCURRENCE_MATRIX_PATH: Final[Path] = DATA_DIR / "Event_occurrence_matrix.csv"

ANOMALY_LABEL_PATH: Final[Path] = DATA_DIR / "anomaly_label.csv"
LOG_TEMPLATES_PATH: Final[Path] = DATA_DIR / "HDFS.log_templates.csv"

# ---------------------------------------------------------------------------
# Constantes do conjunto HDFS_v1 (Loghub, Zenodo 8196385)
# ---------------------------------------------------------------------------

# Contagens publicadas da distribuição. Servem para conferir a carga dos arquivos:
# divergência indica arquivo truncado ou versão diferente do conjunto.
N_BLOCKS: Final[int] = 575_061
N_ANOMALOUS_BLOCKS: Final[int] = 16_838

N_EVENT_TEMPLATES: Final[int] = 29

# Ordem numérica (E1, E2, ..., E29), não lexicográfica (E1, E10, E11, ...): fixa a
# ordem das colunas da matriz de ocorrência independentemente da ordem no CSV.
EVENT_IDS: Final[tuple[str, ...]] = tuple(
    f"E{i}" for i in range(1, N_EVENT_TEMPLATES + 1)
)

# Eventos-núcleo do ciclo de vida do bloco, sobre cujos pares roda o teste de
# precedência (decisões, seção 5, etapa 2).
CORE_EVENTS: Final[tuple[str, ...]] = ("E5", "E9", "E11", "E21", "E22", "E23", "E26")

# ---------------------------------------------------------------------------
# Divisão treino/validação/teste (decisões, seção 4.1)
# ---------------------------------------------------------------------------

# Os três métodos são não supervisionados: o treino contém apenas blocos normais.
NORMAL_TRAIN_FRACTION: Final[float] = 0.60
NORMAL_VALIDATION_FRACTION: Final[float] = 0.20
NORMAL_TEST_FRACTION: Final[float] = 0.20

ANOMALY_TRAIN_FRACTION: Final[float] = 0.0
ANOMALY_VALIDATION_FRACTION: Final[float] = 0.50
ANOMALY_TEST_FRACTION: Final[float] = 0.50

# ---------------------------------------------------------------------------
# Isolation Forest (decisões, seção 4.3)
# ---------------------------------------------------------------------------

# NÃO usar o padrão 256. Liu, Ting e Zhou (2008): com treino só de normais é preciso
# aumentar a subamostra; além disso, 256 blocos quase nunca contêm os eventos raros
# que definem as anomalias. O valor vem da literatura, não de medição nesta base: a
# varredura que o justifica está pendente (decisões, seção 4.3).
IFOREST_MAX_SAMPLES: Final[int] = 65_536

# ---------------------------------------------------------------------------
# Classificação sequencial × não-sequencial (decisões, seção 5)
# ---------------------------------------------------------------------------

# Tipos com menos blocos que isto ficam "indeterminados": frequências estimadas sobre
# poucos blocos oscilam demais para sustentar qualquer uma das duas classificações.
MIN_BLOCKS_PER_TYPE: Final[int] = 10

# Etapa 1 — eventos distintivos (sinal NÃO-sequencial, definido por presença).
# Um evento é distintivo de um Type quando aparece em MAIS de DISTINCTIVE_MIN_TYPE_FREQ
# dos blocos da categoria E em MENOS de DISTINCTIVE_MAX_NORMAL_FREQ dos normais
# (desigualdades estritas).
DISTINCTIVE_MIN_TYPE_FREQ: Final[float] = 0.50
DISTINCTIVE_MAX_NORMAL_FREQ: Final[float] = 0.05

# Etapa 2 — teste de precedência (sinal sequencial).
# Um par (A, B) só entra na referência se, nos normais, P(A antes de B) for quase fixa:
# P >= PRECEDENCE_FIXED_ORDER_HIGH ou P <= PRECEDENCE_FIXED_ORDER_LOW (inclusivas).
# Pares sem ordem estável nos normais não permitem falar em "violação".
PRECEDENCE_FIXED_ORDER_HIGH: Final[float] = 0.90
PRECEDENCE_FIXED_ORDER_LOW: Final[float] = 0.10

# Divergência |P_categoria - P_normal| a partir da qual (inclusive) o par conta como
# violação de ordem. As probabilidades consideram só blocos em que AMBOS os eventos
# aparecem; caso contrário a ausência de um evento, que é sinal não-sequencial,
# contaminaria a medida de ordem.
PRECEDENCE_MIN_DIVERGENCE: Final[float] = 0.25

# Mínimo de blocos do tipo com AMBOS os eventos do par para que a divergência conte como
# violação. Abaixo disso, uma divergência é indistinguível de coincidência: o Type 13
# geraria violação E21/E23 a partir de 2 blocos. O par continua na tabela, marcado, para
# que o descarte seja auditável.
MIN_PAIR_SUPPORT: Final[int] = 5

# Etapa 1, sinal de comprimento (NÃO-sequencial: comprimento é a soma das contagens).
# Landauer et al. (2024) apontam três manifestações de anomalia no HDFS: tipos de
# evento novos, comprimento de sequência e contagens. Sem este sinal, o Type 3
# (truncamento puro: 2 eventos por bloco, contra mínimo de 13 nos normais) fica sem
# sinal algum. Um tipo é marcado quando pelo menos esta fração (inclusiva) de seus
# blocos tem comprimento fora da faixa [min, max] dos normais; 0,90 caracteriza o
# comprimento anômalo como traço da categoria, não de uns poucos blocos, espelhando
# PRECEDENCE_FIXED_ORDER_HIGH e SIMULTANEITY_MIN_FRACTION.
LENGTH_SIGNAL_MIN_FRACTION: Final[float] = 0.90

# Etapa 3 — verificação de simultaneidade (OBRIGATÓRIA).
# Landauer et al. (2024): no HDFS, eventos com o mesmo carimbo de tempo são registrados
# em ordem aleatória, então uma "violação" da etapa 2 pode ser artefato de registro.
# A violação vira artefato quando, entre os blocos do tipo com o par na ordem invertida,
# a fração com intervalo |t(B) - t(A)| <= SIMULTANEITY_MAX_GAP_SECONDS é
# >= SIMULTANEITY_MIN_FRACTION (ambas inclusivas).
#   - 0 s: "intervalo nulo" no sentido literal das decisões; os carimbos do HDFS.log
#     têm resolução de 1 s, então 0 significa "mesmo segundo".
#   - 0,90: "consistentemente", espelhando PRECEDENCE_FIXED_ORDER_HIGH da etapa 2.
# Nos dados reais o resultado é idêntico para gap de 0 a 2 s e fração de 0,5 a 1,0
# (decisões, seção 5).
SIMULTANEITY_MAX_GAP_SECONDS: Final[float] = 0
SIMULTANEITY_MIN_FRACTION: Final[float] = 0.90


# ---------------------------------------------------------------------------
# Coerência interna
# ---------------------------------------------------------------------------


def _validate() -> None:
    """Falha na importação se as constantes acima se contradisserem.

    Usa ``raise`` em vez de ``assert`` porque ``python -O`` remove asserts, e um erro
    de configuração precisa aparecer sempre.
    """
    unknown_core = set(CORE_EVENTS) - set(EVENT_IDS)
    if unknown_core:
        raise ValueError(f"CORE_EVENTS fora de EVENT_IDS: {sorted(unknown_core)}")

    # Tolerância só para absorver erro de ponto flutuante na soma das frações.
    tol = 1e-9
    normal_sum = (
        NORMAL_TRAIN_FRACTION + NORMAL_VALIDATION_FRACTION + NORMAL_TEST_FRACTION
    )
    anomaly_sum = (
        ANOMALY_TRAIN_FRACTION + ANOMALY_VALIDATION_FRACTION + ANOMALY_TEST_FRACTION
    )
    if abs(normal_sum - 1.0) > tol or abs(anomaly_sum - 1.0) > tol:
        raise ValueError("Frações de divisão não somam 1.")
    if ANOMALY_TRAIN_FRACTION != 0.0:
        raise ValueError("Treino não pode conter anomalias (decisões, seção 4.1).")

    if not PRECEDENCE_FIXED_ORDER_LOW < PRECEDENCE_FIXED_ORDER_HIGH:
        raise ValueError("Limiares de ordem fixa invertidos.")
    if MIN_PAIR_SUPPORT < 1 or not 0 < LENGTH_SIGNAL_MIN_FRACTION <= 1:
        raise ValueError("MIN_PAIR_SUPPORT ou LENGTH_SIGNAL_MIN_FRACTION fora do domínio.")
    if SIMULTANEITY_MAX_GAP_SECONDS < 0 or not 0 < SIMULTANEITY_MIN_FRACTION <= 1:
        raise ValueError("Limiares de simultaneidade fora do domínio.")


_validate()