# Sequencialidade de anomalias em logs de sistemas

Código do Projeto Final de Curso em Engenharia de Computação (UnB/ENE).

Comparação entre baseline estatístico (PCA), Isolation Forest e LSTM Autoencoder na
detecção de anomalias no conjunto HDFS_v1 (Loghub), com **avaliação segmentada** entre
anomalias sequenciais e não-sequenciais.

> As decisões metodológicas do projeto estão em
> [`docs/decisoes-metodologicas.md`](docs/decisoes-metodologicas.md). Leia antes de
> alterar qualquer coisa em `src/labeling/`, `src/models/` ou `src/evaluation/`.

## Estrutura

```
src/
├── config.py          caminhos, constantes, hiperparâmetros
├── data/              carregamento e divisão treino/validação/teste
├── features/          matriz de ocorrência, TF-IDF, sequências
├── labeling/          classificação sequencial × não-sequencial
├── models/            baseline PCA, Isolation Forest, LSTM Autoencoder
└── evaluation/        métricas segmentadas, tempo de detecção
experiments/           scripts reproduzíveis de ponta a ponta
docs/                  decisões metodológicas e suas justificativas
notebooks/             orquestração no Colab (sem lógica)
results/               saídas .csv (versionadas)
data/                  dados brutos (NÃO versionados)
```

## Onde cada coisa roda

| Ambiente | Para quê |
|---|---|
| Local (Claude Code) | escrever e revisar código, rodar testes, experimentos leves |
| Google Colab | execução com GPU — treinamento do LSTM Autoencoder |
| Git / GitHub | ponte entre os dois; histórico e reprodutibilidade |

Baseline PCA e Isolation Forest rodam em segundos na CPU e não precisam do Colab.

## Dados

Baixar `HDFS_v1.zip` do Loghub:
<https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1>

Colocar em `data/` (local) e numa pasta do Google Drive (Colab). Arquivos necessários:
`Event_traces.csv`, `Event_occurrence_matrix.csv`, `anomaly_label.csv`,
`HDFS.log_templates.csv`.

## Execução

```bash
pip install -r requirements.txt

python -m experiments.run_sequentiality   # classifica as anomalias
python -m experiments.run_comparison      # treina e avalia os modelos
```

Cada script é reproduzível do zero: não depende de estado manual nem de execução prévia
em notebook.

## Referências

| | |
|---|---|
| [1] | XU, W. et al. Detecting large-scale system problems by mining console logs. SOSP, 2009. |
| [2] | HE, S. et al. Experience report: System log analysis for anomaly detection. ISSRE, 2016. |
| [3] | LIU, F. T.; TING, K. M.; ZHOU, Z.-H. Isolation Forest. ICDM, 2008. |
| [4] | HOCHREITER, S.; SCHMIDHUBER, J. Long Short-Term Memory. Neural Computation, 1997. |
| [5] | LANDAUER, M.; SKOPIK, F.; WURZENBERGER, M. A critical review of common log data sets. FSE, 2024. |
| [6] | ZHU, J. et al. Loghub: a large collection of system log datasets. ISSRE, 2023. |
| [7] | HE, S. et al. Drain: An online log parsing approach with fixed depth tree. ICWS, 2017. |
