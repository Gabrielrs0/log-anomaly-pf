# Decisões metodológicas e contexto do projeto

Este documento registra o contexto de pesquisa e as **decisões metodológicas já fechadas**.
Leia-o antes de propor qualquer alteração em `src/labeling/`, `src/models/` ou
`src/evaluation/`. Várias decisões abaixo contrariam o padrão das bibliotecas ou a
recomendação usual da literatura — todas têm justificativa registrada. **Não as
reverta sem discussão explícita.**

---

## 1. O que é este projeto

Projeto Final de Curso em Engenharia de Computação (UnB/ENE).

**Pergunta de pesquisa:** o ganho de desempenho do LSTM Autoencoder sobre um baseline
estatístico e sobre o Isolation Forest se mantém proporcional à sua complexidade quando
as anomalias do HDFS_v1 são segmentadas em sequenciais e não-sequenciais?

**Hipótese:** o ganho não é uniforme. Para anomalias não-sequenciais — definidas pela
presença ou ausência de eventos específicos — espera-se ganho pequeno ou nulo. Para
anomalias genuinamente sequenciais, espera-se vantagem mais evidente.

O valor do trabalho **não está no código**, está nas decisões metodológicas. Código que
roda embutindo uma premissa errada é pior que código que não roda, porque não avisa.

---

## 2. Conjunto de dados

HDFS_v1, do projeto Loghub (Zenodo, registro 8196385).

- 575.061 blocos; 16.838 anomalias (2,9%)
- 29 templates de evento (E1–E29), já parseados na distribuição
- Agrupamento por `BlockId` (janela de sessão)

**Arquivos** (ficam em `data/`, que é ignorado pelo git — ver seção 7):
- `Event_traces.csv` — sequência **ordenada** de eventos por bloco (campo `Features`)
- `Event_occurrence_matrix.csv` — matriz de **contagem** por bloco (bag-of-events)
- `anomaly_label.csv` — rótulos
- `HDFS.log_templates.csv` — mapa EventId → template

**Campo `Type`:** subcategoria de falha, preenchida apenas nos blocos anômalos. Ausente
da documentação do `loglizer`. Zhu et a# Decisões metodológicas e contexto do projeto

Este documento registra o contexto de pesquisa e as **decisões metodológicas já fechadas**.
Leia-o antes de propor qualquer alteração em `src/labeling/`, `src/models/` ou
`src/evaluation/`. Várias decisões abaixo contrariam o padrão das bibliotecas ou a
recomendação usual da literatura — todas têm justificativa registrada. **Não as
reverta sem discussão explícita.**

---

## 1. O que é este projeto

Projeto Final de Curso em Engenharia de Computação (UnB/ENE).

**Pergunta de pesquisa:** o ganho de desempenho do LSTM Autoencoder sobre um baseline
estatístico e sobre o Isolation Forest se mantém proporcional à sua complexidade quando
as anomalias do HDFS_v1 são segmentadas em sequenciais e não-sequenciais?

**Hipótese:** o ganho não é uniforme. Para anomalias não-sequenciais — definidas pela
presença ou ausência de eventos específicos — espera-se ganho pequeno ou nulo. Para
anomalias genuinamente sequenciais, espera-se vantagem mais evidente.

O valor do trabalho **não está no código**, está nas decisões metodológicas. Código que
roda embutindo uma premissa errada é pior que código que não roda, porque não avisa.

---

## 2. Conjunto de dados

HDFS_v1, do projeto Loghub (Zenodo, registro 8196385).

- 575.061 blocos; 16.838 anomalias (2,9%)
- 29 templates de evento (E1–E29), já parseados na distribuição
- Agrupamento por `BlockId` (janela de sessão)

**Arquivos** (ficam em `data/`, que é ignorado pelo git — ver seção 7):
- `Event_traces.csv` — sequência **ordenada** de eventos por bloco (campo `Features`)
- `Event_occurrence_matrix.csv` — matriz de **contagem** por bloco (bag-of-events)
- `anomaly_label.csv` — rótulos
- `HDFS.log_templates.csv` — mapa EventId → template

**Campo `Type`:** subcategoria de falha, preenchida apenas nos blocos anômalos. Ausente
da documentação do `loglizer`. Zhu et al. (2023) o descrevem como "tipo específico de
anomalia", disponibilizado para pesquisa em identificação de problemas duplicados.
Provavelmente corresponde às 11 categorias que Xu et al. (2009) rotularam manualmente:
sete dos tipos têm contagem de blocos coincidente com as categorias nomeadas no artigo
*(observação exploratória, medida fora do pipeline)*. É **indício, não fato verificado** —
a confirmação exige conferência bloco a bloco e, até que ela seja feita, a
correspondência não deve ser afirmada na monografia.

---

## 3. As duas representações — e por que a distinção é o centro do trabalho

| Representação | Fonte | Preserva ordem? | Usada por |
|---|---|---|---|
| Matriz de ocorrência | `Event_occurrence_matrix.csv` | **Não** | Baseline PCA, Isolation Forest |
| Sequência ordenada | `Event_traces.csv` (campo `Features`) | **Sim** | LSTM Autoencoder, teste de sequencialidade |

Essa separação **é** a pergunta de pesquisa codificada. Nunca alimente o PCA ou o
Isolation Forest com informação de ordem, nem o LSTM com a matriz de contagem — isso
destrói o contraste que o trabalho investiga.

---

## 4. Decisões metodológicas fechadas

### 4.1 Treino contém APENAS blocos normais

Os três métodos são não supervisionados e aprendem o padrão de normalidade. Incluir
anomalias no treino contamina exatamente o que precisam aprender.

- Normais: 60% treino, 20% validação, 20% teste
- Anomalias: 50% validação, 50% teste — **nenhuma no treino**

### 4.2 Padronizar antes do PCA — obrigatório

Sem `StandardScaler`, os eventos frequentes dominam a variância e o PCA praticamente
ignora os eventos raros. Como a maioria das anomalias do HDFS é definida justamente pela
presença de evento raro, omitir a padronização inutiliza o método.

### 4.3 Isolation Forest: `max_samples` grande, não o padrão

O padrão da biblioteca é 256, seguindo a recomendação do artigo original. Há duas razões
documentadas para esperar que ele degrade o desempenho **neste** conjunto:

1. Liu, Ting e Zhou (2008), seção sobre treino apenas com instâncias normais: quando o
   treinamento não contém anomalias, **é preciso aumentar a subamostra** para restaurar
   o desempenho. É exatamente o cenário aqui (4.1).
2. As anomalias são definidas por eventos muito raros; uma subamostra de 256 blocos
   quase nunca os contém, e a árvore sequer percebe que a coluna varia.

**Valor adotado: `IFOREST_MAX_SAMPLES = 65_536`.** Não reverter para 256 sem antes rodar
o experimento abaixo.

A escolha é, por ora, uma previsão derivada da literatura — **não** um resultado deste
trabalho. Quando `src/models/isolation_forest.py` existir, a primeira execução deve ser
uma varredura de `max_samples` (256, 4.096, 65.536) avaliada por AUC na validação, com a
tabela gravada em `results/`. É ela que justifica, na monografia, o afastamento do padrão
da biblioteca; até lá o valor adotado não tem medição própria que o sustente.

### 4.4 Ajuste de hiperparâmetro vai na VALIDAÇÃO, não no treino

O treino não tem rótulos (4.1), logo validação cruzada supervisionada sobre ele é
impossível. Todo ajuste de limiar ou hiperparâmetro usa o conjunto de validação.

### 4.5 Ponderação TF-IDF antes do PCA

Três fontes independentes recomendam. Xu et al. (2009) a descrevem como **essencial**:
sem TF-IDF o método detecta 10.217 das 16.916 anomalias; com TF-IDF, 16.808. Landauer
et al. (2024) confirmam o ganho em vetores de contagem.

Deve ser implementada como etapa **opcional e comparável** (flag), para que o trabalho
reporte o desempenho com e sem.

### 4.6 Métricas: incluir TNR

Precisão, recall, F1, **TNR** e tempo de detecção.

A TNR não é opcional. Landauer et al. documentam um detector com F1 de 91,5% e recall de
100% que classificava **todas** as instâncias como anômalas — degeneração revelada apenas
pela TNR nula. Com desbalanceamento de 2,9%, reportar só P/R/F1 permite a mesma leitura
enganosa.

### 4.7 Avaliação SEGMENTADA por categoria — o núcleo do trabalho

Métricas calculadas separadamente para anomalias sequenciais e não-sequenciais.

**Cuidado que não pode ser perdido:** ao avaliar uma categoria, os positivos são apenas
as anomalias daquela categoria, mas os **negativos continuam sendo todos os blocos
normais**. Descartar os normais infla artificialmente a precisão e o número não significa
nada.

### 4.8 Múltiplas execuções com variância

Cada configuração roda várias vezes com amostragens distintas; reportar média e variância,
não o melhor resultado. Recomendação explícita de Landauer et al.

---

## 5. Classificação sequencial × não-sequencial

Procedimento em três etapas, em `src/labeling/`:

1. **Eventos distintivos** — para cada `Type`, eventos presentes em >50% dos blocos da
   categoria e em <5% dos normais. Indica anomalia **não-sequencial** (definida por
   presença).

2. **Teste de precedência** — para pares de eventos-núcleo
   (`E5, E9, E11, E21, E22, E23, E26`), compara P(A antes de B) na categoria contra o
   padrão dos blocos normais. Pares com ordem quase fixa nos normais (P ≥ 0,90 ou ≤ 0,10)
   formam a referência; divergência ≥ 0,25 conta como violação de ordem, indicando
   anomalia **sequencial**.

   Considerar apenas blocos em que **ambos** os eventos do par aparecem — senão a ausência
   de um evento (que é sinal não-sequencial) contamina a medida de ordem.

3. **Verificação de simultaneidade — OBRIGATÓRIA.** Landauer et al. demonstram que, no
   HDFS, eventos gerados simultaneamente são registrados em **ordem aleatória**: as três
   sequências *normais* mais comuns diferem apenas pela posição do evento 22 entre os
   quatro primeiros, que ocorrem no mesmo instante.

   Logo, toda violação detectada na etapa 2 deve ser confrontada com os carimbos de tempo
   do par. Intervalo consistentemente nulo ⇒ **artefato de registro**, não anomalia
   sequencial. Separação temporal clara ⇒ reordenação genuína.

   **Sem essa verificação, o resultado da etapa 2 não é confiável.**

   Limiares adotados (`config.py`): para cada violação, tomam-se os blocos da categoria
   com o par na ordem **invertida** e mede-se `|t(B) − t(A)|` entre as primeiras
   ocorrências (carimbos acumulados de `TimeInterval`). A violação é artefato quando
   ≥ 90% desses blocos têm intervalo ≤ 0 s, isto é, os dois eventos caem no mesmo
   segundo (a resolução do log é de 1 s). Nos dados reais o resultado não muda para
   intervalos de 0 a 2 s e frações de 0,5 a 1,0.

   Convenção comum às etapas 2 e 3: como os eventos-núcleo se repetem no bloco (um por
   réplica), "A antes de B" compara a **primeira** ocorrência de cada um.

### 5.4 Regra de classificação final

Os sinais das etapas 1 e 2 **não são mutuamente excludentes**: uma categoria pode
introduzir eventos raros e, ao mesmo tempo, reordenar eventos existentes. Colapsá-los
em uma dicotomia exigiria arbitrar qual prevalece, e qualquer arbitragem esvaziaria
uma das categorias — ambos os tipos com violação confirmada (0 e 8) também possuem
eventos distintivos.

A classificação adota, portanto, quatro saídas:

| Classificação | Sinal não-sequencial | Violação confirmada (etapa 3) |
|---|---|---|
| não-sequencial | sim | não |
| ambos os sinais | sim | sim |
| sequencial | não | sim |
| indeterminado | não | não, ou amostra insuficiente |

**Sinal não-sequencial** reúne duas manifestações, ambas detectáveis por contagem e
independentes de ordem: a presença de evento distintivo (etapa 1) e o comprimento de
sequência fora do intervalo observado nos blocos normais. A inclusão do comprimento
segue Landauer, Skopik e Wurzenberger (2024), que identificam três manifestações de
anomalia no HDFS — novos tipos de evento, comprimento de sequência e contagens.

O intervalo de referência é `[mín, máx]` do comprimento nos blocos normais (13 a 298
eventos). Uma categoria tem sinal de comprimento quando ≥ `LENGTH_SIGNAL_MIN_FRACTION`
= 0,90 de seus blocos caem fora dele — o mesmo patamar de "consistente" usado nas
etapas 2 e 3, para que o sinal seja traço da categoria e não de uns poucos blocos. Nos
dados reais, só os Types 3 e 31 o têm (100% dos blocos fora); o 31 já tinha evento
distintivo (E7).

Sem o sinal de comprimento, o Type 3 (2.950 blocos, 17,5% das anomalias) não
apresentaria sinal algum: seus blocos têm exatamente dois eventos, `[E5,E22]` ou
`[E22,E5]` *(observação exploratória: 1.643 e 1.307 blocos)*, contra o mínimo de treze
observado nos blocos normais. Os dois eventos são os mais frequentes do conjunto, de modo
que nenhum deles é distintivo.

O teste de precedência também não alcança o Type 3, mas **não** por a sequência ser curta:
o par de eventos-núcleo E5/E22 está presente nos 2.950 blocos e tem suporte de sobra. O
que o exclui é a etapa de referência — se a ordem entre E5 e E22 não é fixa nos blocos
normais, o par nunca chega a ser candidato a violação. É o esperado, porque a permutação
por simultaneidade descrita por Landauer, Skopik e Wurzenberger (2024) atinge justamente
o evento 22; mas a afirmação precisa ser lida na coluna `is_reference` da tabela de
precedência, que hoje não é gravada em `results/` (seção 9). A variação de ordem entre as
duas formas do Type 3 é, ela mesma, permutação por simultaneidade *(observação
exploratória: 88% desses blocos têm E5 e E22 no mesmo segundo, intervalo máximo de 3 s)*,
e o critério corretamente não a classifica como violação.

O teste de precedência exige ainda suporte mínimo de `MIN_PAIR_SUPPORT` = 5 blocos da
categoria contendo ambos os eventos do par. Sem esse filtro, o Type 13 produziria
violação a partir de dois blocos, estatisticamente indistinguível de coincidência. O par
descartado continua na tabela do teste (`has_min_support = False`), para que o
descarte seja auditável.

---

## 6. Arquitetura

Cada camada conhece apenas as camadas abaixo:

```
experiments/  →  models/ + labeling/ + evaluation/  →  data/  →  config.py
```

- `config.py` — caminhos, constantes, hiperparâmetros. Sem número mágico espalhado.
- `data/` — lê arquivos, devolve estruturas. **Não sabe o que é anomalia.**
- `labeling/` — classifica anomalias. **Não sabe que existem modelos.**
- `models/` — treina e pontua. Interface única: `fit`, `score_samples`, `predict`.
- `evaluation/` — calcula métricas. **Não sabe qual modelo as gerou.**
- `experiments/` — única camada que amarra tudo. Cada script é reproduzível de ponta a ponta.

**Interface obrigatória dos modelos.** Todos expõem `fit(X)`, `score_samples(X)` (escore
contínuo, **maior = mais anômalo**) e `predict(X)` (1 = anomalia, 0 = normal). O
scikit-learn inverte essas convenções no Isolation Forest (-1 = outlier, escore menor =
mais anômalo); normalizar no wrapper, nunca vazar a convenção dele para cima.

---

## 7. Execução e dados

Desenvolvimento local com Claude Code; execução pesada no Google Colab (GPU para o LSTM).
Git é a ponte.

**Os arquivos de dados NÃO vão para o git.** `Event_traces.csv` tem 120 MB e estoura o
limite de 100 MB por arquivo do GitHub. Ficam no Google Drive e são montados no Colab.
`data/` está no `.gitignore`.

**Resultados (`results/*.csv`) VÃO para o git** — são pequenos e constituem o registro
histórico dos experimentos.

---

## 8. Convenções de código

- Python 3.10+; identificadores e nomes de função em inglês
- Docstrings e comentários em **português** (a monografia é em português)
- Comentário explica **por quê**, não o quê
- Sem notebooks para lógica: notebook só orquestra, a lógica vive em `src/`
- `RANDOM_SEED = 42` em tudo que amostra
- Toda decisão metodológica não óbvia leva comentário citando a fonte
- No código, seções deste documento são citadas como `(decisões, seção X)`

## 9. Reprodutibilidade

Landauer et al. recomendam — e a Seção 3.4 da monografia adota — publicar código, dados,
estratégia de amostragem, divisão treino/teste e parâmetros. Nenhum experimento depende
de estado manual: rodar o script do zero reproduz o resultado.

### 9.1 Procedência dos números deste documento

Cada número aqui é de um de dois tipos, e eles são marcados de forma diferente:

- **Resultado do pipeline** — produzido por código deste repositório. Aponta para o
  arquivo de `results/` que o contém, ou nomeia a função que o calcula. Verificável
  rodando o script.
- **Observação exploratória** — medida fora do pipeline, para entender a base. Vem
  marcada *(observação exploratória)* no ponto em que aparece.

Nenhuma afirmação deve ficar numa terceira categoria: **resultado apresentado como se
fosse do pipeline, sem código que o produza.** Era o caso da tabela de AUC da seção 4.3,
que descrevia uma varredura de `max_samples` jamais executada neste repositório — o
módulo `src/models/` ainda não existe. Um número nessa situação não é defensável na banca
e não deve entrar na monografia.


## 10. Limitações já registradas (não contornar silenciosamente)

- Os rótulos do HDFS_v1 vêm de **regras heurísticas construídas à mão** sobre vetores de
  contagem não ordenados, não de falhas confirmadas em produção. Conclusões sobre "a
  natureza das anomalias do HDFS" são, a rigor, conclusões sobre o que aquelas regras
  capturam.
- A permutação por simultaneidade compromete estruturalmente a capacidade do HDFS de
  avaliar detectores sequenciais.
l. (2023) o descrevem como "tipo específico de
anomalia", disponibilizado para pesquisa em identificação de problemas duplicados.
Provavelmente corresponde às 11 categorias que Xu et al. (2009) rotularam manualmente:
sete dos tipos têm contagem de blocos coincidente com as categorias nomeadas no artigo
*(observação exploratória, medida fora do pipeline)*. É **indício, não fato verificado** —
a confirmação exige conferência bloco a bloco e, até que ela seja feita, a
correspondência não deve ser afirmada na monografia.

---

## 3. As duas representações — e por que a distinção é o centro do trabalho

| Representação | Fonte | Preserva ordem? | Usada por |
|---|---|---|---|
| Matriz de ocorrência | `Event_occurrence_matrix.csv` | **Não** | Baseline PCA, Isolation Forest |
| Sequência ordenada | `Event_traces.csv` (campo `Features`) | **Sim** | LSTM Autoencoder, teste de sequencialidade |

Essa separação **é** a pergunta de pesquisa codificada. Nunca alimente o PCA ou o
Isolation Forest com informação de ordem, nem o LSTM com a matriz de contagem — isso
destrói o contraste que o trabalho investiga.

---

## 4. Decisões metodológicas fechadas

### 4.1 Treino contém APENAS blocos normais

Os três métodos são não supervisionados e aprendem o padrão de normalidade. Incluir
anomalias no treino contamina exatamente o que precisam aprender.

- Normais: 60% treino, 20% validação, 20% teste
- Anomalias: 50% validação, 50% teste — **nenhuma no treino**

### 4.2 Padronizar antes do PCA — obrigatório

Sem `StandardScaler`, os eventos frequentes dominam a variância e o PCA praticamente
ignora os eventos raros. Como a maioria das anomalias do HDFS é definida justamente pela
presença de evento raro, omitir a padronização inutiliza o método.

### 4.3 Isolation Forest: `max_samples` grande, não o padrão

O padrão da biblioteca é 256, seguindo a recomendação do artigo original. Há duas razões
documentadas para esperar que ele degrade o desempenho **neste** conjunto:

1. Liu, Ting e Zhou (2008), seção sobre treino apenas com instâncias normais: quando o
   treinamento não contém anomalias, **é preciso aumentar a subamostra** para restaurar
   o desempenho. É exatamente o cenário aqui (4.1).
2. As anomalias são definidas por eventos muito raros; uma subamostra de 256 blocos
   quase nunca os contém, e a árvore sequer percebe que a coluna varia.

**Valor adotado: `IFOREST_MAX_SAMPLES = 65_536`.** Não reverter para 256 sem antes rodar
o experimento abaixo.

A escolha é, por ora, uma previsão derivada da literatura — **não** um resultado deste
trabalho. Quando `src/models/isolation_forest.py` existir, a primeira execução deve ser
uma varredura de `max_samples` (256, 4.096, 65.536) avaliada por AUC na validação, com a
tabela gravada em `results/`. É ela que justifica, na monografia, o afastamento do padrão
da biblioteca; até lá o valor adotado não tem medição própria que o sustente.

### 4.4 Ajuste de hiperparâmetro vai na VALIDAÇÃO, não no treino

O treino não tem rótulos (4.1), logo validação cruzada supervisionada sobre ele é
impossível. Todo ajuste de limiar ou hiperparâmetro usa o conjunto de validação.

### 4.5 Ponderação TF-IDF antes do PCA

Três fontes independentes recomendam. Xu et al. (2009) a descrevem como **essencial**:
sem TF-IDF o método detecta 10.217 das 16.916 anomalias; com TF-IDF, 16.808. Landauer
et al. (2024) confirmam o ganho em vetores de contagem.

Deve ser implementada como etapa **opcional e comparável** (flag), para que o trabalho
reporte o desempenho com e sem.

### 4.6 Métricas: incluir TNR

Precisão, recall, F1, **TNR** e tempo de detecção.

A TNR não é opcional. Landauer et al. documentam um detector com F1 de 91,5% e recall de
100% que classificava **todas** as instâncias como anômalas — degeneração revelada apenas
pela TNR nula. Com desbalanceamento de 2,9%, reportar só P/R/F1 permite a mesma leitura
enganosa.

### 4.7 Avaliação SEGMENTADA por categoria — o núcleo do trabalho

Métricas calculadas separadamente para anomalias sequenciais e não-sequenciais.

**Cuidado que não pode ser perdido:** ao avaliar uma categoria, os positivos são apenas
as anomalias daquela categoria, mas os **negativos continuam sendo todos os blocos
normais**. Descartar os normais infla artificialmente a precisão e o número não significa
nada.

### 4.8 Múltiplas execuções com variância

Cada configuração roda várias vezes com amostragens distintas; reportar média e variância,
não o melhor resultado. Recomendação explícita de Landauer et al.

---

## 5. Classificação sequencial × não-sequencial

Procedimento em três etapas, em `src/labeling/`:

1. **Eventos distintivos** — para cada `Type`, eventos presentes em >50% dos blocos da
   categoria e em <5% dos normais. Indica anomalia **não-sequencial** (definida por
   presença).

2. **Teste de precedência** — para pares de eventos-núcleo
   (`E5, E9, E11, E21, E22, E23, E26`), compara P(A antes de B) na categoria contra o
   padrão dos blocos normais. Pares com ordem quase fixa nos normais (P ≥ 0,90 ou ≤ 0,10)
   formam a referência; divergência ≥ 0,25 conta como violação de ordem, indicando
   anomalia **sequencial**.

   Considerar apenas blocos em que **ambos** os eventos do par aparecem — senão a ausência
   de um evento (que é sinal não-sequencial) contamina a medida de ordem.

3. **Verificação de simultaneidade — OBRIGATÓRIA.** Landauer et al. demonstram que, no
   HDFS, eventos gerados simultaneamente são registrados em **ordem aleatória**: as três
   sequências *normais* mais comuns diferem apenas pela posição do evento 22 entre os
   quatro primeiros, que ocorrem no mesmo instante.

   Logo, toda violação detectada na etapa 2 deve ser confrontada com os carimbos de tempo
   do par. Intervalo consistentemente nulo ⇒ **artefato de registro**, não anomalia
   sequencial. Separação temporal clara ⇒ reordenação genuína.

   **Sem essa verificação, o resultado da etapa 2 não é confiável.**

   Limiares adotados (`config.py`): para cada violação, tomam-se os blocos da categoria
   com o par na ordem **invertida** e mede-se `|t(B) − t(A)|` entre as primeiras
   ocorrências (carimbos acumulados de `TimeInterval`). A violação é artefato quando
   ≥ 90% desses blocos têm intervalo ≤ 0 s, isto é, os dois eventos caem no mesmo
   segundo (a resolução do log é de 1 s). Nos dados reais o resultado não muda para
   intervalos de 0 a 2 s e frações de 0,5 a 1,0.

   Convenção comum às etapas 2 e 3: como os eventos-núcleo se repetem no bloco (um por
   réplica), "A antes de B" compara a **primeira** ocorrência de cada um.

### 5.4 Regra de classificação final

Os sinais das etapas 1 e 2 **não são mutuamente excludentes**: uma categoria pode
introduzir eventos raros e, ao mesmo tempo, reordenar eventos existentes. Colapsá-los
em uma dicotomia exigiria arbitrar qual prevalece, e qualquer arbitragem esvaziaria
uma das categorias — os três tipos com violação confirmada (0, 8 e 19) também possuem
eventos distintivos (`results/sequentiality_types.csv`).

**A porta de amostra é aplicada antes da regra.** Um tipo com menos de
`MIN_BLOCKS_PER_TYPE` = 10 blocos sai como `indeterminado` qualquer que seja a
combinação de sinais: frequências estimadas sobre poucos blocos não sustentam nenhuma
das outras três saídas. Nos dados reais a porta atua em 13 dos 26 tipos, e no Type 19
ela suprime uma combinação que, pelos sinais, seria "ambos os sinais" — 8 blocos, com
eventos distintivos e violação E21/E23 confirmada. Registrar isso importa porque o
Type 19 é o terceiro tipo com reordenação genuína, embora não apareça como tal na
tabela de classificação.

Para os tipos que passam a porta, a classificação é função dos dois sinais:

| Classificação | Sinal não-sequencial | Violação confirmada (etapa 3) |
|---|---|---|
| não-sequencial | sim | não |
| ambos os sinais | sim | sim |
| sequencial | não | sim |
| indeterminado | não | não |

**Sinal não-sequencial** reúne duas manifestações, ambas detectáveis por contagem e
independentes de ordem: a presença de evento distintivo (etapa 1) e o comprimento de
sequência fora do intervalo observado nos blocos normais. A inclusão do comprimento
segue Landauer, Skopik e Wurzenberger (2024), que identificam três manifestações de
anomalia no HDFS — novos tipos de evento, comprimento de sequência e contagens.

O intervalo de referência é `[mín, máx]` do comprimento nos blocos normais (13 a 298
eventos). Uma categoria tem sinal de comprimento quando ≥ `LENGTH_SIGNAL_MIN_FRACTION`
= 0,90 de seus blocos caem fora dele — o mesmo patamar de "consistente" usado nas
etapas 2 e 3, para que o sinal seja traço da categoria e não de uns poucos blocos. Nos
dados reais, só os Types 3 e 31 o têm (100% dos blocos fora); o 31 já tinha evento
distintivo (E7).

Sem o sinal de comprimento, o Type 3 (2.950 blocos, 17,5% das anomalias) não
apresentaria sinal algum: seus blocos têm exatamente dois eventos, `[E5,E22]` ou
`[E22,E5]` *(observação exploratória: 1.643 e 1.307 blocos)*, contra o mínimo de treze
observado nos blocos normais. Os dois eventos são os mais frequentes do conjunto, de modo
que nenhum deles é distintivo.

O teste de precedência também não alcança o Type 3, mas **não** por a sequência ser curta:
o par de eventos-núcleo E5/E22 está presente nos 2.950 blocos e tem suporte de sobra. O
que o exclui é a etapa de referência — se a ordem entre E5 e E22 não é fixa nos blocos
normais, o par nunca chega a ser candidato a violação. É o esperado, porque a permutação
por simultaneidade descrita por Landauer, Skopik e Wurzenberger (2024) atinge justamente
o evento 22; mas a afirmação precisa ser lida na coluna `is_reference` da tabela de
precedência, que hoje não é gravada em `results/` (seção 9). A variação de ordem entre as
duas formas do Type 3 é, ela mesma, permutação por simultaneidade *(observação
exploratória: 88% desses blocos têm E5 e E22 no mesmo segundo, intervalo máximo de 3 s)*,
e o critério corretamente não a classifica como violação.

O teste de precedência exige ainda suporte mínimo de `MIN_PAIR_SUPPORT` = 5 blocos da
categoria contendo ambos os eventos do par. Sem esse filtro, o Type 13 produziria
violação a partir de dois blocos, estatisticamente indistinguível de coincidência. O par
descartado continua na tabela do teste (`has_min_support = False`), para que o
descarte seja auditável.

---

## 6. Arquitetura

Cada camada conhece apenas as camadas abaixo:

```
experiments/  →  models/ + labeling/ + evaluation/  →  data/  →  config.py
```

- `config.py` — caminhos, constantes, hiperparâmetros. Sem número mágico espalhado.
- `data/` — lê arquivos, devolve estruturas. **Não sabe o que é anomalia.**
- `labeling/` — classifica anomalias. **Não sabe que existem modelos.**
- `models/` — treina e pontua. Interface única: `fit`, `score_samples`, `predict`.
- `evaluation/` — calcula métricas. **Não sabe qual modelo as gerou.**
- `experiments/` — única camada que amarra tudo. Cada script é reproduzível de ponta a ponta.

**Interface obrigatória dos modelos.** Todos expõem `fit(X)`, `score_samples(X)` (escore
contínuo, **maior = mais anômalo**) e `predict(X)` (1 = anomalia, 0 = normal). O
scikit-learn inverte essas convenções no Isolation Forest (-1 = outlier, escore menor =
mais anômalo); normalizar no wrapper, nunca vazar a convenção dele para cima.

---

## 7. Execução e dados

Desenvolvimento local com Claude Code; execução pesada no Google Colab (GPU para o LSTM).
Git é a ponte.

**Os arquivos de dados NÃO vão para o git.** `Event_traces.csv` tem 120 MB e estoura o
limite de 100 MB por arquivo do GitHub. Ficam no Google Drive e são montados no Colab.
`data/` está no `.gitignore`.

**Resultados (`results/*.csv`) VÃO para o git** — são pequenos e constituem o registro
histórico dos experimentos.

---

## 8. Convenções de código

- Python 3.10+; identificadores e nomes de função em inglês
- Docstrings e comentários em **português** (a monografia é em português)
- Comentário explica **por quê**, não o quê
- Sem notebooks para lógica: notebook só orquestra, a lógica vive em `src/`
- `RANDOM_SEED = 42` em tudo que amostra
- Toda decisão metodológica não óbvia leva comentário citando a fonte
- No código, seções deste documento são citadas como `(decisões, seção X)`

## 9. Reprodutibilidade

Landauer et al. recomendam — e a Seção 3.4 da monografia adota — publicar código, dados,
estratégia de amostragem, divisão treino/teste e parâmetros. Nenhum experimento depende
de estado manual: rodar o script do zero reproduz o resultado.

### 9.1 Procedência dos números deste documento

Cada número aqui é de um de dois tipos, e eles são marcados de forma diferente:

- **Resultado do pipeline** — produzido por código deste repositório. Aponta para o
  arquivo de `results/` que o contém, ou nomeia a função que o calcula. Verificável
  rodando o script.
- **Observação exploratória** — medida fora do pipeline, para entender a base. Vem
  marcada *(observação exploratória)* no ponto em que aparece.

Nenhuma afirmação deve ficar numa terceira categoria: **resultado apresentado como se
fosse do pipeline, sem código que o produza.** Era o caso da tabela de AUC da seção 4.3,
que descrevia uma varredura de `max_samples` jamais executada neste repositório — o
módulo `src/models/` ainda não existe. Um número nessa situação não é defensável na banca
e não deve entrar na monografia.

### 9.2 Pendências de persistência

`precedence_test` e `simultaneity_check` devolvem tabelas que este documento cita —
`is_reference`, `has_min_support`, `n_type_both`, `median_gap_seconds` —, mas
`experiments/run_sequentiality.py` grava apenas `sequentiality_types.csv` e
`block_categories.csv`. Enquanto isso, a auditabilidade prometida na seção 5.4 ("o par
descartado continua na tabela do teste, para que o descarte seja auditável") existe no
objeto em memória e não no repositório, e afirmações como "o Type 13 produziria violação
a partir de dois blocos" não têm arquivo que as sustente. As duas tabelas devem ir para
`results/`.

Falta também a verificação de redundância **bloco a bloco**: para cada violação
confirmada, a fração dos blocos invertidos que já é sinalizada por evento distintivo.
Hoje a redundância só é afirmável no agregado — os três tipos com violação confirmada
têm, todos, eventos distintivos (`results/sequentiality_types.csv`) —, e é essa
afirmação agregada, não a bloco a bloco, que o trabalho pode usar enquanto a
verificação não existir.


## 10. Limitações já registradas (não contornar silenciosamente)

- Os rótulos do HDFS_v1 vêm de **regras heurísticas construídas à mão** sobre vetores de
  contagem não ordenados, não de falhas confirmadas em produção. Conclusões sobre "a
  natureza das anomalias do HDFS" são, a rigor, conclusões sobre o que aquelas regras
  capturam.
- A permutação por simultaneidade compromete estruturalmente a capacidade do HDFS de
  avaliar detectores sequenciais.