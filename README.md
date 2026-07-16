# FaultSense — Pipeline de Manutenção Preditiva para Indústria 4.0

> Projeto avaliativo final do curso de **Desenvolvimento de IA para Análise Preditiva**

---

## Sobre o Projeto

Equipamentos industriais falham sem aviso. Cada parada não planejada gera custos com reparo, retrabalho e perda de produção. O **FaultSense** é um pipeline completo de machine learning que aprende os padrões operacionais de máquinas industriais e **prevê falhas antes que elas aconteçam**, permitindo que equipes de manutenção ajam de forma proativa.

O sistema classifica o estado de cada máquina como **operacional (0)** ou **em risco de falha (1)**, com base em variáveis como temperatura, rotação, torque e desgaste da ferramenta.

---

## Índice

- [Problema](#problema)
- [Dataset](#dataset)
- [Arquitetura do Pipeline](#arquitetura-do-pipeline)
- [Técnicas e Tecnologias](#técnicas-e-tecnologias)
- [Resultados](#resultados)
- [Como Executar](#como-executar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Melhorias Futuras](#melhorias-futuras)

---

## Problema

Na manufatura, a manutenção reativa (consertar após a quebra) é cara e ineficiente. A manutenção preventiva por calendário desperdiça recursos. A **manutenção preditiva** usa dados reais do equipamento para intervir apenas quando necessário.

**Desafio:** O dataset possui forte desbalanceamento de classes — apenas **~3,4% dos registros** correspondem a falhas reais. Um modelo ingênuo que sempre prevê "sem falha" teria 96,6% de acurácia e seria completamente inútil. O pipeline trata isso explicitamente com técnicas de balanceamento.

---

## Dataset

| Atributo | Detalhe |
|---|---|
| Fonte | `data/manutencao_preditiva.csv` |
| Registros | 10.000 linhas |
| Features | 14 colunas |
| Alvo | `falha_maquina` (binário: 0 ou 1) |
| Taxa de falha | ~3,4% |

**Variáveis preditoras utilizadas:**

| Coluna | Descrição |
|---|---|
| `temperatura_ar_k` | Temperatura do ar (Kelvin) |
| `temperatura_processo_k` | Temperatura do processo (Kelvin) |
| `velocidade_rotacao_rpm` | Velocidade de rotação (RPM) |
| `torque_nm` | Torque aplicado (Newton-metro) |
| `desgaste_ferramenta_min` | Desgaste acumulado da ferramenta (minutos) |
| `tipo` | Tipo do produto (L / M / H) |
| `potencia` | **Feature engenheirada:** `velocidade_rotacao_rpm × torque_nm` |

> As colunas de subtipo de falha (`falha_twf`, `falha_hdf`, etc.) foram removidas para evitar vazamento de dados (*data leakage*).

---

## Arquitetura do Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     FAULTSENSE PIPELINE                      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  FASE 1: EDA    │  Inspeção, histogramas, heatmap de correlação,
│  Análise        │  distribuição do alvo → identifica 500 nulos e
│  Exploratória   │  desbalanceamento de classes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FASE 2:        │  Remove duplicatas; imputa nulos com média
│  Limpeza        │  (temperatura) e mediana (rotação/torque);
│  dos Dados      │  mantém outliers (podem sinalizar falha)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FASE 3:        │  Cria feature "potencia" = RPM × Torque
│  Feature        │  (aproxima potência mecânica real do equipamento)
│  Engineering    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FASE 4:        │  One-hot encoding de variáveis categóricas;
│  Split &        │  divisão estratificada 80/20;
│  Balanceamento  │  SMOTE apenas no treino (7.729 amostras por classe)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FASE 5:        │  StandardScaler nas 6 colunas contínuas
│  Escalonamento  │  (somente para KNN; Árvore é invariante à escala)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FASE 6:        │  KNN: k ∈ {3, 5, 7}
│  Ajuste de      │  Árvore de Decisão: max_depth ∈ {3, 5, None}
│  Hiperparâmetros│  Avaliação por acurácia treino/teste
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FASE 7:        │  KNN (k=5): 90,90%
│  Avaliação      │  Árvore (depth=5): 91,20% ← vencedor
│  Final          │  Gráfico comparativo
└─────────────────┘
```

---

## Técnicas e Tecnologias

### Linguagem e Ambiente

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

### Bibliotecas

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação e análise do dataset |
| `numpy` | Operações numéricas |
| `scikit-learn` | Modelos ML, split, StandardScaler, métricas |
| `imbalanced-learn` | SMOTE para balanceamento de classes |
| `matplotlib` | Geração de gráficos |
| `seaborn` | Heatmap de correlação e visualizações estatísticas |

### Algoritmos de Machine Learning

**K-Nearest Neighbors (KNN)**
- Classificação por proximidade no espaço de features
- Sensível à escala → requer normalização (StandardScaler aplicado)
- Hiperparâmetro testado: número de vizinhos `k`

**Árvore de Decisão (Decision Tree)**
- Classificação por regras binárias sequenciais
- Invariante à escala → não requer normalização
- Hiperparâmetro testado: profundidade máxima `max_depth`

### Técnicas Aplicadas

| Técnica | Finalidade |
|---|---|
| Análise Exploratória (EDA) | Entender distribuições, correlações e qualidade dos dados |
| Imputação por média/mediana | Tratar valores ausentes sem descartar registros |
| Feature Engineering | Criar `potencia` para capturar relação física entre RPM e torque |
| One-Hot Encoding | Converter variável categórica `tipo` em colunas numéricas |
| Divisão estratificada | Preservar proporção de classes no treino e teste |
| SMOTE | Sobreamostrar a classe minoritária (falhas) sinteticamente |
| StandardScaler | Normalizar features contínuas para o KNN |
| Busca manual de hiperparâmetros | Avaliar overfitting/underfitting por curva treino × teste |

---

## Resultados

### Busca de Hiperparâmetros — KNN

| k | Acurácia Treino | Acurácia Teste |
|---|---|---|
| 3 | 97,1% | 91,4% |
| **5** | **96,1%** | **90,9%** |
| 7 | 95,4% | 90,2% |

> k=5 apresenta o menor gap entre treino e teste, indicando melhor generalização.

### Busca de Hiperparâmetros — Árvore de Decisão

| max_depth | Acurácia Treino | Acurácia Teste |
|---|---|---|
| 3 | 85,8% | 85,9% |
| **5** | **90,3%** | **91,2%** |
| None | 99,5% | 94,4%* |

> *max_depth=None apresenta overfitting severo (gap de ~5 p.p.).

### Comparativo Final

| Modelo | Acurácia no Teste |
|---|---|
| KNN (k=5) | 90,90% |
| **Árvore de Decisão (depth=5)** | **91,20%** |

**Modelo recomendado: Árvore de Decisão com `max_depth=5`**

A árvore captura melhor os padrões de falha por meio de regras de decisão não-lineares, enquanto o KNN é mais sensível à presença de outliers e à distribuição local dos dados. Além disso, a Árvore de Decisão é interpretável — as regras podem ser auditadas pela equipe de manutenção.

---

## Como Executar

### Pré-requisitos

- Python 3.9 ou superior
- `pip` para gerenciamento de pacotes

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/eduardaecher/pipeline-preditivo.git
cd pipeline-preditivo

# 2. Crie e ative um ambiente virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
```

### Execução

```bash
# Inicie o Jupyter Notebook
jupyter notebook notebook.ipynb
```

Abra o arquivo `notebook.ipynb` no navegador e execute as células sequencialmente (Kernel → Restart & Run All).

> **Atenção:** O notebook importa módulos de `src/`. Certifique-se de que o diretório raiz do projeto está no Python path, o que já ocorre automaticamente ao abrir o notebook a partir da raiz do projeto.

---

## Estrutura do Projeto

```
pipeline-preditivo/
│
├── data/
│   └── manutencao_preditiva.csv   # Dataset com 10.000 registros
│
├── src/
│   ├── analysis.py                # Inspeção e estatísticas descritivas
│   ├── cleaning.py                # Remoção de duplicatas e imputação
│   ├── feature.py                 # Engenharia de features (coluna potencia)
│   ├── training.py                # Split X/y, treino/teste e SMOTE
│   ├── scaler.py                  # Normalização com StandardScaler
│   ├── parameters.py              # Busca de hiperparâmetros KNN e Árvore
│   ├── results.py                 # Treinamento e avaliação final
│   └── visualization.py           # Geração de todos os gráficos
│
├── notebook.ipynb                 # Notebook principal com o pipeline completo
├── requirements.txt               # Dependências do projeto
└── README.md                      # Este arquivo
```

---

## Melhorias Futuras

- [ ] **Métricas além de acurácia:** Incluir F1-Score, Precision, Recall e AUC-ROC, que são mais adequadas para problemas com desbalanceamento de classes.
- [ ] **Matriz de confusão:** Visualizar falsos negativos (falhas não detectadas) e falsos positivos (alarmes desnecessários), que têm custos diferentes no contexto industrial.
- [ ] **Random Forest e Gradient Boosting:** Testar algoritmos de ensemble (XGBoost, LightGBM) que tendem a superar modelos simples neste tipo de tarefa.
- [ ] **Validação cruzada (k-fold):** Substituir a avaliação por hold-out único por cross-validation para uma estimativa mais robusta da performance.
- [ ] **Busca automatizada de hiperparâmetros:** Implementar `GridSearchCV` ou `RandomizedSearchCV` no lugar da busca manual.
- [ ] **Pipeline sklearn:** Encapsular os passos de pré-processamento e modelo em um `sklearn.pipeline.Pipeline` para evitar data leakage e facilitar deploy.
- [ ] **Exportação do modelo:** Serializar o modelo final com `joblib` ou `pickle` para uso em produção sem re-treinamento.
- [ ] **Interface de inferência:** Criar uma API REST (FastAPI) ou interface web simples para classificar novas leituras de sensores em tempo real.

---

## Autor

Desenvolvido como projeto avaliativo final do curso de **Desenvolvimento de IA para Análise Preditiva**.
