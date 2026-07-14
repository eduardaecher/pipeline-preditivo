#FUNÇÃO PARA CRIAÇÃO DE VISUALIZAÇÕES EM GRÁFICOS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#GRÁFICO 1 - HISTOGRAMA DAS VARIÁVEIS PREDITORAS

def gerar_histogramas(df, variaveis, bins=20, cor="steelblue", n_colunas=3):
    """Gera um histograma para cada variável numérica de um DataFrame."""
    n_vars = len(variaveis)
    n_linhas = int(np.ceil(n_vars / n_colunas))

    fig, axes = plt.subplots(n_linhas, n_colunas, figsize=(5 * n_colunas, 4 * n_linhas))
    axes = np.array(axes).reshape(-1)

    for i, var in enumerate(variaveis):
        ax = axes[i]
        ax.hist(df[var].dropna(), bins=bins, color=cor, edgecolor="black", alpha=0.8)
        ax.set_title(var)
        ax.set_xlabel(var)
        ax.set_ylabel("Frequência")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    for j in range(n_vars, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

#GRÁFICO 2 - GRÁFICO DE BARRAS - TAXA DE DESBALANCEAMENTO DA VARIÁVEL ALVO

import matplotlib.pyplot as plt
import seaborn as sns

def gerar_grafico_distribuicao_alvo(df, coluna_alvo, labels=None, 
                                     cores=None, titulo=None):
    """
    Gera um gráfico de barras mostrando a distribuição (contagem e percentual)
    de uma variável categórica/alvo, útil para evidenciar desbalanceamento.
    """
    contagem = df[coluna_alvo].value_counts()
    percentual = df[coluna_alvo].value_counts(normalize=True) * 100

    if cores is None:
        cores = ["#4C72B0", "#C44E52"]
    if titulo is None:
        titulo = f"Distribuição da variável alvo: {coluna_alvo}"

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=contagem.index, y=contagem.values, palette=cores)

    if labels is not None:
        ax.set_xticklabels(labels)

    ax.set_ylabel('Quantidade de registros')
    ax.set_title(titulo)

    for i, v in enumerate(contagem.values):
        ax.text(i, v + (contagem.values.max() * 0.01), 
                f"{v} ({percentual.values[i]:.2f}%)", 
                ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(contagem)
    print(percentual)


#GRÁFICO 3 - MAPA DE CALOR COM CORRELAÇÃO DE PEARSON

import matplotlib.pyplot as plt
import seaborn as sns

def gerar_heatmap_correlacao(df, colunas, metodo='pearson', titulo=None, 
                               figsize=(8, 6), cmap="coolwarm"):
    """
    Gera um mapa de calor com a matriz de correlação entre as colunas informadas.
    """
    matriz_corr = df[colunas].corr(method=metodo)

    if titulo is None:
        titulo = f"Correlação de {metodo.capitalize()} entre variáveis"

    plt.figure(figsize=figsize)
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap=cmap, 
                vmin=-1, vmax=1, square=True)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

    return matriz_corr