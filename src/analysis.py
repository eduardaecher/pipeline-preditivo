#FUNÇÃO PARA INSPEÇÃO DOS DADOS

import pandas as pd
import numpy as np

def inspecionar_dados(df, colunas_numericas=None):
    """
    Exibe informações básicas do DataFrame.

    Parâmetros:
        df: DataFrame do pandas
        colunas_numericas: lista de colunas para o resumo estatístico.
                            Se None, usa todas as colunas numéricas do df.
    """
    print(f"Dimensões do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print()
    print("Tipos de dados:")
    print(df.dtypes)
    print()
    print("Resumo estatístico (colunas numéricas):")

    if colunas_numericas is not None:
        base = df[colunas_numericas]
    else:
        base = df.select_dtypes(include='number')

    resumo = base.describe()
    print(resumo)
    return resumo