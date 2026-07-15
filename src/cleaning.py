#FUNÇÃO PARA EXCLUSÃO DE LINHAS DUPLICADAS

import pandas as pd

def remover_duplicados(df):

    """
    Identifica e remove linhas duplicadas de um DataFrame.
 
    Parâmetros:
        df: DataFrame do pandas
 
    Retorna:
        DataFrame sem duplicados (índice resetado)
    """
    duplicados = df.duplicated().sum()
    print(f"Linhas duplicadas encontradas: {duplicados}")
 
    df_tratado = df.drop_duplicates().reset_index(drop=True)
    print(f"Dimensões após remoção de duplicados: {df_tratado.shape}")
    return df_tratado


#FUNÇÃO PARA IDENTIFICAÇÃO DE DADOS AUSENTES

def identificar_ausentes(df, colunas=None):
    """
    Identifica a quantidade e o percentual de valores ausentes por coluna.

    """
    base = df[colunas] if colunas is not None else df
 
    print("Valores ausentes por coluna:")
    print(base.isnull().sum())
    print()
 
    resumo = pd.DataFrame({
        'ausentes': base.isnull().sum(),
        'percentual (%)': (base.isnull().mean() * 100).round(2)
    })
    return resumo

#FUNÇÃO PARA IMPUTAR MÉDIA E MEDIANA NOS VALORES AUSENTES

import numpy as np

def imputar_media(df, colunas):
    df_tratado = df.copy()

    for col in colunas:
        media = df_tratado[col].mean()
        df_tratado[col] = df_tratado[col].fillna(media)
        print(f"{col}: nulos preenchidos com a média ({media:.2f})")

    return df_tratado


def imputar_mediana(df, colunas):
    df_tratado = df.copy()

    for col in colunas:
        mediana = df_tratado[col].median()
        df_tratado[col] = df_tratado[col].fillna(mediana)
        print(f"{col}: nulos preenchidos com a mediana ({mediana:.2f})")

    return df_tratado