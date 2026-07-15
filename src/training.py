#CRIANDO FUNÇÃO DE SEPARAÇÃO DE VARIÁVEIS

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preparar_x_y(df, colunas_remover, coluna_alvo, colunas_categoricas=None):
    """
    Remove colunas de vazamento/identificação, aplica one-hot encoding
    nas colunas categóricas e separa X (preditoras) de y (alvo).
    """
    df_modelo = df.drop(columns=colunas_remover)
 
    if colunas_categoricas:
        df_modelo = pd.get_dummies(df_modelo, columns=colunas_categoricas, drop_first=True)
 
    X = df_modelo.drop(columns=[coluna_alvo])
    y = df_modelo[coluna_alvo]
 
    print("Colunas de X:", list(X.columns))
    print("Distribuição de y:")
    print(y.value_counts())
 
    return X, y


#CRIANDO FUNÇÃO PARA DIVIDIR DADOS DE TREINO/TESTE

def dividir_treino_teste(X, y, test_size=0.2, random_state=42):
    """
    Divide os dados em treino e teste.

    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
 
    print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    print("Proporção da classe positiva no treino:", round(y_train.mean() * 100, 2), "%")
    print("Proporção da classe positiva no teste: ", round(y_test.mean() * 100, 2), "%")
 
    return X_train, X_test, y_train, y_test


#CRIANDO FUNÇÃO QUE APLICA REAMOSTRAGEM (SMOTE)

def aplicar_smote(X_train, y_train, random_state=42):
    """
    Aplica SMOTE (reamostragem) exclusivamente no conjunto de treino,
    para evitar vazamento de dados (data leakage).
    """
    print("Distribuição do treino ANTES do SMOTE:")
    print(y_train.value_counts())
 
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
 
    print()
    print("Distribuição do treino DEPOIS do SMOTE:")
    print(y_train_bal.value_counts())
 
    return X_train_bal, y_train_bal