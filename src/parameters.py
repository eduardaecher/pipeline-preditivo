#CRIANDO FUNÇÃO PARA TESTAR O KNN COM AJUSTE DE PARÂMETROS

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def testar_knn(X_train, y_train, X_test, y_test, valores_k):
    """
    Treina o KNN variando n_neighbors e registra a acurácia de treino e teste.

    """
    resultados = []
 
    for k in valores_k:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
 
        acc_treino = accuracy_score(y_train, knn.predict(X_train))
        acc_teste = accuracy_score(y_test, knn.predict(X_test))
 
        resultados.append({'k': k, 'acuracia_treino': acc_treino, 'acuracia_teste': acc_teste})
 
    return pd.DataFrame(resultados)

#CRIANDO A FUNÇÃO PARA TESTAR ARVORE DE DECISÃO COM AJUSTE DE PARÂMETROS

def testar_arvore(X_train, y_train, X_test, y_test, valores_profundidade, random_state=42):
    """
    Treina a Árvore de Decisão variando max_depth e registra a acurácia
    de treino e teste.

    """
    resultados = []
 
    for profundidade in valores_profundidade:
        arvore = DecisionTreeClassifier(max_depth=profundidade, random_state=random_state)
        arvore.fit(X_train, y_train)
 
        acc_treino = accuracy_score(y_train, arvore.predict(X_train))
        acc_teste = accuracy_score(y_test, arvore.predict(X_test))
 
        resultados.append({
            'max_depth': profundidade if profundidade is not None else 'None (sem limite)',
            'acuracia_treino': acc_treino,
            'acuracia_teste': acc_teste
        })
 
    return pd.DataFrame(resultados)