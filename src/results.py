#CRIANDO A FUNÇÃO QUE MOSTRA A ACURÁCIA FINAL DOS MODELOS

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def treinar_avaliar_knn_final(X_train, y_train, X_test, y_test, melhor_k):
    """
    Treina o KNN final com o melhor k e retorna o modelo e a acurácia no teste.
    """
    modelo = KNeighborsClassifier(n_neighbors=int(melhor_k))
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test))
    print(f"Acurácia final do melhor KNN (k={melhor_k}) no teste: {acc:.4f}")
    return modelo, acc

def treinar_avaliar_arvore_final(X_train, y_train, X_test, y_test, melhor_profundidade, random_state=42):
    """
    Treina a Árvore de Decisão final com a melhor profundidade e retorna
    o modelo e a acurácia no teste.
    """
    profundidade = None if melhor_profundidade == 'None (sem limite)' else int(melhor_profundidade)
    modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=random_state)
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test))
    print(f"Acurácia final da melhor Árvore (max_depth={melhor_profundidade}) no teste: {acc:.4f}")
    return modelo, acc