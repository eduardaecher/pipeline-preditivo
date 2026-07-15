#CRIANDO FUNÇÃO PARA ESCALONAMENTO DE VARIÁVEIS

from sklearn.preprocessing import StandardScaler

def escalonar_para_knn(X_train, X_test, colunas_continuas):
    """
    Aplica StandardScaler nas colunas contínuas informadas, usando
    fit_transform no treino e transform no teste.
    As demais colunas de X_train/X_test permanecem inalteradas.

    """
    scaler = StandardScaler()
 
    X_train_escalonado = X_train.copy()
    X_test_escalonado = X_test.copy()
 
    X_train_escalonado[colunas_continuas] = scaler.fit_transform(X_train[colunas_continuas])
    X_test_escalonado[colunas_continuas] = scaler.transform(X_test[colunas_continuas])
 
    return X_train_escalonado, X_test_escalonado, scaler