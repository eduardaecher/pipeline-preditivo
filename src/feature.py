#CRIANDO FUNÇÃO PARA GERAR NOVA COLUNA "POTENCIA"

def criar_feature_produto(df, coluna_a, coluna_b, nome_nova_coluna):
    """
    Cria uma nova coluna numérica a partir do produto entre duas colunas
    existentes (sendo: potencia = velocidade_rotacao_rpm * torque_nm).
    
    Valida que não há nulos nas colunas de origem antes de calcular.
    """
    assert df[[coluna_a, coluna_b]].isnull().sum().sum() == 0, (
        "Ainda há nulos nas colunas de origem! Trate antes de criar a feature."
    )
 
    df_novo = df.copy()
    df_novo[nome_nova_coluna] = df_novo[coluna_a] * df_novo[coluna_b]
    return df_novo