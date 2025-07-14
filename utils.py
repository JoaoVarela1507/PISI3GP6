import numpy as np

COTACAO_DOLAR = 5.68

def clean_numeric_column(series, patterns_to_remove=None):
    """
    Limpa uma série removendo padrões específicos e convertendo para float.
    
    Args:
        series: Série pandas para limpar
        patterns_to_remove: Lista de tuplas (padrão, is_regex) para remover
        
    Returns:
        Série pandas limpa e convertida para float
    """
    if patterns_to_remove is None:
        patterns_to_remove = []
    
    result = series.astype(str).str.strip()
    
    for pattern, is_regex in patterns_to_remove:
        result = result.str.replace(pattern, "", regex=is_regex)
    
    return result.astype(float)


def clean_camera_column(series):
    """
    Limpa colunas de câmera removendo unidades e informações extras.
    
    Args:
        series: Série pandas da câmera
        
    Returns:
        Série pandas limpa
    """
    camera_patterns = [
        ("MP", False),
        ("Dual ", False),
        (r"/.*", True),     # Remove tudo após "/"
        (r"\+.*", True),    # Remove tudo após "+"
        (r",.*", True),     # Remove tudo após ","
        (r"\(.*?\)", True)  # Remove tudo entre "()"
    ]
    
    return clean_numeric_column(series, camera_patterns)


def dataframe_convert_columns(df):
    """
    Converte e limpa colunas do dataframe de smartphones.
    
    Args:
        df: DataFrame com dados dos smartphones
        
    Returns:
        DataFrame com colunas convertidas e limpas
    """
    # Mapeamento de transformações simples
    simple_transformations = {
        "Ano": "Launched Year",
        "Marca": "Company Name"
    }
    
    # Aplica transformações simples
    for new_col, original_col in simple_transformations.items():
        df[new_col] = df[original_col]
    
    # Conversão de preço USD para float
    df["Preço (USD)"] = clean_numeric_column(
        df["Launched Price (USA)"], 
        [("USD", False), (",", False)]
    )
    
    # Conversão de preço para reais
    df["Preço (R$)"] = df["Preço (USD)"] * COTACAO_DOLAR
    
    # Sistema operacional
    df["Sistema Operacional"] = df["Sistema Operacional"].astype(str).str.strip()
    df["Sistema Operacional (Binário)"] = np.where(
        df["Sistema Operacional"].str.lower().str.contains("android"), 0, 1
    )
    
    # Peso
    df["Peso (g)"] = clean_numeric_column(df["Mobile Weight"], [("g", False)])
    
    # Memória interna
    df["Memoria Interna (GB)"] = clean_numeric_column(
        df["RAM"], 
        [("GB", False), (r"/.*", True)]
    )
    
    # Câmeras
    df["Câmera Frontal (MP)"] = clean_camera_column(df["Front Camera"])
    df["Câmera Traseira (MP)"] = clean_camera_column(df["Back Camera"])
    
    # Tela
    df["Tela (polegadas)"] = clean_numeric_column(
        df["Screen Size"],
        [(" inches", False), (r",.*", True), (r"\(.*?\)", True)]
    )
    
    # Bateria
    df["Bateria (mAh)"] = clean_numeric_column(
        df["Battery Capacity"],
        [("mAh", False), (",", False)]
    )
    
    return df