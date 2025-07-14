import streamlit as st

EXCEL_FILE = "datasets/pisi3basededados.xlsx"

# Dados do Índice Big Mac para correção de inflação (aproximados)
BIG_MAC_INDEX = {
    2010: 3.71, 2011: 4.07, 2012: 4.33, 2013: 4.56, 2014: 4.80,
    2015: 4.79, 2016: 5.04, 2017: 5.30, 2018: 5.51, 2019: 5.74,
    2020: 5.71, 2021: 5.81, 2022: 5.15, 2023: 5.58, 2024: 5.69
}

def corrigir_inflacao(preco, ano_origem, ano_base=2024):
    """Corrige inflação usando o índice Big Mac"""
    if ano_origem in BIG_MAC_INDEX and ano_base in BIG_MAC_INDEX:
        fator_correcao = BIG_MAC_INDEX[ano_base] / BIG_MAC_INDEX[ano_origem]
        return preco * fator_correcao
    return preco

def classificar_por_faixareal(preco_corrigido):
    if preco_corrigido <= 1500:
        return "Low-End"
    elif preco_corrigido <= 3000:
        return "Mid-Range"
    else:
        return "High-End"
    
def classificar_por_faixa(preco_usd_corrigido):
    if preco_usd_corrigido <= 300:
        return "Low-End"
    elif preco_usd_corrigido <= 700:
        return "Mid-Range"
    else:
        return "High-End"


# Configuração da página
st.set_page_config(page_title="PhoneDex", layout="wide")

# Estilo global
st.markdown("""
    <style>
        .main { background-color: #fff; }
        .block-container { padding-top: 2rem; }
        h1, h2, h3, h4 { color: #b30000; }
        .stDataFrame, .stPlotlyChart {
            border: 1px solid #b30000;
            border-radius: 8px;
            padding: 1rem;
        }
        ul { padding-left: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📱 PhoneDex - Pokedex de Smartphones")

st.markdown("""
    <div style='text-align: justify; font-size: 18px'>
    Bem-vindo ao <strong>PhoneDex</strong>, uma plataforma que organiza e apresenta dados sobre smartphones de forma inteligente.<br><br>
    Aqui você poderá:
    <ul>
        <li>📊 Analisar a evolução dos preços por marca e ano</li>
        <li>📈 Acompanhar tendências do mercado de celulares</li>
        <li>📱 Explorar comparações entre modelos, acompanhar lançamentos e salvar favoritos <em>(em breve)</em></li>
    </ul>
    Navegue pelo menu no topo ou lateral (dependendo da tela) para acessar os recursos disponíveis.
    </div>
""", unsafe_allow_html=True)