import streamlit as st

EXCEL_FILE = "datasets/pisi3basededados.xlsx"

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