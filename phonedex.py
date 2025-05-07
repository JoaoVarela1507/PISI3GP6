import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração da página
st.set_page_config(page_title="Análise PhoneDex", layout="wide")

# Estilo visual (vermelho e moderno)
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

# Menu lateral
menu = st.sidebar.selectbox("📂 Menu", ["Início", "Análise Exploratória"])

# Nome do arquivo Excel
EXCEL_FILE = "pisi3basededados.xlsx"

# TELA INICIAL
if menu == "Início":

    # Título e descrição
    st.markdown("""
        <h1 style='text-align: center;'>📱 PhoneDex - Pokedex de Smartphones</h1>
        <p style='text-align: center; font-size: 18px'>
            Bem-vindo ao <strong>PhoneDex</strong>, uma plataforma que organiza e apresenta dados sobre smartphones de forma inteligente.
        </p>
        <div style='margin-top: 30px; font-size: 17px;'>
            Aqui você poderá:
            <ul>
                <li>📊 Analisar a evolução dos preços por marca e ano</li>
                <li>📈 Acompanhar tendências do mercado de celulares</li>
                <li>📱 Explorar comparações entre modelos, acompanhar lançamentos e salvar favoritos <em>(em breve)</em></li>
            </ul>
            <p>Navegue pelo menu lateral para acessar os recursos disponíveis no momento.</p>
        </div>
    """, unsafe_allow_html=True)

# ANÁLISE EXPLORATÓRIA
elif menu == "Análise Exploratória":
    st.title("📊 Análise Exploratória de Preços de Smartphones")

    try:
        df = pd.read_excel(EXCEL_FILE)
        df.columns = df.columns.str.strip()

        df['Preço'] = df['Launched Price (USA)'].str.replace("USD", "").str.replace(",", "").astype(float)
        df['Ano'] = df['Launched Year']
        df['Marca'] = df['Company Name']

        media_preco = df.groupby(['Marca', 'Ano'])['Preço'].mean().reset_index()

        st.subheader("💲 Evolução Média dos Preços por Marca")
        fig = px.line(
            media_preco, x='Ano', y='Preço', color='Marca', markers=True,
            title="Preço Médio por Marca ao Longo dos Anos"
        )
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📈 Variação Percentual do Preço Inicial ao Final (por Marca)")
        variacoes = []

        for marca in media_preco['Marca'].unique():
            dados_marca = media_preco[media_preco['Marca'] == marca].sort_values('Ano')
            if len(dados_marca) > 1:
                preco_inicio = dados_marca.iloc[0]['Preço']
                preco_fim = dados_marca.iloc[-1]['Preço']
                variacao = preco_fim - preco_inicio
                perc = (variacao / preco_inicio) * 100
                variacoes.append({
                    'Marca': marca,
                    'Ano Inicial': dados_marca.iloc[0]['Ano'],
                    'Ano Final': dados_marca.iloc[-1]['Ano'],
                    'Preço Inicial ($)': round(preco_inicio, 2),
                    'Preço Final ($)': round(preco_fim, 2),
                    'Variação (%)': round(perc, 2),
                    'Tendência': '↑ Aumento' if perc > 0 else '↓ Queda'
                })

        df_variacoes = pd.DataFrame(variacoes)
        st.dataframe(df_variacoes)

    except FileNotFoundError:
        st.error(f"Arquivo '{EXCEL_FILE}' não encontrado. Certifique-se de que ele está na mesma pasta do script.")
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
