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
menu = st.sidebar.selectbox("📂 Menu", ["Início", "Análise Exploratória", "Filtros e Comparações"])

# Nome do arquivo
EXCEL_FILE = "pisi3basededados.xlsx"

# =========================
# INÍCIO
# =========================
if menu == "Início":
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
        Navegue pelo menu lateral para acessar os recursos disponíveis no momento.
        </div>
    """, unsafe_allow_html=True)

# =========================
# ANÁLISE EXPLORATÓRIA
# =========================
elif menu == "Análise Exploratória":
    st.title("📊 Análise Exploratória de Smartphones")

    try:
        cotacao_dolar = 5.68  # Conversão atualizada
        df = pd.read_excel(EXCEL_FILE)
        df.columns = df.columns.str.strip()
        df['Preço (USD)'] = df['Launched Price (USA)'].str.replace("USD", "").str.replace(",", "").astype(float)
        df['Preço (R$)'] = df['Preço (USD)'].apply(lambda x: round(x * cotacao_dolar, 2))
        df['Ano'] = df['Launched Year']
        df['Marca'] = df['Company Name']
        df['Sistema Operacional'] = df['Sistema Operacional'].str.strip()
        df['Bateria (mAh)'] = df['Battery Capacity'].str.replace("mAh", "").str.replace(",", "").astype(float)

        st.subheader("💲 Evolução Média dos Preços por Marca")
        marcas_disponiveis = sorted(df['Marca'].dropna().unique())

        st.markdown("**Selecione as marcas que deseja visualizar:**")
        col1, col2, col3 = st.columns(3)
        marcas_selecionadas = []
        for i, marca in enumerate(marcas_disponiveis):
            with [col1, col2, col3][i % 3]:
                if st.checkbox(marca, value=True, key=f"preco_{marca}"):
                    marcas_selecionadas.append(marca)

        media_preco = df[df['Marca'].isin(marcas_selecionadas)].groupby(['Marca', 'Ano'])['Preço (R$)'].mean().reset_index()
        fig1 = px.line(media_preco, x='Ano', y='Preço (R$)', color='Marca', markers=True,
                       title="Preço Médio (R$) por Marca ao Longo dos Anos")
        fig1.update_traces(line=dict(width=3))
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📈 Evolução da Capacidade Média de Bateria por Marca")
        st.markdown("**Selecione as marcas que deseja visualizar:**")
        col4, col5, col6 = st.columns(3)
        marcas_bateria_selecionadas = []
        for i, marca in enumerate(marcas_disponiveis):
            with [col4, col5, col6][i % 3]:
                if st.checkbox(marca, value=True, key=f"bateria_{marca}"):
                    marcas_bateria_selecionadas.append(marca)

        media_bateria = df[df['Marca'].isin(marcas_bateria_selecionadas)].groupby(['Marca', 'Ano'])['Bateria (mAh)'].mean().reset_index()
        fig_bateria = px.line(media_bateria, x='Ano', y='Bateria (mAh)', color='Marca', markers=True,
                              title="📈 Bateria Média (mAh) por Marca e Ano")
        fig_bateria.update_traces(line=dict(width=2))
        st.plotly_chart(fig_bateria, use_container_width=True)

        st.subheader("💵 Evolução do Preço Médio por Sistema Operacional")
        media_preco_so = df.groupby(['Sistema Operacional', 'Ano'])['Preço (R$)'].mean().reset_index()
        fig_so = px.line(media_preco_so, x='Ano', y='Preço (R$)', color='Sistema Operacional', markers=True,
                         title="💵 Preço Médio (R$) por Sistema Operacional e Ano")
        fig_so.update_traces(line=dict(width=2))
        st.plotly_chart(fig_so, use_container_width=True)

        st.subheader("📈 Variação Percentual do Preço Inicial ao Final (por Marca)")
        variacoes = []
        for marca in df['Marca'].dropna().unique():
            dados_marca = df[df['Marca'] == marca].groupby('Ano')['Preço (R$)'].mean().reset_index().sort_values('Ano')
            if len(dados_marca) > 1:
                preco_inicio = dados_marca.iloc[0]['Preço (R$)']
                preco_fim = dados_marca.iloc[-1]['Preço (R$)']
                perc = (preco_fim - preco_inicio) / preco_inicio * 100
                variacoes.append({
                    'Marca': marca,
                    'Ano Inicial': dados_marca.iloc[0]['Ano'],
                    'Ano Final': dados_marca.iloc[-1]['Ano'],
                    'Preço Inicial (R$)': round(preco_inicio, 2),
                    'Preço Final (R$)': round(preco_fim, 2),
                    'Variação (%)': round(perc, 2),
                    'Tendência': '↑ Aumento' if perc > 0 else '↓ Queda'
                })
        st.dataframe(pd.DataFrame(variacoes))

        st.subheader("📱 Distribuição de Sistemas Operacionais")
        so_counts = df['Sistema Operacional'].value_counts().reset_index()
        so_counts.columns = ['Sistema Operacional', 'Quantidade']
        fig2 = px.pie(so_counts, values='Quantidade', names='Sistema Operacional',
                      title='Distribuição de Sistemas Operacionais')
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📆 Lançamentos por Ano")
        lanc_ano = df['Ano'].value_counts().sort_index().reset_index()
        lanc_ano.columns = ['Ano', 'Quantidade']
        fig3 = px.bar(lanc_ano, x='Ano', y='Quantidade',
                      title='Quantidade de Smartphones Lançados por Ano')
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("💵 Preço por Sistema Operacional (em R$)")
        fig4 = px.box(df, x='Sistema Operacional', y='Preço (R$)',
                      title='Distribuição de Preços por Sistema Operacional (R$)')
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("🔋 Capacidade de Bateria por Marca")
        fig5 = px.box(df, x='Marca', y='Bateria (mAh)',
                      title='Capacidade de Bateria por Marca')
        st.plotly_chart(fig5, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")



# =========================
# FILTROS E COMPARAÇÕES COM MELHORIAS
# =========================
elif menu == "Filtros e Comparações":
    st.title("🔍 Filtros e Comparações de Modelos")

    try:
        df = pd.read_excel(EXCEL_FILE)
        df.columns = df.columns.str.strip()
        df['Preço'] = df['Launched Price (USA)'].str.replace("USD", "").str.replace(",", "").astype(float)
        df['Bateria (mAh)'] = df['Battery Capacity'].str.replace("mAh", "").str.replace(",", "").astype(float)
        df['RAM (GB)'] = df['RAM'].str.extractall(r'(\d+)').groupby(level=0).max().astype(float)
        df['Camera (MP)'] = df['Back Camera'].str.extractall(r'(\d+)').groupby(level=0).max().astype(float)
        df['Sistema Operacional'] = df['Sistema Operacional'].str.strip()

        # Conversão do preço para reais
        cotacao_dolar = 5.68
        df['Preço (R$)'] = df['Preço'].apply(lambda x: f"R$ {x * cotacao_dolar:.2f}")

        # Cálculo de custo-benefício
        df['Custo-Benefício'] = round(((df['RAM (GB)'] + df['Bateria (mAh)'] / 1000 + df['Camera (MP)'] / 10) / df['Preço']) * 100, 2)

        # Classificação por nível
        df['Nível'] = pd.cut(df['Preço'] * cotacao_dolar,
                             bins=[0, 1000, 2000, float('inf')],
                             labels=['Básico', 'Intermediário', 'Top de Linha'])

        st.subheader("🎯 Buscar Modelo por Nome")
        nome_busca = st.text_input("Digite parte do nome:")
        if nome_busca:
            st.dataframe(df[df['Model Name'].str.contains(nome_busca, case=False)])

        st.markdown("---")
        st.subheader("💰 Filtrar por Orçamento")

        preco_min_real = int((df['Preço'] * cotacao_dolar).min())
        preco_max_real = int((df['Preço'] * cotacao_dolar).max())

        orcamento = st.slider("Filtrar modelos com preço até (R$):",
                              min_value=preco_min_real,
                              max_value=preco_max_real,
                              value=preco_max_real)

        df_filtrado = df[df['Preço'] * cotacao_dolar <= orcamento]
        st.dataframe(df_filtrado[['Model Name', 'Company Name', 'Preço (R$)', 'Bateria (mAh)', 'RAM', 'Camera (MP)', 'Custo-Benefício', 'Nível']])

        st.markdown("---")
        st.subheader("🏆 Top 5 Melhores Custo-Benefício")
        top_cb = df.sort_values('Custo-Benefício', ascending=False).head(5)
        st.dataframe(top_cb[['Model Name', 'Company Name', 'Preço (R$)', 'RAM', 'Bateria (mAh)', 'Camera (MP)', 'Custo-Benefício', 'Nível']])

        st.markdown("---")
        st.subheader("🔋 Top 10 por Maior Capacidade de Bateria")
        top_bateria = df.sort_values('Bateria (mAh)', ascending=False).head(10)
        st.dataframe(top_bateria[['Model Name', 'Company Name', 'Preço (R$)', 'Bateria (mAh)', 'RAM', 'Nível']])

        st.subheader("💵 Top 10 Mais Baratos")
        top_preco = df.sort_values('Preço').head(10)
        st.dataframe(top_preco[['Model Name', 'Company Name', 'Preço (R$)', 'RAM', 'Bateria (mAh)', 'Camera (MP)', 'Nível']])

        st.markdown("---")
        st.subheader("📊 Comparar Dois Modelos")

        modelo_1 = st.selectbox("Modelo 1", sorted(df['Model Name'].unique()), key="modelo_1")
        modelo_2 = st.selectbox("Modelo 2", sorted(df['Model Name'].unique()), key="modelo_2")

        df_comp = df[df['Model Name'].isin([modelo_1, modelo_2])][[
            'Model Name', 'Company Name', 'Sistema Operacional', 'Preço (R$)',
            'Bateria (mAh)', 'RAM', 'Camera (MP)', 'Front Camera', 'Back Camera',
            'Processor', 'Screen Size', 'Custo-Benefício', 'Nível'
        ]]

        for _, row in df_comp.iterrows():
            st.markdown(f"#### 📱 {row['Model Name']}")
            st.table(pd.DataFrame(row).T)

        # Recomendação de modelos semelhantes
        st.markdown("---")
        st.subheader("✨ Recomendação de Modelos Semelhantes")

        modelo_base = st.selectbox("Escolha um modelo como base para recomendação:", sorted(df['Model Name'].unique()), key="recomendacao")

        referencia = df[df['Model Name'] == modelo_base].iloc[0]
        so_ref = referencia['Sistema Operacional']
        nivel_ref = referencia['Nível']
        preco_ref = referencia['Preço']

        limite_inferior = preco_ref * 0.85
        limite_superior = preco_ref * 1.15

        recomendados = df[
            (df['Sistema Operacional'] == so_ref) &
            (df['Nível'] == nivel_ref) &
            (df['Preço'] >= limite_inferior) &
            (df['Preço'] <= limite_superior) &
            (df['Model Name'] != modelo_base)
        ].sort_values('Custo-Benefício', ascending=False).head(3)

        if not recomendados.empty:
            st.markdown("#### 🔁 Modelos Recomendados:")
            for _, row in recomendados.iterrows():
                cor = {
                    "Básico": "#4CAF50",
                    "Intermediário": "#2196F3",
                    "Top de Linha": "#9C27B0"
                }.get(row['Nível'], "#999")

                st.markdown(f"""
                    <div style="border:1px solid #ccc; border-radius:10px; padding:10px; margin:10px 0;">
                        <h4 style="margin:0;">📱 <strong>{row['Model Name']}</strong></h4>
                        <p style="margin:0;">💼 Marca: {row['Company Name']}</p>
                        <p style="margin:0;">💰 Preço: {row['Preço (R$)']} | 📊 Custo-Benefício: {row['Custo-Benefício']}</p>
                        <span style="display:inline-block; padding:4px 10px; background-color:{cor}; color:#fff; border-radius:5px; font-size:14px;">
                            {row['Nível']}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Nenhum modelo semelhante encontrado com base no critério atual.")

    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
# =========================

