import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

def classificar_por_faixa(preco_corrigido):
    """Classifica smartphones por faixa de preço"""
    if preco_corrigido <= 300:
        return "Low-End"
    elif preco_corrigido <= 700:
        return "Mid-Range"
    else:
        return "High-End"

# Menu lateral
menu = st.sidebar.selectbox("📂 Menu", ["Início", "Análise Exploratória", "Análise por Faixas", "Filtros e Comparações"])

# Nome do arquivo
EXCEL_FILE = "datasets/pisi3basededados.xlsx"

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
            <li>📊 Analisar a evolução dos preços por marca e ano (com correção de inflação)</li>
            <li>📈 Acompanhar tendências do mercado de celulares por faixas de preço</li>
            <li>🎯 Explorar análises segmentadas por Low-End, Mid-Range e High-End</li>
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
    st.info("💡 **Todas as análises de preço utilizam valores corrigidos pela inflação através do índice Big Mac para maior precisão temporal!**")

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
        
        # Correção de inflação
        df['Preço Corrigido (USD)'] = df.apply(lambda row: corrigir_inflacao(row['Preço (USD)'], row['Ano']), axis=1)
        df['Preço Corrigido (R$)'] = df['Preço Corrigido (USD)'].apply(lambda x: round(x * cotacao_dolar, 2))
        
        # Classificação por faixa
        df['Faixa'] = df['Preço Corrigido (USD)'].apply(classificar_por_faixa)

        # Sempre usar preços corrigidos pela inflação
        coluna_preco = 'Preço Corrigido (R$)'
        
        st.subheader("💲 Evolução Média dos Preços por Marca (Corrigido por Inflação)")
        marcas_disponiveis = sorted(df['Marca'].dropna().unique())

        st.markdown("**Selecione as marcas que deseja visualizar:**")
        col1, col2, col3 = st.columns(3)
        marcas_selecionadas = []
        for i, marca in enumerate(marcas_disponiveis):
            with [col1, col2, col3][i % 3]:
                if st.checkbox(marca, value=True, key=f"preco_{marca}"):
                    marcas_selecionadas.append(marca)

        media_preco = df[df['Marca'].isin(marcas_selecionadas)].groupby(['Marca', 'Ano'])[coluna_preco].mean().reset_index()
        
        fig1 = px.line(media_preco, x='Ano', y=coluna_preco, color='Marca', markers=True,
                       title="Preço Médio (Corrigido por Inflação) por Marca ao Longo dos Anos")
        fig1.update_traces(line=dict(width=3))
        fig1.add_annotation(
            text="* Preços corrigidos usando índice Big Mac",
            xref="paper", yref="paper", x=0, y=-0.1, showarrow=False
        )
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

        st.subheader("💵 Evolução do Preço Médio por Sistema Operacional (Corrigido por Inflação)")
        media_preco_so = df.groupby(['Sistema Operacional', 'Ano'])[coluna_preco].mean().reset_index()
        fig_so = px.line(media_preco_so, x='Ano', y=coluna_preco, color='Sistema Operacional', markers=True,
                         title="💵 Preço Médio (Corrigido por Inflação) por Sistema Operacional e Ano")
        fig_so.update_traces(line=dict(width=2))
        st.plotly_chart(fig_so, use_container_width=True)

        st.subheader("📈 Variação Percentual do Preço (Corrigido por Inflação)")
        variacoes = []
        for marca in df['Marca'].dropna().unique():
            dados_marca = df[df['Marca'] == marca].groupby('Ano')[coluna_preco].mean().reset_index().sort_values('Ano')
            if len(dados_marca) > 1:
                preco_inicio = dados_marca.iloc[0][coluna_preco]
                preco_fim = dados_marca.iloc[-1][coluna_preco]
                perc = (preco_fim - preco_inicio) / preco_inicio * 100
                variacoes.append({
                    'Marca': marca,
                    'Ano Inicial': dados_marca.iloc[0]['Ano'],
                    'Ano Final': dados_marca.iloc[-1]['Ano'],
                    f'Preço Inicial (Corrigido)': f"R$ {preco_inicio:.2f}",
                    f'Preço Final (Corrigido)': f"R$ {preco_fim:.2f}",
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

        st.subheader("💵 Preço por Sistema Operacional (Corrigido por Inflação)")
        fig4 = px.box(df, x='Sistema Operacional', y=coluna_preco,
                      title='Distribuição de Preços por Sistema Operacional (Corrigido por Inflação)')
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("🔋 Capacidade de Bateria por Marca")
        fig5 = px.box(df, x='Marca', y='Bateria (mAh)',
                      title='Capacidade de Bateria por Marca')
        st.plotly_chart(fig5, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")

# =========================
# NOVA SEÇÃO: ANÁLISE POR FAIXAS
# =========================
elif menu == "Análise por Faixas":
    st.title("🎯 Análise por Faixas de Preço")
    st.info("💡 **Todas as análises desta seção utilizam preços corrigidos pela inflação através do índice Big Mac**")

    try:
        cotacao_dolar = 5.68
        df = pd.read_excel(EXCEL_FILE)
        df.columns = df.columns.str.strip()
        df['Preço (USD)'] = df['Launched Price (USA)'].str.replace("USD", "").str.replace(",", "").astype(float)
        df['Preço (R$)'] = df['Preço (USD)'].apply(lambda x: round(x * cotacao_dolar, 2))
        df['Ano'] = df['Launched Year']
        df['Marca'] = df['Company Name']
        df['Sistema Operacional'] = df['Sistema Operacional'].str.strip()
        df['Bateria (mAh)'] = df['Battery Capacity'].str.replace("mAh", "").str.replace(",", "").astype(float)
        
        # Correção de inflação e classificação
        df['Preço Corrigido (USD)'] = df.apply(lambda row: corrigir_inflacao(row['Preço (USD)'], row['Ano']), axis=1)
        df['Preço Corrigido (R$)'] = df['Preço Corrigido (USD)'].apply(lambda x: round(x * cotacao_dolar, 2))
        df['Faixa'] = df['Preço Corrigido (USD)'].apply(classificar_por_faixa)

        st.info("📊 **Faixas de Preço**: Low-End (≤$300), Mid-Range ($300-$700), High-End (>$700)")

        # Distribuição por faixas
        st.subheader("📊 Distribuição de Smartphones por Faixa de Preço")
        faixa_counts = df['Faixa'].value_counts().reset_index()
        faixa_counts.columns = ['Faixa', 'Quantidade']
        
        colors = {'Low-End': '#4CAF50', 'Mid-Range': '#2196F3', 'High-End': '#9C27B0'}
        fig_faixas = px.pie(faixa_counts, values='Quantidade', names='Faixa',
                           title='Distribuição de Smartphones por Faixa de Preço',
                           color='Faixa', color_discrete_map=colors)
        st.plotly_chart(fig_faixas, use_container_width=True)

        # Evolução das faixas ao longo do tempo
        st.subheader("📈 Evolução das Faixas de Preço ao Longo do Tempo")
        faixas_ano = df.groupby(['Ano', 'Faixa']).size().reset_index(name='Quantidade')
        fig_evolucao = px.bar(faixas_ano, x='Ano', y='Quantidade', color='Faixa',
                             title='Evolução do Número de Lançamentos por Faixa e Ano',
                             color_discrete_map=colors)
        st.plotly_chart(fig_evolucao, use_container_width=True)

        # Análise por marca e faixa
        st.subheader("🏢 Estratégia de Marcas por Faixa de Preço")
        marca_faixa = df.groupby(['Marca', 'Faixa']).size().unstack(fill_value=0)
        marca_faixa_pct = marca_faixa.div(marca_faixa.sum(axis=1), axis=0) * 100
        
        # Selecionar marcas para visualizar
        marcas_disponiveis = sorted(df['Marca'].dropna().unique())
        marcas_selecionadas = st.multiselect("Selecione as marcas:", marcas_disponiveis, 
                                           default=marcas_disponiveis[:5])
        
        if marcas_selecionadas:
            marca_faixa_filtrado = marca_faixa_pct.loc[marcas_selecionadas]
            fig_marca_faixa = px.bar(marca_faixa_filtrado.reset_index(), x='Marca', 
                                   y=['Low-End', 'Mid-Range', 'High-End'],
                                   title='Distribuição Percentual de Produtos por Marca e Faixa',
                                   color_discrete_map=colors)
            st.plotly_chart(fig_marca_faixa, use_container_width=True)

        # Características médias por faixa
        st.subheader("🔧 Características Médias por Faixa de Preço")
        
        # Preparar dados para características
        df['RAM (GB)'] = df['RAM'].str.extractall(r'(\d+)').groupby(level=0).max().astype(float)
        df['Camera (MP)'] = df['Back Camera'].str.extractall(r'(\d+)').groupby(level=0).max().astype(float)
        
        caracteristicas = df.groupby('Faixa').agg({
            'Preço Corrigido (R$)': 'mean',
            'Bateria (mAh)': 'mean',
            'RAM (GB)': 'mean',
            'Camera (MP)': 'mean'
        }).round(2)
        
        st.dataframe(caracteristicas)

        # Gráfico de radar comparando faixas
        st.subheader("🎯 Comparação de Características por Faixa (Radar)")
        
        # Normalizar dados para o radar (0-100)
        caracteristicas_norm = caracteristicas.copy()
        for col in caracteristicas_norm.columns:
            max_val = caracteristicas_norm[col].max()
            caracteristicas_norm[col] = (caracteristicas_norm[col] / max_val) * 100

        fig_radar = go.Figure()
        
        for faixa in caracteristicas_norm.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=caracteristicas_norm.loc[faixa].values.tolist() + [caracteristicas_norm.loc[faixa].values[0]],
                theta=list(caracteristicas_norm.columns) + [caracteristicas_norm.columns[0]],
                fill='toself',
                name=faixa,
                line_color=colors[faixa]
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title="Comparação de Características por Faixa (Normalizado 0-100)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Top performers por faixa
        st.subheader("🏆 Top 3 Modelos por Faixa de Preço")
        
        for faixa in ['Low-End', 'Mid-Range', 'High-End']:
            st.markdown(f"#### {faixa}")
            df_faixa = df[df['Faixa'] == faixa].copy()
            
            # Calcular score baseado em características
            df_faixa['Score'] = (
                (df_faixa['RAM (GB)'].fillna(0) * 10) +
                (df_faixa['Bateria (mAh)'].fillna(0) / 100) +
                (df_faixa['Camera (MP)'].fillna(0) * 2)
            )
            
            top_faixa = df_faixa.nlargest(3, 'Score')[['Model Name', 'Marca', 'Preço Corrigido (R$)', 
                                                     'RAM (GB)', 'Bateria (mAh)', 'Camera (MP)', 'Score']]
            st.dataframe(top_faixa)

        # Análise de tendências por faixa
        st.subheader("📊 Tendências de Preço Médio por Faixa ao Longo do Tempo")
        preco_medio_faixa = df.groupby(['Ano', 'Faixa'])['Preço Corrigido (R$)'].mean().reset_index()
        
        fig_tendencia = px.line(preco_medio_faixa, x='Ano', y='Preço Corrigido (R$)', color='Faixa',
                               title='Evolução do Preço Médio por Faixa (Corrigido por Inflação)',
                               color_discrete_map=colors, markers=True)
        st.plotly_chart(fig_tendencia, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")

# =========================
# FILTROS E COMPARAÇÕES COM MELHORIAS
# =========================
elif menu == "Filtros e Comparações":
    st.title("🔍 Filtros e Comparações de Modelos")
    st.info("💡 **Todas as análises de preço utilizam valores corrigidos pela inflação através do índice Big Mac**")

    try:
        df = pd.read_excel(EXCEL_FILE)
        df.columns = df.columns.str.strip()
        df['Preço'] = df['Launched Price (USA)'].str.replace("USD", "").str.replace(",", "").astype(float)
        df['Bateria (mAh)'] = df['Battery Capacity'].str.replace("mAh", "").str.replace(",", "").astype(float)
        df['RAM (GB)'] = df['RAM'].str.extractall(r'(\d+)').groupby(level=0).max().astype(float)
        df['Camera (MP)'] = df['Back Camera'].str.extractall(r'(\d+)').groupby(level=0).max().astype(float)
        df['Sistema Operacional'] = df['Sistema Operacional'].str.strip()
        df['Ano'] = df['Launched Year']

        # Conversão e correção de inflação para todos os preços
        cotacao_dolar = 5.68
        df['Preço Corrigido (USD)'] = df.apply(lambda row: corrigir_inflacao(row['Preço'], row['Ano']), axis=1)
        df['Preço (R$)'] = df['Preço'].apply(lambda x: f"R$ {x * cotacao_dolar:.2f}")
        df['Preço Corrigido (R$)'] = df['Preço Corrigido (USD)'].apply(lambda x: f"R$ {x * cotacao_dolar:.2f}")
        
        # Usar SEMPRE preços corrigidos nas análises principais
        df['Preço Principal (USD)'] = df['Preço Corrigido (USD)']
        df['Preço Principal (R$)'] = df['Preço Corrigido (R$)']

        # Cálculo de custo-benefício com preço corrigido
        df['Custo-Benefício'] = round(((df['RAM (GB)'].fillna(0) + df['Bateria (mAh)'].fillna(0) / 1000 + df['Camera (MP)'].fillna(0) / 10) / df['Preço Corrigido (USD)']) * 100, 2)

        # Classificação por nível com preço corrigido
        df['Faixa'] = df['Preço Corrigido (USD)'].apply(classificar_por_faixa)

        st.subheader("🎯 Buscar Modelo por Nome")
        nome_busca = st.text_input("Digite parte do nome:")
        if nome_busca:
            st.dataframe(df[df['Model Name'].str.contains(nome_busca, case=False)])

        st.markdown("---")
        st.subheader("💰 Filtrar por Orçamento (Preços Corrigidos pela Inflação)")

        preco_min_real = int((df['Preço Corrigido (USD)'] * cotacao_dolar).min())
        preco_max_real = int((df['Preço Corrigido (USD)'] * cotacao_dolar).max())

        orcamento = st.slider("Filtrar modelos com preço corrigido até (R$):",
                              min_value=preco_min_real,
                              max_value=preco_max_real,
                              value=preco_max_real)

        df_filtrado = df[df['Preço Corrigido (USD)'] * cotacao_dolar <= orcamento]
        st.dataframe(df_filtrado[['Model Name', 'Company Name', 'Preço Corrigido (R$)', 'Bateria (mAh)', 'RAM (GB)', 'Camera (MP)', 'Custo-Benefício', 'Faixa']])

        st.markdown("---")
        st.subheader("🏆 Top 5 Melhores Custo-Benefício")
        top_cb = df.sort_values('Custo-Benefício', ascending=False).head(5)
        st.dataframe(top_cb[['Model Name', 'Company Name', 'Preço Corrigido (R$)', 'RAM (GB)', 'Bateria (mAh)', 'Camera (MP)', 'Custo-Benefício', 'Faixa']])

        st.markdown("---")
        st.subheader("🔋 Top 10 por Maior Capacidade de Bateria")
        top_bateria = df.sort_values('Bateria (mAh)', ascending=False).head(10)
        st.dataframe(top_bateria[['Model Name', 'Company Name', 'Preço Corrigido (R$)', 'Bateria (mAh)', 'RAM (GB)', 'Faixa']])

        st.subheader("💵 Top 10 Mais Baratos")
        top_preco = df.sort_values('Preço Corrigido (USD)').head(10)
        st.dataframe(top_preco[['Model Name', 'Company Name', 'Preço Corrigido (R$)', 'RAM (GB)', 'Bateria (mAh)', 'Camera (MP)', 'Faixa']])

        st.markdown("---")
        st.subheader("📊 Comparar Dois Modelos")

        modelo_1 = st.selectbox("Modelo 1", sorted(df['Model Name'].unique()), key="modelo_1")
        modelo_2 = st.selectbox("Modelo 2", sorted(df['Model Name'].unique()), key="modelo_2")

        df_comp = df[df['Model Name'].isin([modelo_1, modelo_2])][[
            'Model Name', 'Company Name', 'Sistema Operacional', 'Preço Corrigido (R$)',
            'Bateria (mAh)', 'RAM (GB)', 'Camera (MP)', 'Front Camera', 'Back Camera',
            'Processor', 'Screen Size', 'Custo-Benefício', 'Faixa'
        ]]

        for _, row in df_comp.iterrows():
            st.markdown(f"#### 📱 {row['Model Name']}")
            st.table(pd.DataFrame(row).T)

        # Recomendação de modelos semelhantes (com preço corrigido)
        st.markdown("---")
        st.subheader("✨ Recomendação de Modelos Semelhantes")

        modelo_base = st.selectbox("Escolha um modelo como base para recomendação:", sorted(df['Model Name'].unique()), key="recomendacao")

        referencia = df[df['Model Name'] == modelo_base].iloc[0]
        so_ref = referencia['Sistema Operacional']
        faixa_ref = referencia['Faixa']
        preco_ref = referencia['Preço Corrigido (USD)']

        limite_inferior = preco_ref * 0.85
        limite_superior = preco_ref * 1.15

        recomendados = df[
            (df['Sistema Operacional'] == so_ref) &
            (df['Faixa'] == faixa_ref) &
            (df['Preço Corrigido (USD)'] >= limite_inferior) &
            (df['Preço Corrigido (USD)'] <= limite_superior) &
            (df['Model Name'] != modelo_base)
        ].sort_values('Custo-Benefício', ascending=False).head(3)

        if not recomendados.empty:
            st.markdown("#### 🔁 Modelos Recomendados:")
            for _, row in recomendados.iterrows():
                cor = {
                    "Low-End": "#4CAF50",
                    "Mid-Range": "#2196F3",
                    "High-End": "#9C27B0"
                }.get(row['Faixa'], "#999")

                st.markdown(f"""
                    <div style="border:1px solid #ccc; border-radius:10px; padding:10px; margin:10px 0;">
                        <h4 style="margin:0;">📱 <strong>{row['Model Name']}</strong></h4>
                        <p style="margin:0;">💼 Marca: {row['Company Name']}</p>
                        <p style="margin:0;">💰 Preço: {row['Preço Corrigido (R$)']} | 📊 Custo-Benefício: {row['Custo-Benefício']}</p>
                        <span style="display:inline-block; padding:4px 10px; background-color:{cor}; color:#fff; border-radius:5px; font-size:14px;">
                            {row['Faixa']}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Nenhum modelo semelhante encontrado com base no critério atual.")

    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
# =========================