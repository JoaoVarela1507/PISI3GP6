import streamlit as st
import pandas as pd

from Home import EXCEL_FILE

# Configurações da página
st.set_page_config(page_title="Filtros e Comparações de Modelos", page_icon="🔍", layout="wide")
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