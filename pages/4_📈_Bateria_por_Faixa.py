import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from utils import dataframe_convert_columns
from Home import corrigir_inflacao, classificar_por_faixareal

st.set_page_config(page_title="📈 Evolução da Bateria por Faixa", layout="wide")
st.title("🔋 Evolução da Capacidade de Bateria por Faixa de Preço")

# Carregar e tratar dados
df = pd.read_excel("datasets/pisi3basededados.xlsx")
df = dataframe_convert_columns(df)

# Diagnóstico de preço e faixa
df["Preco Corrigido"] = df.apply(lambda row: corrigir_inflacao(row["Preço (R$)"], row["Ano"]), axis=1)
df["Faixa"] = df["Preco Corrigido"].apply(classificar_por_faixareal)


# Agrupamento
df["Faixa"] = df["Faixa"].astype(str).str.strip()
media_bateria = df.groupby(["Faixa", "Ano"])["Bateria (mAh)"].mean().reset_index()

# Obter faixas disponíveis
faixas_disponiveis = sorted(df["Faixa"].dropna().unique())
faixas_selecionadas = st.multiselect(
    "Selecione as faixas de preço para exibir no gráfico:",
    options=faixas_disponiveis,
    default=faixas_disponiveis
)

# Gráfico 1 - Evolução
st.subheader("📈 Evolução da Capacidade de Bateria por Faixa de Preço")
fig, ax = plt.subplots(figsize=(10, 6))

for faixa in faixas_selecionadas:
    dados_faixa = media_bateria[media_bateria["Faixa"] == faixa]
    if not dados_faixa.empty:
        ax.plot(dados_faixa["Ano"], dados_faixa["Bateria (mAh)"], marker='o', label=faixa)

ax.set_title("Capacidade Média de Bateria por Faixa ao Longo dos Anos")
ax.set_xlabel("Ano")
ax.set_ylabel("Bateria Média (mAh)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Slider
anos_disponiveis = sorted(df["Ano"].dropna().unique())
ano_selecionado = st.slider("Selecione o ano para análise detalhada:", min_value=int(min(anos_disponiveis)),
                            max_value=int(max(anos_disponiveis)), value=int(max(anos_disponiveis)))
df_ano = df[df["Ano"] == ano_selecionado]

# Gráfico 2 - Média por faixa
st.subheader(f"🔋 Média de Bateria por Faixa ({ano_selecionado})")
media_por_faixa = df_ano.groupby("Faixa")["Bateria (mAh)"].mean().reset_index()

fig_barra = px.bar(media_por_faixa, x="Faixa", y="Bateria (mAh)", color="Faixa",
                   title="Média de Bateria por Faixa de Preço")
st.plotly_chart(fig_barra, use_container_width=True)

# Gráfico 3 - Boxplot
st.subheader(f"📦 Distribuição de Bateria por Faixa ({ano_selecionado})")
fig_box = px.box(df_ano, x="Faixa", y="Bateria (mAh)", color="Faixa",
                 title="Distribuição da Capacidade de Bateria por Faixa de Preço")
st.plotly_chart(fig_box, use_container_width=True)


st.markdown("""
### 💡 Observações:
- Verifique os preços e faixas com base nos dados reais.
- Se apenas High-End aparecer, os preços dos outros modelos podem estar acima do esperado.
""")
