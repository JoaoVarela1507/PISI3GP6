import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Home import corrigir_inflacao, classificar_por_faixa

st.set_page_config(page_title="📈 Evolução da Bateria por Faixa", layout="wide")
st.title("🔋 Evolução da Capacidade de Bateria por Faixa de Preço")

# Carregar dados
df = pd.read_excel("datasets/pisi3basededados.xlsx")

# Corrigir inflação e classificar por faixa
df["Preco Corrigido"] = df.apply(lambda row: corrigir_inflacao(row["Preco (US$)"], row["Ano"]), axis=1)
df["Faixa"] = df["Preco Corrigido"].apply(classificar_por_faixa)

# Agrupar média da bateria por ano e faixa
media_bateria = df.groupby(["Ano", "Faixa"])["Bateria (mAh)"].mean().reset_index()

# Plotar gráfico de linha com Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
for faixa in media_bateria["Faixa"].unique():
    dados_faixa = media_bateria[media_bateria["Faixa"] == faixa]
    ax.plot(dados_faixa["Ano"], dados_faixa["Bateria (mAh)"], marker='o', label=faixa)

ax.set_title("Evolução da Capacidade Média de Bateria por Faixa de Preço")
ax.set_xlabel("Ano")
ax.set_ylabel("Bateria Média (mAh)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Observações
st.markdown("""
### 💡 Observações:
- Modelos High-End tendem a manter a bateria alta, mas nem sempre lideram.
- Alguns anos mostram crescimento significativo na faixa Mid-Range.
- Low-End tem evoluído, mas em ritmo mais lento.
""")
