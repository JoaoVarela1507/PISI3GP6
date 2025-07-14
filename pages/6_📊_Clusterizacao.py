import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

from Home import EXCEL_FILE
from utils import dataframe_convert_columns

st.title("📊 Análise de Clusterização de Smartphones")

try:
    df = pd.read_excel(EXCEL_FILE)
    df.columns = df.columns.str.strip()
    df = dataframe_convert_columns(df)

    st.subheader("📌 Variáveis Selecionadas para Agrupamento")

    variaveis_default = [
        "Preço (R$)", "Bateria (mAh)", "Peso (g)",
        "Memoria Interna (GB)", "Câmera Frontal (MP)",
        "Câmera Traseira (MP)", "Tela (polegadas)"
    ]

    variaveis_escolhidas = st.multiselect(
        "Escolha as variáveis:",
        options=variaveis_default,
        default=variaveis_default
    )

    df_cluster = df[variaveis_escolhidas].dropna()
    X_scaled = StandardScaler().fit_transform(df_cluster)

    k = st.slider("🔢 Escolha o número de clusters (K):", 2, 10, 5)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df_cluster["PCA1"] = coords[:, 0]
    df_cluster["PCA2"] = coords[:, 1]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_cluster["cluster"] = kmeans.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, df_cluster["cluster"])
    st.metric("Silhouette Score", round(silhouette, 3))

    st.subheader("📍 Visualização dos Clusters (PCA)")
    fig = px.scatter(df_cluster, x="PCA1", y="PCA2", color=df_cluster["cluster"].astype(str),
                     title="Visualização com PCA dos Clusters",
                     labels={"cluster": "Cluster"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Boxplot por Variável")
    for var in variaveis_escolhidas:
        st.plotly_chart(
            px.box(df_cluster, x="cluster", y=var, color="cluster",
                   title=f"Distribuição de {var} por Cluster"),
            use_container_width=True
        )

    st.subheader("📋 Médias por Cluster")
    st.dataframe(df_cluster.groupby("cluster")[variaveis_escolhidas].mean().round(2))

except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
