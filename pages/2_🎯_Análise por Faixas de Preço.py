import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from Home import EXCEL_FILE, corrigir_inflacao, classificar_por_faixa

# Configuração da página
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

    colors = ['#4CAF50', '#2196F3', '#9C27B0']  # Cores para cada faixa
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(faixa_counts['Quantidade'], labels=faixa_counts['Faixa'], autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Distribuição de Smartphones por Faixa de Preço', fontsize=16)
    st.pyplot(fig)

    # Evolução das faixas ao longo do tempo
    st.subheader("📈 Evolução das Faixas de Preço ao Longo do Tempo")
    faixas_ano = df.groupby(['Ano', 'Faixa']).size().reset_index(name='Quantidade')

    fig, ax = plt.subplots(figsize=(10, 6))
    for faixa, color in zip(['Low-End', 'Mid-Range', 'High-End'], ['#4CAF50', '#2196F3', '#9C27B0']):
        data_faixa = faixas_ano[faixas_ano['Faixa'] == faixa]
        ax.bar(data_faixa['Ano'], data_faixa['Quantidade'], label=faixa, color=color)

    ax.set_title('Evolução do Número de Lançamentos por Faixa e Ano', fontsize=16)
    ax.set_xlabel('Ano')
    ax.set_ylabel('Quantidade')
    ax.legend(title='Faixa de Preço')
    st.pyplot(fig)

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
        # Configurar o gráfico de barras com Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        for faixa, color in zip(['Low-End', 'Mid-Range', 'High-End'], ['#4CAF50', '#2196F3', '#9C27B0']):
            ax.bar(
                marca_faixa_filtrado.index,
                marca_faixa_filtrado[faixa],
                label=faixa,
                color=color,
                alpha=0.8
            )

        # Configurar título, rótulos e legenda
        ax.set_title('Distribuição Percentual de Produtos por Marca e Faixa', fontsize=16)
        ax.set_xlabel('Marca', fontsize=12)
        ax.set_ylabel('Percentual (%)', fontsize=12)
        ax.legend(title='Faixa de Preço', loc='upper right')

        # Ajustar rotação dos rótulos do eixo x para melhor visualização
        plt.xticks(rotation=45)

        # Exibir o gráfico no Streamlit
        st.pyplot(fig)

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

    # Configurar o gráfico de radar
    labels = list(caracteristicas_norm.columns)
    num_vars = len(labels)

    # Adicionar o primeiro elemento ao final para fechar o gráfico
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plotar cada faixa
    for faixa in caracteristicas_norm.index:
        values = caracteristicas_norm.loc[faixa].values.tolist()
        values += values[:1]  # Fechar o gráfico
        ax.plot(angles, values, label=faixa, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    # Configurar os rótulos do eixo
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color="gray")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Adicionar título e legenda
    ax.set_title("Comparação de Características por Faixa (Normalizado 0-100)", size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.show()

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

    # Configurar o gráfico de linha com Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    for faixa, color in zip(['Low-End', 'Mid-Range', 'High-End'], ['#4CAF50', '#2196F3', '#9C27B0']):
        data_faixa = preco_medio_faixa[preco_medio_faixa['Faixa'] == faixa]
        ax.plot(data_faixa['Ano'], data_faixa['Preço Corrigido (R$)'], label=faixa, color=color, marker='o')

    # Configurar título, rótulos e legenda
    ax.set_title('Evolução do Preço Médio por Faixa (Corrigido por Inflação)', fontsize=16)
    ax.set_xlabel('Ano', fontsize=12)
    ax.set_ylabel('Preço Médio Corrigido (R$)', fontsize=12)
    ax.legend(title='Faixa de Preço', loc='upper left')

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)

except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")