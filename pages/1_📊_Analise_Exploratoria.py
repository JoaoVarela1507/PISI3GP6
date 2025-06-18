import streamlit as st
import pandas as pd
import plotly.express as px

from Home import EXCEL_FILE, corrigir_inflacao, classificar_por_faixa

# Configuração da página
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