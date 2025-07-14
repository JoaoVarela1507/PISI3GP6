import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from collections import Counter
import shap

from Home import EXCEL_FILE
from utils import dataframe_convert_columns

def get_model(model_name, model_params):
    """Retorna o modelo de classificação baseado no nome"""
    if model_name == "RandomForest":
        return RandomForestClassifier(**model_params)
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(**model_params)
    elif model_name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', **model_params)
    else:
        raise ValueError("Modelo desconhecido")

st.set_page_config(layout="wide")
st.title("📱 Classificação de Custo-Benefício de Smartphones")

if EXCEL_FILE:
    df = pd.read_excel(EXCEL_FILE)
    df.columns = df.columns.str.strip()

    df = dataframe_convert_columns(df)

    df.drop(columns=[
        "Mobile Weight", "Launched Price (USA)", "Battery Capacity", "RAM",
        "Front Camera", "Back Camera", "Screen Size", "Launched Year",
        "Company Name", "Model Name", "Processor", "Sistema Operacional",
        "Ano", "Marca", "Preço (R$)", "Sistema Operacional (Binário)", "Peso (g)"
    ], inplace=True)

    st.subheader("Prévia dos dados")
    st.dataframe(df.head())

    # Colunas a serem consideradas
    benefit_cols = ["Memoria Interna (GB)", "Câmera Frontal (MP)", "Câmera Traseira (MP)", "Tela (polegadas)", "Bateria (mAh)", "Preço (USD)"]
    df_copy = df.dropna(subset=benefit_cols).copy()

    scaler = StandardScaler()

    for col in benefit_cols:
        df_copy[f"z_{col}"] = scaler.fit_transform(df_copy[[col]])

    # Score ponderado
    df_copy["z_score"] = (
        df_copy["z_Bateria (mAh)"] +
        df_copy["z_Câmera Traseira (MP)"] * 0.7 +
        df_copy["z_Câmera Frontal (MP)"] * 0.3 +
        df_copy["z_Memoria Interna (GB)"] +
        df_copy["z_Tela (polegadas)"] -
        df_copy["z_Preço (USD)"] * 1.5
    )

    # Classificação em faixas, 0 = Baixo, 1 = Médio, 2 = Alto
    df_copy["CustoBeneficio"] = pd.qcut(df_copy["z_score"], 3, labels=[0, 1, 2])

    # Visualização da distribuição do score
    st.subheader("Distribuição do Score de Custo-Benefício")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_copy["z_score"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Score de Custo-Benefício")
    ax.set_ylabel("Frequência")
    ax.axvline(df_copy["z_score"].mean(), color='red', linestyle='--', label='Média')
    ax.legend()
    st.plotly_chart(fig, use_container_width=True)

    # Remove colunas iniciais
    df_copy.drop(columns=(benefit_cols), inplace=True)
    df_clean = df_copy.copy()
    df_clean.drop(columns=["z_score"], inplace=True)

    st.subheader("Dados após limpeza e transformação")
    st.dataframe(df_clean.head())

    # Hint sobre as classificações
    st.info("💡 **Classificação:** 0 = Baixo | 1 = Médio | 2 = Alto")

    # Separate features (X) and target (y)
    X = df_clean.drop('CustoBeneficio', axis=1)
    y = df_clean['CustoBeneficio']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    # Parâmetros do modelo
    st.subheader("Configuração dos Parâmetros do Modelo")

    # Escolha do modelo
    model_option = st.selectbox("Escolha o modelo de classificação:",
        ["RandomForest", "GradientBoosting", "XGBoost"])
    
    if model_option != "XGBoost":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.number_input("N Estimators", 
                min_value=10, 
                max_value=1000, 
                value=50, 
                step=10,
                help="Número de árvores no ensemble")
            
            max_depth = st.number_input("Max Depth", 
                min_value=1, 
                max_value=50, 
                value=8, 
                step=1,
                help="Profundidade máxima das árvores")
        
        with col2:
            random_state = st.number_input("Random State", 
                min_value=0, 
                max_value=1000, 
                value=42, 
                step=1,
                help="Seed para reprodutibilidade")
            
            min_samples_split = st.number_input("Min Samples Split", 
                min_value=2, 
                max_value=50, 
                value=4, 
                step=1,
                help="Número mínimo de amostras para dividir um nó")
        
        with col3:
            min_samples_leaf = st.number_input("Min Samples Leaf", 
                min_value=1, 
                max_value=50, 
                value=4, 
                step=1,
                help="Número mínimo de amostras em uma folha")
            
            max_features = st.selectbox("Max Features", 
                options=['sqrt', 'log2', None],
                index=0,
                help="Número de features a considerar ao procurar a melhor divisão")
    
        model_params = {
            'n_estimators': n_estimators, 
            'random_state': random_state, 
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }

    model = get_model(model_option, model_params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Importância das features")
    feature_importances = pd.DataFrame(model.feature_importances_,
        index=X_train.columns,
        columns=['Importance']).sort_values('Importance', ascending=False)
    st.bar_chart(feature_importances, horizontal=True, use_container_width=True, height=300)

    st.subheader("Matriz de Confusão - Original")
    cm_train = confusion_matrix(y_train, model.predict(X_train))
    cm_test = confusion_matrix(y_test, y_pred)
    
    report_train = classification_report(y_train, model.predict(X_train), output_dict=True)
    report_test = classification_report(y_test, y_pred, output_dict=True)

    # Usando colunas para centralizar e reduzir o tamanho
    col1, col2 = st.columns([1, 3])
    
    # Matriz de Confusão - Treino
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_xticklabels(['Baixo', 'Médio', 'Alto'])
        ax.set_yticklabels(['Baixo', 'Médio', 'Alto'])
        ax.set_title("Matriz de Confusão - Treino")
        st.pyplot(fig)
    with col2:
        st.subheader("Relatório de Classificação - Treino")
        st.dataframe(pd.DataFrame(report_train).transpose())

    # Matriz de Confusão - Teste
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_xticklabels(['Baixo', 'Médio', 'Alto'])
        ax.set_yticklabels(['Baixo', 'Médio', 'Alto'])
        ax.set_title("Matriz de Confusão - Teste")
        st.pyplot(fig)
    with col2:
        st.subheader("Relatório de Classificação - Teste")
        st.dataframe(pd.DataFrame(report_test).transpose())

    st.subheader("Distribuição das Classes")
    class_counts = Counter(y)
    class_counts_df = pd.DataFrame(class_counts.items(), columns=['Classe', 'Contagem'])
    class_counts_df['Classe'] = class_counts_df['Classe'].map({0: 'Baixo', 1: 'Médio', 2: 'Alto'})
    class_counts_df['Percentual'] = (class_counts_df['Contagem'] / class_counts_df['Contagem'].sum()) * 100
    st.dataframe(class_counts_df)

    # st.subheader("SHAP Values para Interpretação do Modelo")
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test, plot_type="bar")
    # fig = plt.gcf()
    # st.plotly_chart(fig, use_container_width=True)

    # shap.summary_plot(shap_values, X_test)
    # fig = plt.gcf()
    # st.plotly_chart(fig, use_container_width=True)

    st.header("SMOTE para Balanceamento de Classes e Redução de Overfitting")
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN

    samplers_analysis = {
        "Original": (X_train, y_train),
        "Oversampling": RandomOverSampler(random_state=42, sampling_strategy='auto'),
        "Undersampling": RandomUnderSampler(random_state=42, sampling_strategy='auto'),
        "SMOTE": SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5),
        "SMOTEENN": SMOTEENN(random_state=42, sampling_strategy='auto'),
    }

    for sampler_name, sampler in samplers_analysis.items():
        if isinstance(sampler, tuple):
            X_resampled, y_resampled = sampler
        else:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        st.subheader(f"Resultados com {sampler_name}")

        new_model = get_model(model_option, model_params)
        new_model.fit(X_resampled, y_resampled)

        X_test_scaled = X_test.copy()
        y_train_pred = new_model.predict(X_test_scaled)
        y_test_pred = new_model.predict(X_resampled)
            
        cm_test = confusion_matrix(y_resampled, y_test_pred)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")
            ax.set_xticklabels(['Baixo', 'Médio', 'Alto'])
            ax.set_yticklabels(['Baixo', 'Médio', 'Alto'])
            ax.set_title(f"Matriz de Confusão - {sampler_name} (Teste)")
            st.pyplot(fig)
        with col3:
            st.write(f"Tamanho original do treino: {X_train.shape[0]} amostras")
            st.write(f"Tamanho após {sampler_name}: {X_resampled.shape[0]} amostras")
            st.write(f"Aumento: {X_resampled.shape[0] - X_train.shape[0]} amostras sintéticas")

            balanced_class_counts = Counter(y_resampled)
            for class_label, count in balanced_class_counts.items():
                percentage = (count / len(y_resampled)) * 100
                st.write(f"Classe {class_label}: {count} amostras ({percentage:.2f}%)")

        train_acc = new_model.score(X_resampled, y_resampled)
        test_acc = new_model.score(X_test, y_test)

        st.write(f"**Acurácia Treino:** {train_acc:.2f}")
        st.write(f"**Acurácia Teste:** {test_acc:.2f}")
        st.write(f"**Gap (overfitting):** {train_acc - test_acc:.2f}")

