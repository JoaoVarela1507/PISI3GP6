import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from collections import Counter
import shap
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import pickle
import os

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
    
def export_model(model_name, model_params, sampler_name, sampler, X, y):
    # Treina o modelo com os dados balanceados
    final_model = get_model(model_name, model_params)
    if isinstance(sampler, tuple):
        X_balanced, y_balanced = sampler
    else:
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
    final_model.fit(X_balanced, y_balanced)

    # Cria o diretorio de exportação se não existir
    os.makedirs("exported_models", exist_ok=True)

    # Salva o modelo treinado com pickle
    filename = f"{model_option}_{sampler_name}_model.pkl"
    with open(os.path.join("exported_models", filename), 'wb') as file:
        pickle.dump(final_model, file)

    st.success(f"Modelo {model_name} exportado com sucesso como {filename}")

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


    samplers_analysis = {
        "Original": (X_train, y_train),
        "Oversampling": RandomOverSampler(random_state=42, sampling_strategy='auto'),
        "Undersampling": RandomUnderSampler(random_state=42, sampling_strategy='auto'),
        "SMOTE": SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5),
        "SMOTEENN": SMOTEENN(random_state=42, sampling_strategy='auto'),
    }

    results_comparison = {}
    for sampler_name, sampler in samplers_analysis.items():
        if isinstance(sampler, tuple):
            X_resampled, y_resampled = sampler
        else:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        st.subheader(f"Resultados com {sampler_name}")
        st.write(f"Tamanho original do treino: {X_train.shape[0]} amostras")
        st.write(f"Tamanho após {sampler_name}: {X_resampled.shape[0]} amostras")
        st.write(f"Aumento: {X_resampled.shape[0] - X_train.shape[0]} amostras sintéticas")

        # Treinamento do modelo com os dados balanceados
        new_model = get_model(model_option, model_params)
        new_model.fit(X_resampled, y_resampled)

        # Previsões
        X_test_scaled = X_test.copy()
        X_train_scaled = X_train.copy()
        y_test_pred = new_model.predict(X_test_scaled)
        y_train_pred = new_model.predict(X_train_scaled)

        # Metricas de desempenho
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        overfitting_gap = train_acc - test_acc

        # Validação cruzada no conjunto original (sem balanceamento)
        cv_scores = cross_val_score(new_model, X, y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Métricas por classe (macro average)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro')
            
        cm_test = confusion_matrix(y_test, y_test_pred)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")
            ax.set_xticklabels(['Baixo', 'Médio', 'Alto'])
            ax.set_yticklabels(['Baixo', 'Médio', 'Alto'])
            ax.set_title(f"Matriz de Confusão - {sampler_name} (Teste)")
            st.pyplot(fig)
        with col2:
            st.subheader(f"Relatório de Classificação - {sampler_name} (Teste)")
            report = classification_report(y_test, y_test_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        with col3:
            st.subheader(f"Desempenho do Modelo")
            st.write(f"**Acurácia Treino:** {train_acc:.3f}")
            st.write(f"**Acurácia Teste:** {test_acc:.3f}")
            st.write(f"**Gap (overfitting):** {overfitting_gap:.3f}")
            st.write(f"**CV Mean:** {cv_mean:.3f} ± {cv_std:.3f}")
            st.write(f"**Precisão (Macro):** {precision:.3f}")
            st.write(f"**Recall (Macro):** {recall:.3f}")
            st.write(f"**F1 Score (Macro):** {f1:.3f}")

        # Armazenar resultados
        results_comparison[sampler_name] = {
            'Samples': len(X_resampled),
            'Train_Acc': train_acc,
            'Test_Acc': test_acc,
            'Overfitting_Gap': overfitting_gap,
            'CV_Mean': cv_mean,
            'CV_Std': cv_std,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }

        if overfitting_gap > 0.05:
            st.error(f"⚠️ O modelo apresenta overfitting muito alto ({overfitting_gap:.3f}).")
        elif overfitting_gap > 0.03:
            st.warning(f"⚠️ O modelo apresenta overfitting significativo ({overfitting_gap:.3f}).")
        elif overfitting_gap >= 0:
            st.success(f"✅ O modelo apresenta um gap de overfitting muito baixo ({overfitting_gap:.3f}).")
        elif overfitting_gap > -0.03:
            st.success(f"⚠️ O modelo apresenta um gap de underfitting muito baixo ({overfitting_gap:.3f}).")
        elif overfitting_gap > -0.05:
            st.warning(f"⚠️ O modelo apresenta um gap de underfitting significativo ({overfitting_gap:.3f}).")
        else:
            st.error(f"⚠️ O modelo apresenta um gap de underfitting muito alto ({overfitting_gap:.3f}).")

    st.subheader("Comparativo entre os balanceamentos")
    # Criar DataFrame comparativo
    df_results = pd.DataFrame(results_comparison).T
    df_results = df_results.round(3)

    # Análise e recomendações
    best_overfitting = df_results['Overfitting_Gap'].idxmin()
    best_test_acc = df_results['Test_Acc'].idxmax()
    best_cv = df_results['CV_Mean'].idxmax()
    best_f1 = df_results['F1_Score'].idxmax()

    st.dataframe(df_results)
    st.subheader("Análise dos Resultados")
    st.write("Menor overfitting:", best_overfitting, "(gap:", df_results.loc[best_overfitting, 'Overfitting_Gap'], ")")
    st.write("Melhor acurácia teste:", best_test_acc, "(", df_results.loc[best_test_acc, 'Test_Acc'], ")")
    st.write("Melhor F1-Score:", best_f1, "(", df_results.loc[best_f1, 'F1_Score'], ")")

    st.subheader("Visualização comparativa")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    techniques = df_results.index

    # Acurácia de teste
    ax1.bar(techniques, df_results['Test_Acc'], color='lightblue', alpha=0.7)
    ax1.set_title('Acurácia no Teste')
    ax1.set_ylabel('Acurácia')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Gap de overfitting
    colors = ['red' if gap > 0.05 else 'orange' if gap > 0.03 else 'green' 
            for gap in df_results['Overfitting_Gap']]
    ax2.bar(techniques, df_results['Overfitting_Gap'], color=colors, alpha=0.7)
    ax2.set_title('Gap de Overfitting (menor é melhor)')
    ax2.set_ylabel('Gap (Treino - Teste)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Limite crítico')
    ax2.axhline(y=0.03, color='orange', linestyle='--', alpha=0.5, label='Limite bom')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Validação cruzada
    ax3.bar(techniques, df_results['CV_Mean'], color='lightgreen', alpha=0.7)
    ax3.errorbar(techniques, df_results['CV_Mean'], yerr=df_results['CV_Std'], 
                fmt='none', color='black', capsize=5)
    ax3.set_title('Validação Cruzada (5-fold)')
    ax3.set_ylabel('Acurácia Média ± Desvio')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # F1-Score macro
    ax4.bar(techniques, df_results['F1_Score'], color='lightcoral', alpha=0.7)
    ax4.set_title('F1-Score Macro')
    ax4.set_ylabel('F1-Score')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    plt.clf()

    st.subheader("Análise de Importância de Features com SHAP")
    # Inicialização do SHAP
    shap.initjs()

    # Explicador SHAP
    explainer = shap.TreeExplainer(model)

    X_shap = pd.DataFrame(X_test, columns=X.columns)
    X_shap.head()

    shap_values = explainer.shap_values(X_shap)

    shap.summary_plot(shap_values, X_shap)
    fig = plt.gcf()
    st.plotly_chart(fig)

    st.subheader("Exportar modelos treinados para pickle")
    
    st.write("Clique no botão abaixo para exportar o modelo treinado com o sampler correspondente:")
    
    # Criar colunas para organizar os botões
    num_samplers = len(samplers_analysis)
    cols = st.columns(num_samplers)
    
    for idx, (sampler_name, sampler_data) in enumerate(results_comparison.items()):
        current_sampler = samplers_analysis[sampler_name]
        with cols[idx]:
            st.write(f"**{sampler_name}**")
            st.write(f"Acurácia: {sampler_data['Test_Acc']:.3f}")
            st.write(f"F1-Score: {sampler_data['F1_Score']:.3f}")
            
            # Botão para exportar o modelo
            if st.button(f"Exportar {model_option} ({sampler_name})", key=f"export_{model_option}_{sampler_name}"):
                try:
                    export_model(model_option, model_params, sampler_name, current_sampler, X_resampled, y_resampled)
                    
                except Exception as e:
                    st.error(f"❌ Erro ao exportar modelo: {str(e)}")
    
    # Seção adicional com informações sobre os modelos exportados
    st.markdown("---")
    st.subheader("ℹ️ Informações sobre a Exportação")
    st.write("""
    - Os modelos são salvos no formato pickle (.pkl)
    - Cada arquivo contém o modelo treinado com o respectivo sampler
    - Para carregar um modelo, use: `pickle.load(open('caminho_do_arquivo.pkl', 'rb'))`
    - Os modelos mantêm todas as configurações de hiperparâmetros definidas
    """)
    
    # Mostrar resumo dos modelos disponíveis
    with st.expander("📋 Resumo dos Modelos Disponíveis"):
        summary_data = []
        for sampler_name, data in results_comparison.items():
            summary_data.append({
                'Sampler': sampler_name,
                'Amostras': data['Samples'],
                'Acurácia Teste': f"{data['Test_Acc']:.3f}",
                'F1-Score': f"{data['F1_Score']:.3f}",
                'Overfitting Gap': f"{data['Overfitting_Gap']:.3f}"
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

