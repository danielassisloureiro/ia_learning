import streamlit as st

st.set_page_config(page_title="Escolha de Modelo de ML", layout="wide")
st.title("🤖 Fluxograma Interativo de Escolha de Modelos de Machine Learning")

st.markdown("""
### 📍 Passo 1: Seu problema tem rótulos (labels)?
""")

label_type = st.radio("Selecione uma opção:", ["Sim", "Não"])

if label_type == "Sim":
    st.markdown("""
    ### 🧠 Passo 2: Qual o tipo de saída?
    """)
    output_type = st.radio("Tipo de saída:", ["Categórica (Classificação)", "Numérica contínua (Regressão)"])

    if output_type == "Categórica (Classificação)":
        st.markdown("""
        #### Recomendação de Modelos:
        - **Simples:** `Logistic Regression`, `Decision Tree`
        - **Média complexidade:** `Random Forest`, `SVM`
        - **Alta performance:** `XGBoost`, `Neural Networks`
        """)

    elif output_type == "Numérica contínua (Regressão)":
        st.markdown("""
        #### Recomendação de Modelos:
        - **Relacionamento Linear:** `Linear Regression`
        - **Não linear:** `Random Forest Regressor`, `XGBoost Regressor`
        """)

else:
    st.markdown("""
    ### 🧩 Passo 2: O que você deseja fazer?
    """)
    task_type = st.radio("Objetivo:", ["Agrupar dados (Clustering)", "Reduzir variáveis (Redução de dimensionalidade)"])

    if task_type == "Agrupar dados (Clustering)":
        st.markdown("""
        #### Recomendação de Modelos:
        - **Grupos bem separados:** `K-Means`
        - **Com ruído ou formas arbitrárias:** `DBSCAN`, `Hierarchical Clustering`
        """)
    else:
        st.markdown("""
        #### Recomendação de Modelos:
        - **Visualização de alta dimensionalidade:** `t-SNE`, `UMAP`
        - **Pré-processamento / compressão:** `PCA`
        """)

st.markdown("---")

st.markdown("""
### 🤖 Deseja trabalhar com dados de imagem, texto ou séries temporais?

Se sim, explore modelos de **Deep Learning**:
- **Imagens:** `CNN`
- **Texto:** `Transformers` (ex: BERT, GPT)
- **Sequências temporais:** `RNN`, `LSTM`
""")
