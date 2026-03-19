import streamlit as st

def show():

    st.markdown("<h3>Project Introduction</h3>", unsafe_allow_html=True)

    st.write(
        """
        **Topic 2: Developing a Vietnamese News Classification System Using XGBoost**

        Project Objectives:
        - Collect and process Vietnamese news data from multiple sources.
        - Convert text into TF-IDF vectors.
        - Train news classification models using XGBoost.
        - Allow users to upload CSV, Excel, or .txt files.
        - The system will automatically train 3 models (XGBoost, SVM, Logistic Regression) and compare their accuracy.
        - Save model.pkl after training.
        - Allow users to input new news articles and predict their labels.
        """
    )

    st.image("rose.png", width=300)
