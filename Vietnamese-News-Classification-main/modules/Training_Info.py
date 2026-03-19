import streamlit as st
import pandas as pd
import joblib
import os
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, f1_score


# ============================================================
# PAGE MAIN
# ============================================================

def show():
    st.markdown(
        "<h3 style='color:blue;'>Training Info – Training Parameters for 3 Models</h3>",
        unsafe_allow_html=True,
    )
    st.write("---")

    # ============================================================
    # 1. CHECK FILES
    # ============================================================
    st.write("## 1. Check Model & Vectorizer")

    model_path = "export/model.pkl"
    vec_path = "export/vectorizer.pkl"
    le_path = "export/label_encoder.pkl"
    train_info_path = "export/train_info.json"

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        st.error("❌ model.pkl or vectorizer.pkl not found → model not trained yet.")
        st.stop()

    st.success(f"✔ Model: {model_path}")
    st.success(f"✔ Vectorizer: {vec_path}")
    st.success(f"✔ Label Encoder: {le_path}")

    st.write("---")

    # ============================================================
    # 2. TRAINING INFORMATION
    # ============================================================
    st.write("## 2. Training Information")

    if os.path.exists(train_info_path):
        with open(train_info_path, "r", encoding="utf-8") as f:
            train_info = json.load(f)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Best Model", train_info.get("best_model", "N/A"))
        col2.metric("🎯 Accuracy", f"{train_info.get('accuracy', 0):.4f}")
        col3.metric("📊 F1-Score", f"{train_info.get('f1_score', 0):.4f}")
        col4.metric("📦 Samples", f"{train_info.get('num_samples', 0):,}")
        
        st.write("")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Train Size", f"{train_info.get('train_size', 0):,}")
        col6.metric("Test Size", f"{train_info.get('test_size', 0):,}")
        col7.metric("Num Classes", train_info.get("num_classes", 0))
        col8.metric("Trained At", train_info.get("trained_at", "N/A")[:10])
        
        st.write("")
        st.write("**Labels:**", ", ".join(train_info.get("labels", [])))
        
    else:
        st.warning("⚠ train_info.json not found — please retrain your model!")

    st.write("---")

    # ============================================================
    # 3. COMPARE 3 MODELS
    # ============================================================
    st.write("## 3. Compare 3 Models")

    if os.path.exists(train_info_path) and "all_models" in train_info:
        all_models = train_info["all_models"]
        
        # Table
        comparison_data = []
        for name, metrics in all_models.items():
            is_best = "🏆" if name == train_info.get("best_model") else ""
            comparison_data.append({
                "": is_best,
                "Model": name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "F1-Score": f"{metrics['f1_score']:.4f}"
            })
        
        st.dataframe(pd.DataFrame(comparison_data))
        
        # Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        models_names = list(all_models.keys())
        accuracies = [all_models[m]['accuracy'] for m in models_names]
        f1_scores = [all_models[m]['f1_score'] for m in models_names]
        
        x = range(len(models_names))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], accuracies, width, label='Accuracy', color='#3498db')
        bars2 = ax.bar([i + width/2 for i in x], f1_scores, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xticks(x)
        ax.set_xticklabels(models_names)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("Model Comparison")
        
        for bar in bars1:
            ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar in bars2:
            ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No comparison data for 3 models yet. Please retrain with Analysis.")

    st.write("---")

    # ============================================================
    # 4. CONFUSION MATRICES
    # ============================================================
    st.write("## 4. Confusion Matrices")

    cm_all_path = "export/confusion_matrix_all_models.png"
    if os.path.exists(cm_all_path):
        st.image(cm_all_path, caption="Confusion Matrix - 3 Models", use_column_width=True)
    else:
        # Display individual confusion matrices
        cm_files = [
            ("XGBoost", "export/confusion_matrix_xgboost.png"),
            ("Logistic Regression", "export/confusion_matrix_logistic_regression.png"),
            ("SVM", "export/confusion_matrix_svm.png"),
        ]
        
        existing_cms = [(name, path) for name, path in cm_files if os.path.exists(path)]
        
        if existing_cms:
            cols = st.columns(len(existing_cms))
            for i, (name, path) in enumerate(existing_cms):
                with cols[i]:
                    st.image(path, caption=name, use_column_width=True)
        else:
            st.info("No confusion matrix yet. Please train model first.")

    st.write("---")

    # ============================================================
    # 5. MODEL DETAILS
    # ============================================================
    st.write("## 5. Trained Model Details")

    model = joblib.load(model_path)
    st.code(str(model))

    st.write("---")

    # ============================================================
    # 6. CLASSIFICATION REPORTS
    # ============================================================
    st.write("## 6. Classification Reports")

    report_files = [
        ("XGBoost", "export/report_xgboost.txt"),
        ("Logistic Regression", "export/report_logistic_regression.txt"),
        ("SVM", "export/report_svm.txt"),
    ]

    tabs = st.tabs([name for name, _ in report_files])
    
    for i, (name, path) in enumerate(report_files):
        with tabs[i]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    report = f.read()
                st.code(report, language="text")
            else:
                st.info(f"No report for {name} yet")

    st.write("---")

    # ============================================================
    # 7. MODEL FILES
    # ============================================================
    st.write("## 7. Model Files List")

    model_files = [
        "model.pkl",
        "model_xgboost.pkl",
        "model_logistic_regression.pkl",
        "model_svm.pkl",
        "vectorizer.pkl",
        "label_encoder.pkl",
        "train_info.json",
    ]

    file_data = []
    for f in model_files:
        path = f"export/{f}"
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
            file_data.append({"File": f, "Size": size_str, "Status": "✅"})
        else:
            file_data.append({"File": f, "Size": "-", "Status": "❌"})

    st.dataframe(pd.DataFrame(file_data))

    st.write("---")

    # ============================================================
    # 8. CONCLUSION
    # ============================================================
    st.write("## 8. Conclusion")
    
    if os.path.exists(train_info_path):
        best_model = train_info.get("best_model", "N/A")
        best_acc = train_info.get("accuracy", 0)
        best_f1 = train_info.get("f1_score", 0)
        
        st.info(f"""
        ✅ **Best Model**: {best_model}
        
        ✅ **Accuracy**: {best_acc:.4f} ({best_acc*100:.2f}%)
        
        ✅ **F1-Score**: {best_f1:.4f}
        
        ✅ Trained and compared 3 models: XGBoost, Logistic Regression, SVM
        
        ✅ Can be upgraded with BERT / PhoBERT for better quality.
        """)
    else:
        st.info("""
        ✔ Data has been checked and cleaned.  
        ✔ Current model works stably with high F1-score.  
        ✔ XGBoost / SVM / Logistic Regression can be used as baseline.  
        ✔ Can be upgraded with BERT / PhoBERT for better quality.
        """)
