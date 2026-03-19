import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import io
import re
import unicodedata
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

import joblib


# =========================================================
# 🔧 TIỀN XỬ LÝ TIẾNG VIỆT - HƯỚNG DẪN CHI TIẾT
# =========================================================
"""
📚 HƯỚNG DẪN TIỀN XỬ LÝ DỮ LIỆU VĂN BẢN TIẾNG VIỆT

Quy trình tiền xử lý bao gồm các bước sau:

1️⃣ CHUẨN HÓA UNICODE (Normalization)
   - Sử dụng unicodedata.normalize("NFC", text)
   - Đảm bảo các ký tự tiếng Việt (dấu) được biểu diễn thống nhất
   - Ví dụ: "việt" có thể được encode nhiều cách khác nhau

2️⃣ CHUYỂN THÀNH CHỮ THƯỜNG (Lowercase)
   - text.lower()
   - Giúp model không phân biệt "Việt Nam" và "việt nam"

3️⃣ LOẠI BỎ KÝ TỰ ĐẶC BIỆT (Remove Special Characters)
   - Giữ lại: chữ cái (a-z, A-Z), số (0-9), dấu tiếng Việt (À-ỹ)
   - Loại bỏ: dấu câu, ký hiệu đặc biệt, emoji
   - Pattern: [^a-zA-Z0-9À-ỹ\\s]

4️⃣ TÁCH TỪ TIẾNG VIỆT (Word Segmentation) - QUAN TRỌNG!
   - Tiếng Việt có từ ghép: "học sinh", "sinh viên", "bóng đá"
   - Sử dụng thư viện: pyvi hoặc underthesea
   - Ví dụ: "học sinh giỏi" → "học_sinh giỏi"
   
   Cài đặt: pip install pyvi
   Sử dụng:
   ```python
   from pyvi import ViTokenizer
   text = ViTokenizer.tokenize("Học sinh Việt Nam")
   # Output: "Học_sinh Việt_Nam"
   ```

5️⃣ LOẠI BỎ STOPWORDS (Remove Stopwords)
   - Stopwords: từ xuất hiện nhiều nhưng không mang nghĩa
   - Ví dụ: "và", "là", "của", "những", "các", "một"
   - Giúp giảm nhiễu và tăng độ chính xác

6️⃣ TẠO TF-IDF FEATURES
   - TF (Term Frequency): tần suất từ trong văn bản
   - IDF (Inverse Document Frequency): độ quan trọng của từ
   - Tham số quan trọng:
     * max_features: số lượng features tối đa (10000-15000)
     * ngram_range: (1,2) hoặc (1,3) để bắt cụm từ
     * min_df: loại bỏ từ xuất hiện quá ít
     * max_df: loại bỏ từ xuất hiện quá nhiều

📌 LƯU Ý QUAN TRỌNG:
   - Luôn dùng Word Segmentation cho tiếng Việt
   - Cân bằng dữ liệu nếu các class không đều
   - Title thường quan trọng hơn content → có thể weight cao hơn
"""

# Import pyvi cho word segmentation (cần cài đặt: pip install pyvi)
try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False

# Stopwords tiếng Việt - các từ không mang nghĩa
STOPWORDS = set("""
và là của những cái các một trong được để với từ khi mà thì là đều này kia hoặc 
nên nếu tuy vì nhưng vậy còn rất lại đã đang sẽ có không cũng như cho về theo 
tại vào ra bởi trên dưới qua chỉ hơn đến hay hoặc nữa việc người năm
nào sau ông bà anh chị em họ chúng tôi ta mình ai gì sao thế nào
rất rồi lên xuống nên nay đây kia ấy này khi bao giờ bởi vì
""".split())

# Topic mapping để chuẩn hóa các tên chủ đề khác nhau về dạng thống nhất
TOPIC_MAPPING = {
    'Thể thao': 'Thể thao', 'THỂ THAO': 'Thể thao', 'Bóng đá': 'Thể thao',
    'Bóng đá Việt Nam': 'Thể thao', 'Bóng đá quốc tế': 'Thể thao',
    'Thế giới': 'Thế giới', 'THẾ GIỚI': 'Thế giới', 'Quốc tế': 'Thế giới',
    'Xã hội': 'Xã hội', 'XÃ HỘI': 'Xã hội', 'Đời sống': 'Xã hội', 'Sống': 'Xã hội',
    'Pháp luật': 'Pháp luật', 'PHÁP LUẬT': 'Pháp luật',
    'Thời sự': 'Thời sự', 'THỜI SỰ': 'Thời sự', 'Trong nước': 'Thời sự', 'Tin tức': 'Thời sự',
    'Kinh doanh': 'Kinh doanh', 'Kinh tế': 'Kinh doanh', 'Tài chính': 'Kinh doanh',
    'Tài chính - Kinh doanh': 'Kinh doanh', 'Bất động sản': 'Kinh doanh',
    'Giải trí': 'Giải trí', 'Văn hóa - Giải trí': 'Giải trí', 'Sao Việt': 'Giải trí',
    'Sức khỏe': 'Sức khỏe', 'Y tế': 'Sức khỏe',
    'Giáo dục': 'Giáo dục', 'Tuyển sinh': 'Giáo dục',
    'Công nghệ': 'Công nghệ', 'Số hóa': 'Công nghệ', 'Xe': 'Công nghệ',
    'Văn hóa': 'Văn hóa', 'Du lịch': 'Văn hóa', 'Giới trẻ': 'Văn hóa',
}

def normalize_topic(topic):
    """Chuẩn hóa tên topic về dạng thống nhất"""
    if pd.isna(topic) or topic == '' or topic == 'None':
        return None
    return TOPIC_MAPPING.get(str(topic).strip(), str(topic).strip())

def preprocess_text(text):
    """
    Tiền xử lý văn bản tiếng Việt
    
    Các bước:
    1. Kiểm tra input hợp lệ
    2. Chuẩn hóa Unicode (NFC)
    3. Chuyển thành chữ thường
    4. Loại bỏ ký tự đặc biệt (giữ dấu tiếng Việt)
    5. Tách từ tiếng Việt với pyvi (nếu có)
    6. Loại bỏ stopwords
    
    Args:
        text: Văn bản cần xử lý
        
    Returns:
        Văn bản đã được tiền xử lý
    """
    if not isinstance(text, str):
        return ""
    
    # Bước 1-2: Chuẩn hóa Unicode và chuyển thành chữ thường
    text = unicodedata.normalize("NFC", text).lower()
    
    # Bước 3: Loại bỏ ký tự đặc biệt, giữ lại chữ cái, số và dấu tiếng Việt
    text = re.sub(r"[^a-zA-Z0-9À-ỹ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Bước 4: Tách từ tiếng Việt (Word Segmentation)
    if PYVI_AVAILABLE:
        try:
            text = ViTokenizer.tokenize(text)
        except:
            pass
    
    # Bước 5: Loại bỏ stopwords
    return " ".join([w for w in text.split() if w not in STOPWORDS])


# =========================================================
# 📦 HÀM ĐỌC DATA
# =========================================================

def generate_sample_zip():
    files = {
        "politics_01.txt": "Chính phủ vừa thông qua nghị quyết mới về phát triển kinh tế số.",
        "education_01.txt": "Bộ Giáo dục công bố đổi mới chương trình học phổ thông.",
        "weather_01.txt": "Miền Bắc rét đậm do ảnh hưởng không khí lạnh.",
        "sports_01.txt": "Việt Nam thắng 3-1 trong trận giao hữu."
    }
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w") as z:
        for fn, content in files.items():
            z.writestr(fn, content)
    mem.seek(0)
    return mem


def read_txt_folder(files):
    rows = []
    for f in files:
        if f.name.endswith(".txt"):
            base = os.path.splitext(f.name)[0]
            parts = base.split(".")
            label = parts[-1].strip().upper()
            content = f.read().decode("utf-8", errors="ignore")
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            for line in lines:
                rows.append([line, label])
    return pd.DataFrame(rows, columns=["text", "label"])


def read_txt_zip(file):
    rows = []
    with zipfile.ZipFile(file, "r") as z:
        for fn in z.namelist():
            if fn.endswith(".txt"):
                base = os.path.splitext(fn)[0]
                parts = base.split("_")
                label = parts[0].upper()
                text = z.read(fn).decode("utf-8", errors="ignore")
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                for line in lines:
                    rows.append([line, label])
    return pd.DataFrame(rows, columns=["text", "label"])


def load_kaggle_data():
    """Load dữ liệu từ Kaggle dataset"""
    if not os.path.exists("news_dataset.json"):
        return None
    
    with open("news_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['normalized_topic'] = df['topic'].apply(normalize_topic)
    df_clean = df[df['normalized_topic'].notna()].copy()
    
    # Chọn top 10 topics
    topic_counts = df_clean['normalized_topic'].value_counts()
    top_topics = topic_counts.head(10).index.tolist()
    df_train = df_clean[df_clean['normalized_topic'].isin(top_topics)].copy()
    
    # Tạo text từ title + content
    df_train['full_text'] = df_train['title'].fillna('') + ' ' + df_train['content'].fillna('')
    df_train['clean_text'] = df_train['full_text'].apply(preprocess_text)
    df_train = df_train[df_train['clean_text'].str.len() > 50].copy()
    
    # Rename để phù hợp
    result = df_train[['clean_text', 'normalized_topic']].copy()
    result.columns = ['text', 'label']
    
    return result


# =========================================================
# 🧠 MAIN INTERFACE
# =========================================================

def show():
    st.markdown("### 🧠 Analysis – Train News Classification Models (3 Models)")

    # =========================================================
    # 1️⃣ UPLOAD DATA
    # =========================================================
    st.write("---")
    st.header("1️⃣ Upload Data")

    # Kiểm tra Kaggle data
    kaggle_available = os.path.exists("news_dataset.json")
    
    if kaggle_available:
        mode = st.radio(
            "Select data source:",
            ["📦 Kaggle Dataset (184K+ articles)", "Folder TXT", "ZIP TXT", "CSV / Excel"],
            horizontal=True
        )
    else:
        mode = st.radio(
            "Select data upload mode:",
            ["Folder TXT", "ZIP TXT", "CSV / Excel"],
            horizontal=True
        )
        st.info("💡 Download Kaggle dataset for 184K+ articles: `kagglehub.dataset_download('haitranquangofficial/vietnamese-online-news-dataset')`")

    if "df" not in st.session_state:
        st.session_state.df = None

    df = None

    if mode.startswith("📦 Kaggle"):
        if st.button("📥 Load Kaggle Dataset"):
            with st.spinner("Loading data..."):
                df = load_kaggle_data()
                if df is not None:
                    st.session_state.df = df
                    st.success(f"✅ Loaded {len(df):,} articles from Kaggle dataset!")
                    st.dataframe(df.head(10))
                else:
                    st.error("❌ File news_dataset.json not found")

    elif mode == "Folder TXT":
        files = st.file_uploader("Select multiple TXT files", type=["txt"], accept_multiple_files=True)
        if files:
            df = read_txt_folder(files)
            st.session_state.df = df
            st.success(f"✔ Read {len(df)} news lines!")
            st.dataframe(df)

    elif mode == "ZIP TXT":
        up = st.file_uploader("Upload ZIP", type=["zip"])
        if up:
            df = read_txt_zip(up)
            st.session_state.df = df
            st.success(f"✔ ZIP read successfully ({len(df)} lines)!")
            st.dataframe(df)

    else:
        up = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
        if up:
            ext = up.name.split(".")[-1]
            df = pd.read_csv(up) if ext == "csv" else pd.read_excel(up)
            st.session_state.df = df
            st.success("✔ Spreadsheet file read successfully!")
            st.dataframe(df)

    # =========================================================
    # 📊 DATA STATISTICS
    # =========================================================
    if st.session_state.df is not None:
        st.write("---")
        st.subheader("📊 Data Statistics by Label")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        label_counts = st.session_state.df["label"].value_counts()
        sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax, palette="viridis")
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    # =========================================================
    # 2️⃣ TRAIN MODEL
    # =========================================================
    st.write("---")
    st.header("2️⃣ Train model (3 Models)")
    
    col1, col2 = st.columns(2)
    with col1:
        max_samples = st.number_input("Max samples/class:", min_value=100, max_value=10000, value=5000)
    with col2:
        test_size = st.slider("Test size:", min_value=0.1, max_value=0.4, value=0.2)

    status = st.empty()

    if st.button("🚀 Train 3 Models"):
        df = st.session_state.df
        if df is None or len(df) < 10:
            st.error("❌ Dataset too small. Need at least 10 news lines.")
            return

        # Preprocess if not already done
        if 'text' in df.columns:
            df["clean_text"] = df["text"].apply(preprocess_text)
        else:
            st.error("❌ Column 'text' not found")
            return

        # Label Encoder
        le = LabelEncoder()
        le.fit(df["label"])

        # Balance data
        status.info("🔄 Balancing data...")
        df_balanced = df.groupby('label').apply(
            lambda x: x.sample(min(len(x), max_samples), random_state=42)
        ).reset_index(drop=True)

        y_balanced = le.transform(df_balanced['label'])
        X_balanced = df_balanced['clean_text'].values

        # Train/Validation/Test Split (70% / 10% / 20%)
        # Bước 1: Tách Test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_balanced, y_balanced,
            test_size=test_size,
            stratify=y_balanced,
            random_state=42
        )
        
        # Bước 2: Tách Validation set từ phần còn lại (10% của tổng = 12.5% của temp)
        val_ratio = 0.125  # 10% / 80% = 0.125
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=42
        )

        st.info(f"📦 Train: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}")

        # TF-IDF
        status.info("🔄 Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)      # Transform validation set
        X_test_vec = vectorizer.transform(X_test)

        # =========================================================
        # TRAIN 3 MODELS
        # =========================================================
        results = {}
        models = {}

        # XGBoost
        status.info("🚀 [1/3] Training XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", n_jobs=-1, random_state=42
        )
        xgb_model.fit(X_train_vec, y_train)
        
        # Đánh giá trên Validation set
        y_val_pred_xgb = xgb_model.predict(X_val_vec)
        val_acc_xgb = accuracy_score(y_val, y_val_pred_xgb)
        val_f1_xgb = f1_score(y_val, y_val_pred_xgb, average='macro')
        
        # Đánh giá trên Test set
        y_pred_xgb = xgb_model.predict(X_test_vec)
        results['XGBoost'] = {
            'val_accuracy': val_acc_xgb,
            'val_f1_score': val_f1_xgb,
            'accuracy': accuracy_score(y_test, y_pred_xgb),
            'f1_score': f1_score(y_test, y_pred_xgb, average='macro'),
            'predictions': y_pred_xgb
        }
        models['XGBoost'] = xgb_model

        # Logistic Regression
        status.info("🚀 [2/3] Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
        lr_model.fit(X_train_vec, y_train)
        
        # Đánh giá trên Validation set
        y_val_pred_lr = lr_model.predict(X_val_vec)
        val_acc_lr = accuracy_score(y_val, y_val_pred_lr)
        val_f1_lr = f1_score(y_val, y_val_pred_lr, average='macro')
        
        # Đánh giá trên Test set
        y_pred_lr = lr_model.predict(X_test_vec)
        results['Logistic Regression'] = {
            'val_accuracy': val_acc_lr,
            'val_f1_score': val_f1_lr,
            'accuracy': accuracy_score(y_test, y_pred_lr),
            'f1_score': f1_score(y_test, y_pred_lr, average='macro'),
            'predictions': y_pred_lr
        }
        models['Logistic Regression'] = lr_model

        # SVM
        status.info("🚀 [3/3] Training SVM...")
        svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
        svm_model.fit(X_train_vec, y_train)
        
        # Đánh giá trên Validation set
        y_val_pred_svm = svm_model.predict(X_val_vec)
        val_acc_svm = accuracy_score(y_val, y_val_pred_svm)
        val_f1_svm = f1_score(y_val, y_val_pred_svm, average='macro')
        
        # Đánh giá trên Test set
        y_pred_svm = svm_model.predict(X_test_vec)
        results['SVM'] = {
            'val_accuracy': val_acc_svm,
            'val_f1_score': val_f1_svm,
            'accuracy': accuracy_score(y_test, y_pred_svm),
            'f1_score': f1_score(y_test, y_pred_svm, average='macro'),
            'predictions': y_pred_svm
        }
        models['SVM'] = svm_model

        status.success("✅ Training complete!")

        # =========================================================
        # DISPLAY RESULTS
        # =========================================================
        st.subheader("📊 Comparison Results for 3 Models")
        
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Val Accuracy': [f"{results[m]['val_accuracy']:.4f}" for m in results],
            'Val F1': [f"{results[m]['val_f1_score']:.4f}" for m in results],
            'Test Accuracy': [f"{results[m]['accuracy']:.4f}" for m in results],
            'Test F1': [f"{results[m]['f1_score']:.4f}" for m in results]
        })
        st.dataframe(comparison_df)

        # Best model
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        best_model = models[best_model_name]
        st.success(f"🏆 Best Model: **{best_model_name}** (F1: {results[best_model_name]['f1_score']:.4f})")

        # Comparison chart
        fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
        x = np.arange(len(results))
        width = 0.35
        bars1 = ax_comp.bar(x - width/2, [results[m]['accuracy'] for m in results], width, label='Accuracy', color='#3498db')
        bars2 = ax_comp.bar(x + width/2, [results[m]['f1_score'] for m in results], width, label='F1-Score', color='#e74c3c')
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(list(results.keys()))
        ax_comp.legend()
        ax_comp.set_ylim(0, 1)
        ax_comp.set_title("Comparison of 3 Models")
        for bar in bars1:
            ax_comp.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax_comp.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_comp)

        # Confusion Matrices
        st.subheader("📈 Confusion Matrices")
        fig_cm, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, (name, model) in enumerate(models.items()):
            cm = confusion_matrix(y_test, results[name]['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[idx])
            axes[idx].set_title(f'{name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig_cm)

        # =========================================================
        # SAVE MODELS
        # =========================================================
        os.makedirs("export", exist_ok=True)
        
        # Save best model as main model
        joblib.dump(best_model, "export/model.pkl")
        joblib.dump(vectorizer, "export/vectorizer.pkl")
        joblib.dump(le, "export/label_encoder.pkl")
        
        # Save all models
        for name, model in models.items():
            safe_name = name.lower().replace(' ', '_')
            joblib.dump(model, f"export/model_{safe_name}.pkl")

        # Save train info
        train_info = {
            "best_model": best_model_name,
            "accuracy": float(results[best_model_name]['accuracy']),
            "f1_score": float(results[best_model_name]['f1_score']),
            "model_name": best_model.__class__.__name__,
            "num_samples": len(df_balanced),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "num_classes": len(le.classes_),
            "labels": list(le.classes_),
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "all_models": {
                name: {"accuracy": float(results[name]['accuracy']), "f1_score": float(results[name]['f1_score'])}
                for name in results
            }
        }
        with open("export/train_info.json", "w", encoding="utf-8") as f:
            json.dump(train_info, f, indent=4, ensure_ascii=False)

        # Save confusion matrices
        fig_cm.savefig("export/confusion_matrix_all_models.png", dpi=150, bbox_inches='tight')
        fig_comp.savefig("export/model_comparison.png", dpi=150, bbox_inches='tight')

        st.success("📦 All models saved to export/ folder!")

    # =========================================================
    # 3️⃣ PREDICTION
    # =========================================================
    st.write("---")
    st.header("3️⃣ Prediction")

    txt = st.text_area("Enter news content...")

    if st.button("🔮 Predict"):
        if not os.path.exists("export/model.pkl"):
            st.error("❌ No model found. Please train first.")
            return

        model = joblib.load("export/model.pkl")
        vec = joblib.load("export/vectorizer.pkl")
        le = joblib.load("export/label_encoder.pkl")

        vec_txt = vec.transform([preprocess_text(txt)])
        pred = model.predict(vec_txt)[0]
        label = le.inverse_transform([pred])[0]

        st.success(f"➡ Prediction result: **{label}**")
