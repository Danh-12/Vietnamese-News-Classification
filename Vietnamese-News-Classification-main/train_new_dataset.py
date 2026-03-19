"""
Script huấn luyện mô hình phân loại tin tức tiếng Việt
Sử dụng dataset mới từ Kaggle: sarahhimeko/vietnamese-online-news-csv-dataset
"""

import pandas as pd
import numpy as np
import os
import re
import json
import kagglehub
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Đảm bảo thư mục export tồn tại
os.makedirs('export', exist_ok=True)

print("="*70)
print("🚀 VIETNAMESE NEWS CLASSIFICATION - NEW DATASET")
print("="*70)

# =========================================================
# 1. LOAD DATA FROM KAGGLE
# =========================================================
print("\n📁 Loading dataset from Kaggle...")

# Download dataset first
path = kagglehub.dataset_download("sarahhimeko/vietnamese-online-news-csv-dataset")
print(f"   Downloaded to: {path}")

# Find CSV file
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"   Found files: {csv_files}")

if csv_files:
    df = pd.read_csv(os.path.join(path, csv_files[0]))
else:
    # Try to find any file
    all_files = os.listdir(path)
    print(f"   All files in directory: {all_files}")
    raise ValueError("No CSV file found in dataset")

print(f"✅ Loaded {len(df):,} articles")
print(f"   Columns: {list(df.columns)}")

# =========================================================
# 2. TIỀN XỬ LÝ
# =========================================================
print("\n📝 Preprocessing...")

# Stopwords tiếng Việt - Danh sách đầy đủ
STOPWORDS = {
    # Đại từ nhân xưng
    "tôi", "tao", "ta", "mình", "chúng_tôi", "chúng_ta", "chúng_mình",
    "bạn", "cậu", "mày", "các_bạn", "các_cậu",
    "anh", "chị", "em", "ông", "bà", "cô", "chú", "bác", "dì", "cháu",
    "họ", "nó", "hắn", "người_ta", "ai", "gì",
    # Đại từ chỉ định
    "này", "kia", "đó", "ấy", "nọ", "đây", "kìa",
    "thế_này", "thế_kia", "như_thế", "như_vậy",
    # Liên từ
    "và", "với", "cùng", "hoặc", "hay", "nhưng", "mà", "còn", "song",
    "tuy", "tuy_nhiên", "tuy_vậy", "tuy_thế", "mặc_dù", "dù", "dẫu",
    "nếu", "giá", "giá_mà", "miễn_là", "trừ_phi", 
    "vì", "bởi", "bởi_vì", "do", "vì_vậy", "vì_thế", "cho_nên", "nên",
    "để", "để_mà", "hầu", "ngõ_hầu",
    # Giới từ
    "của", "cho", "từ", "đến", "tại", "ở", "trong", "ngoài",
    "trên", "dưới", "trước", "sau", "giữa", "bên", "cạnh",
    "về", "theo", "qua", "bằng", "vào", "ra",
    # Trợ từ
    "là", "được", "bị", "có", "không", "chẳng", "chả", "đâu",
    "à", "ư", "ừ", "nhé", "nhỉ", "chứ", "đấy", "thế", "vậy",
    "đã", "đang", "sẽ", "vừa", "mới", "từng", "cứ", "vẫn",
    "rồi", "xong", "hết", "xem", "thử",
    # Số từ và lượng từ
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "mấy", "bao_nhiêu", "vài", "dăm", "một_vài", "nhiều", "ít",
    "mỗi", "mọi", "tất_cả", "cả", "toàn_bộ",
    "những", "các", "một_số", "số",
    # Phó từ
    "rất", "quá", "lắm", "hơi", "khá", "cực", "cực_kỳ", "vô_cùng",
    "thật", "thật_sự", "thực_sự", "đúng", "đúng_là",
    "cũng", "đều", "lại", "nữa", "thêm", "hơn", "kém", "nhất",
    "chỉ", "chỉ_là", "duy", "riêng", "đặc_biệt",
    "luôn", "luôn_luôn", "thường", "thường_xuyên",
    "chưa",
    # Từ chỉ thời gian
    "khi", "lúc", "hồi", "bây_giờ", "giờ", "nay", "hiện",
    "sáng", "trưa", "chiều", "tối", "đêm",
    "hôm", "ngày", "tháng", "tuần",
    "hôm_nay", "hôm_qua", "ngày_mai", "ngày_kia",
    # Từ chỉ nơi chốn
    "nơi", "chỗ", "nào", "ở_đây", "ở_đó", "ở_kia", "ở_đâu",
    # Từ nghi vấn
    "sao", "bao_giờ", "bao_lâu", "tại_sao", "vì_sao", "làm_sao", 
    "thế_nào", "như_thế_nào", "có_phải", "phải_không", "chăng",
    # Từ khác thường gặp
    "thì", "như", "phải", "làm", "biết", "thấy", "muốn", "cần",
    "có_thể", "không_thể", "chắc", "chắc_chắn", "có_lẽ", "hình_như",
    "dường_như", "hầu_như", "gần_như",
    "việc", "điều", "cái", "con", "người", "sự",
    "ví_dụ", "chẳng_hạn", "thực_ra", "thật_ra", "nói_chung",
    "tóm_lại", "nói_tóm_lại", "cuối_cùng",
    # Từ phổ biến trong báo chí
    "cho_biết", "cho_hay", "chia_sẻ", "nhận_định",
    "nguồn_tin", "liên_quan", "về_việc", "đối_với",
}

def preprocess_text(text):
    """Tiền xử lý văn bản với Vietnamese word segmentation"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Vietnamese word segmentation với pyvi
    try:
        text = ViTokenizer.tokenize(text)
    except:
        pass
    
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(words)

# Xác định cột text và label
text_col = None
label_col = None

for col in ['content', 'text', 'article', 'body']:
    if col in df.columns:
        text_col = col
        break

for col in ['topic', 'label', 'category', 'class']:
    if col in df.columns:
        label_col = col
        break

if text_col is None:
    text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
if label_col is None:
    label_col = df.columns[0] if len(df.columns) > 1 else df.columns[0]

print(f"   Text column: {text_col}")
print(f"   Label column: {label_col}")

# Thêm title nếu có
if 'title' in df.columns:
    df['full_text'] = (df['title'].fillna('') + ' ') * 3 + df[text_col].fillna('')
else:
    df['full_text'] = df[text_col].fillna('')

print("   Preprocessing text (this may take a while)...")
df['clean_text'] = df['full_text'].apply(preprocess_text)

df_clean = df[df['clean_text'].str.len() > 50].copy()
print(f"   After preprocessing: {len(df_clean):,} articles")

# =========================================================
# 3. LỌC VÀ CÂN BẰNG DỮ LIỆU
# =========================================================
print("\n⚖️ Filtering and balancing data...")

MIN_SAMPLES = 500
topic_counts = df_clean[label_col].value_counts()
valid_topics = topic_counts[topic_counts >= MIN_SAMPLES].index.tolist()
df_balanced = df_clean[df_clean[label_col].isin(valid_topics)].copy()

MAX_SAMPLES = 8000
balanced_dfs = []
for topic in df_balanced[label_col].unique():
    topic_df = df_balanced[df_balanced[label_col] == topic]
    if len(topic_df) > MAX_SAMPLES:
        topic_df = topic_df.sample(MAX_SAMPLES, random_state=42)
    balanced_dfs.append(topic_df)

df_balanced = pd.concat(balanced_dfs, ignore_index=True)
print(f"   Balanced dataset: {len(df_balanced):,} articles, {df_balanced[label_col].nunique()} topics")

# =========================================================
# 4. CHIA DỮ LIỆU
# =========================================================
print("\n🔀 Splitting data...")

X = df_balanced['clean_text'].values
y = df_balanced[label_col].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Classes: {len(le.classes_)}")
print(f"   Labels: {list(le.classes_)}")

# =========================================================
# 5. TF-IDF VECTORIZATION
# =========================================================
print("\n🔢 Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 2), min_df=3, max_df=0.85, sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"   Features: {X_train_vec.shape[1]:,}")

# =========================================================
# 6. HUẤN LUYỆN MÔ HÌNH
# =========================================================
print("\n" + "="*70)
print("🤖 TRAINING MODELS")
print("="*70)

models = {}
results = {}

# XGBoost
print("\n📌 [1/3] Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='mlogloss'
)
xgb_model.fit(X_train_vec, y_train)
y_pred_xgb = xgb_model.predict(X_test_vec)
results['XGBoost'] = {'accuracy': accuracy_score(y_test, y_pred_xgb), 'f1_score': f1_score(y_test, y_pred_xgb, average='weighted'), 'predictions': y_pred_xgb}
models['XGBoost'] = xgb_model
print(f"   ✅ XGBoost: Accuracy = {results['XGBoost']['accuracy']:.4f}, F1 = {results['XGBoost']['f1_score']:.4f}")

# Logistic Regression
print("\n📌 [2/3] Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', class_weight='balanced', n_jobs=-1, random_state=42)
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)
results['Logistic Regression'] = {'accuracy': accuracy_score(y_test, y_pred_lr), 'f1_score': f1_score(y_test, y_pred_lr, average='weighted'), 'predictions': y_pred_lr}
models['Logistic Regression'] = lr_model
print(f"   ✅ Logistic Regression: Accuracy = {results['Logistic Regression']['accuracy']:.4f}, F1 = {results['Logistic Regression']['f1_score']:.4f}")

# SVM
print("\n📌 [3/3] Training SVM...")
svm_model = LinearSVC(C=0.5, max_iter=3000, class_weight='balanced', random_state=42)
svm_model.fit(X_train_vec, y_train)
y_pred_svm = svm_model.predict(X_test_vec)
results['SVM'] = {'accuracy': accuracy_score(y_test, y_pred_svm), 'f1_score': f1_score(y_test, y_pred_svm, average='weighted'), 'predictions': y_pred_svm}
models['SVM'] = svm_model
print(f"   ✅ SVM: Accuracy = {results['SVM']['accuracy']:.4f}, F1 = {results['SVM']['f1_score']:.4f}")

# =========================================================
# 7. BEST MODEL & SAVE
# =========================================================
print("\n" + "="*70)
print("📊 MODEL COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'F1-Score': [results[m]['f1_score'] for m in results]
}).sort_values('F1-Score', ascending=False)
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']
print(f"\n🏆 Best Model: {best_model_name}")

# =========================================================
# 8. LƯU KẾT QUẢ
# =========================================================
print("\n📝 Saving results to export/...")

# Save models
joblib.dump(best_model, 'export/model.pkl')
joblib.dump(vectorizer, 'export/vectorizer.pkl')
joblib.dump(le, 'export/label_encoder.pkl')

for name, model in models.items():
    joblib.dump(model, f"export/model_{name.lower().replace(' ', '_')}.pkl")

# Reports
for name in results:
    report = classification_report(y_test, results[name]['predictions'], target_names=le.classes_, output_dict=False)
    with open(f"export/report_{name.lower().replace(' ', '_')}.txt", "w", encoding="utf-8") as f:
        f.write(f"Classification Report - {name}\n{'='*60}\n\n{report}")

# Training info
train_info = {
    "best_model": best_model_name,
    "accuracy": float(comparison_df.iloc[0]['Accuracy']),
    "f1_score": float(comparison_df.iloc[0]['F1-Score']),
    "model_name": best_model.__class__.__name__,
    "num_samples": len(df_balanced),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "num_classes": len(le.classes_),
    "labels": list(le.classes_),
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": "sarahhimeko/vietnamese-online-news-csv-dataset",
    "all_models": {name: {"accuracy": float(results[name]['accuracy']), "f1_score": float(results[name]['f1_score'])} for name in results}
}
with open("export/train_info.json", "w", encoding="utf-8") as f:
    json.dump(train_info, f, ensure_ascii=False, indent=2)

# =========================================================
# 9. CHARTS
# =========================================================
print("\n📈 Creating charts...")

# Model comparison
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results))
width = 0.35
ax.bar(x - width/2, [results[m]['accuracy'] for m in results], width, label='Accuracy', color='#3498db')
ax.bar(x + width/2, [results[m]['f1_score'] for m in results], width, label='F1-Score', color='#e74c3c')
ax.set_xticks(x)
ax.set_xticklabels(list(results.keys()))
ax.legend()
ax.set_ylim(0, 1)
ax.set_title('Model Comparison')
plt.tight_layout()
plt.savefig('export/model_comparison.png', dpi=150)
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, (name, _) in enumerate(models.items()):
    cm = confusion_matrix(y_test, results[name]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[idx])
    axes[idx].set_title(name)
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('export/confusion_matrix_all_models.png', dpi=150)
plt.close()

# Individual CMs
for name in models:
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, results[name]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f'Confusion Matrix - {name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"export/confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()

# Topic distribution
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=df_balanced[label_col].value_counts().index, y=df_balanced[label_col].value_counts().values, ax=ax, palette='viridis')
ax.set_title('Topic Distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('export/topic_distribution.png', dpi=150)
plt.close()

print("\n" + "="*70)
print("✅ TRAINING COMPLETED!")
print("="*70)
print(f"\n🎉 Best Model: {best_model_name} (Acc: {comparison_df.iloc[0]['Accuracy']:.4f}, F1: {comparison_df.iloc[0]['F1-Score']:.4f})")
print("\n📁 Files saved to export/")
