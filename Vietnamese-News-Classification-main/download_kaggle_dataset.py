# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
print("📁 Loading dataset from Kaggle...")
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "sarahhimeko/vietnamese-online-news-csv-dataset",
    file_path,
)

print(f"✅ Loaded {len(df):,} records")
print(f"   Columns: {list(df.columns)}")
print("\nFirst 5 records:")
print(df.head())

# Save to CSV
df.to_csv("Fixed_news_dataset.csv", index=False, encoding="utf-8")
print(f"\n✅ Saved to Fixed_news_dataset.csv")
