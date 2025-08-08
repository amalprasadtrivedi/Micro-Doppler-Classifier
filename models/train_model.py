from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_FILE = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

# === Load dataset ===
print("📥 Loading features from:", FEATURE_FILE)
df = pd.read_csv(FEATURE_FILE)

# === Preprocess features and labels ===
# Drop unnecessary columns
X = df.drop(columns=["filename", "label"])
y = df["label"]

# Encode labels (if needed)
y = y.map({"drone": 1, "bird": 0})  # Binary classification

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
print("🧠 Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate model ===
y_pred = clf.predict(X_test)

print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["bird", "drone"]))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save trained model ===
joblib.dump(clf, MODEL_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")
