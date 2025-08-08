from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_FILE = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_FILE = Path(__file__).resolve().parent / "model.pkl"

# === Load dataset ===
print("📥 Loading feature data from:", FEATURE_FILE)
df = pd.read_csv(FEATURE_FILE)

# === Preprocess ===
X = df.drop(columns=["filename", "label"])
y = df["label"].map({"drone": 1, "bird": 0})  # Ensure consistent label encoding

# === Split into train and test sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Load trained model ===
print("📦 Loading model from:", MODEL_FILE)
model = joblib.load(MODEL_FILE)

# === Predict and evaluate ===
y_pred = model.predict(X_test)

print("\n📊 Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["bird", "drone"]))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Optional: Visualize confusion matrix ===
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# Plot it
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, labels=["bird", "drone"])
