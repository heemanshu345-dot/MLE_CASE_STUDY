import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report, precision_score, recall_score,
    f1_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "diabetic_data.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Convert '?' to missing values
missing_tokens = ["?", "?"]
df = df.replace(missing_tokens, np.nan)

# -----------------------------
# 2. Target definition
# -----------------------------
# Positive class = readmission within 30 days (<30)
df["target"] = (df["readmitted"] == "<30").astype(int)

print("\nTarget distribution:")
print(df["target"].value_counts().rename({0: "Not readmitted within 30 days", 1: "Readmitted within 30 days"}))

# -----------------------------
# 3. Feature selection
# -----------------------------
# Exclude identifiers and the target/leakage column.
# patient_nbr is retained only for grouped splitting, not as a predictor.
exclude = ["encounter_id", "patient_nbr", "readmitted", "target"]
X = df.drop(columns=exclude)
y = df["target"]
groups = df["patient_nbr"]

# Treat coded hospital IDs as categorical rather than continuous.
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# -----------------------------
# 4. Patient-level train/test split
# -----------------------------
# Prevent the same patient from appearing in both train and test.
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate:  {y_test.mean():.4f}")

# -----------------------------
# 5. Preprocessing
# -----------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# -----------------------------
# 6. Logistic Regression + L2
# -----------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=42
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

# -----------------------------
# 7. Predictions and evaluation
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")

# -----------------------------
# 8. Threshold analysis
# -----------------------------
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
rows = []

for threshold in thresholds:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    rows.append({
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity_recall": recall_score(y_test, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0,
        "precision": precision_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0)
    })

threshold_df = pd.DataFrame(rows)
threshold_df.to_csv(os.path.join(RESULTS_DIR, "threshold_analysis.csv"), index=False)
print("\nThreshold analysis:")
print(threshold_df.to_string(index=False))

# Default 0.50 classification report
pred_05 = (y_prob >= 0.50).astype(int)
print("\nClassification report @ threshold 0.50:")
print(classification_report(y_test, pred_05, target_names=["No 30-day readmission", "30-day readmission"], zero_division=0))

# -----------------------------
# 9. ROC curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - 30-Day Hospital Readmission")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=160)
plt.close()

# -----------------------------
# 10. Precision-Recall curve
# -----------------------------
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"AP = {avg_precision:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - 30-Day Hospital Readmission")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"), dpi=160)
plt.close()

# -----------------------------
# 11. Confusion matrix at 0.20
# -----------------------------
clinical_threshold = 0.20
pred_20 = (y_prob >= clinical_threshold).astype(int)
cm = confusion_matrix(y_test, pred_20, labels=[0, 1])

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - Threshold 0.20")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["No", "Readmitted <30"])
plt.yticks([0, 1], ["No", "Readmitted <30"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_threshold_020.png"), dpi=160)
plt.close()

# -----------------------------
# 12. Save metrics
# -----------------------------
metrics = {
    "dataset_rows": len(df),
    "dataset_columns": 50,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "test_positive_rate": float(y_test.mean()),
    "roc_auc": float(roc_auc),
    "average_precision": float(avg_precision),
    "clinical_threshold_example": clinical_threshold,
    "clinical_threshold_note": "Example threshold only; select using clinical costs and validation."
}

pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

# -----------------------------
# 13. Save model coefficients
# -----------------------------
try:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients)
    }).sort_values("absolute_coefficient", ascending=False)
    coef_df.to_csv(os.path.join(RESULTS_DIR, "model_coefficients.csv"), index=False)
except Exception as exc:
    print(f"Could not save coefficients: {exc}")

print("\nResults saved in ./results/")
print("Done.")
