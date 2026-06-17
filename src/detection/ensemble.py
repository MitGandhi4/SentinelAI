import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import torch
import mlflow
import mlflow.sklearn
import os
import sys

# Add project root to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.detection.autoencoder import Autoencoder

# ── 1. Load cleaned data ───────────────────────────────────────────
df = pd.read_csv("data/processed/cleaned.csv")

cols_to_drop = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "RST Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"
]

X = df.drop(columns=cols_to_drop + ["Label", "is_attack"])
y = df["is_attack"]

# ── 2. Load scaler and scale features ─────────────────────────────
scaler = joblib.load("src/detection/models/scaler.pkl")
X_scaled = scaler.transform(X)

# ── 3. Load Isolation Forest and get predictions ───────────────────
print("Loading Isolation Forest...")
iso_forest = joblib.load("src/detection/models/isolation_forest.pkl")
iso_threshold = np.load("src/detection/models/iso_forest_threshold.npy")

iso_scores = iso_forest.decision_function(X_scaled)
iso_preds = (iso_scores < iso_threshold).astype(int)
print(f"Isolation Forest flagged {iso_preds.sum()} rows as attacks")

# ── 4. Load Autoencoder and get predictions ────────────────────────
print("Loading Autoencoder...")
input_dim = X_scaled.shape[1]
autoencoder = Autoencoder(input_dim=input_dim)
autoencoder.load_state_dict(torch.load("src/detection/models/autoencoder.pt"))
autoencoder.eval()

X_tensor = torch.FloatTensor(X_scaled)
with torch.no_grad():
    reconstructed = autoencoder(X_tensor)

recon_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
ae_threshold = np.load("src/detection/models/autoencoder_threshold.npy")
ae_preds = (recon_errors > ae_threshold).astype(int)
print(f"Autoencoder flagged {ae_preds.sum()} rows as attacks")


# ── 5. Ensemble predictions ────────────────────────────────────────
# OR logic — flag if EITHER model flags it (higher recall)
ensemble_or = ((iso_preds == 1) | (ae_preds == 1)).astype(int)

# AND logic — flag if BOTH models flag it (higher precision)
ensemble_and = ((iso_preds == 1) & (ae_preds == 1)).astype(int)

# ── 6. MLflow experiment tracking ─────────────────────────────────
# MLflow logs all our runs so we can compare them later
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("SentinelAI Anomaly Detection")

# ── Run 1: Isolation Forest alone ─────────────────────────────────
with mlflow.start_run(run_name="Isolation Forest"):
    from sklearn.metrics import precision_score, recall_score, f1_score
    mlflow.log_metric("precision", precision_score(y, iso_preds))
    mlflow.log_metric("recall", recall_score(y, iso_preds))
    mlflow.log_metric("f1", f1_score(y, iso_preds))
    mlflow.log_param("model", "IsolationForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("contamination", 0.02)

# ── Run 2: Autoencoder alone ───────────────────────────────────────
with mlflow.start_run(run_name="Autoencoder"):
    mlflow.log_metric("precision", precision_score(y, ae_preds))
    mlflow.log_metric("recall", recall_score(y, ae_preds))
    mlflow.log_metric("f1", f1_score(y, ae_preds))
    mlflow.log_param("model", "Autoencoder")
    mlflow.log_param("epochs", 50)
    mlflow.log_param("bottleneck", 16)

# ── Run 3: Ensemble OR ─────────────────────────────────────────────
with mlflow.start_run(run_name="Ensemble OR"):
    mlflow.log_metric("precision", precision_score(y, ensemble_or))
    mlflow.log_metric("recall", recall_score(y, ensemble_or))
    mlflow.log_metric("f1", f1_score(y, ensemble_or))
    mlflow.log_param("model", "Ensemble OR")
    mlflow.log_param("logic", "OR")

# ── Run 4: Ensemble AND ────────────────────────────────────────────
with mlflow.start_run(run_name="Ensemble AND"):
    mlflow.log_metric("precision", precision_score(y, ensemble_and))
    mlflow.log_metric("recall", recall_score(y, ensemble_and))
    mlflow.log_metric("f1", f1_score(y, ensemble_and))
    mlflow.log_param("model", "Ensemble AND")
    mlflow.log_param("logic", "AND")

# ── 7. Print all results ───────────────────────────────────────────
print("\n--- Isolation Forest ---")
print(classification_report(y, iso_preds, target_names=["Benign", "Attack"]))

print("\n--- Autoencoder ---")
print(classification_report(y, ae_preds, target_names=["Benign", "Attack"]))

print("\n--- Ensemble OR ---")
print(classification_report(y, ensemble_or, target_names=["Benign", "Attack"]))

print("\n--- Ensemble AND ---")
print(classification_report(y, ensemble_and, target_names=["Benign", "Attack"]))
