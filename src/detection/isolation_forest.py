import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
import joblib
import os

# ── 1. Load cleaned data ───────────────────────────────────────────
df = pd.read_csv("data/processed/cleaned.csv")

# ── 2. Drop useless features identified in EDA ────────────────────
cols_to_drop = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "RST Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"
]

# ── 3. Separate features from labels ──────────────────────────────
# X = all numeric features we'll feed to the model
# y = the binary label (0=benign, 1=attack) for evaluation only
X = df.drop(columns=cols_to_drop + ["Label", "is_attack"])
y = df["is_attack"]

# ── 4. Split into benign-only training set and full test set ──────
# We train ONLY on benign traffic — the model learns what normal looks like
# We test on ALL traffic so we can see if it catches attacks
benign_mask = y == 0
X_train = X[benign_mask]
X_test = X
y_test = y

print(f"Training on {len(X_train)} benign rows")
print(f"Testing on {len(X_test)} total rows")

# ── 5. Scale features to 0-1 range ────────────────────────────────
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 6. Train Isolation Forest ──────────────────────────────────────
# contamination = estimated fraction of anomalies in the data
# We have 556k attacks out of 2.8M rows ≈ 0.02 (2%)
# n_estimators = number of trees (100 is standard)
# random_state = makes results reproducible
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
    n_jobs=-1  # use all CPU cores to speed up training
)
iso_forest.fit(X_train_scaled)
print("Training complete.")

# ── 7. Get predictions on test set ────────────────────────────────
print("\nScoring test set...")
raw_predictions = iso_forest.predict(X_test_scaled)

# Convert sklearn format (1=normal, -1=anomaly) to our format (0=normal, 1=attack)
y_pred = (raw_predictions == -1).astype(int)

# ── 8. Get raw anomaly scores instead of binary predictions ───────
print("\nScoring test set...")
anomaly_scores = iso_forest.decision_function(X_test_scaled)
# More negative score = more anomalous

# ── 9. Calibrate threshold using a small labeled sample ───────────
# We take 1000 random attack examples and 1000 random benign examples
# and find the threshold that maximizes F1 score


attack_scores = anomaly_scores[y_test == 1]
benign_scores = anomaly_scores[y_test == 0]

# Sample 1000 from each to find the best threshold
np.random.seed(42)
sample_attack = np.random.choice(attack_scores, size=min(1000, len(attack_scores)), replace=False)
sample_benign = np.random.choice(benign_scores, size=1000, replace=False)

sample_scores = np.concatenate([sample_attack, sample_benign])
sample_labels = np.concatenate([np.ones(len(sample_attack)), np.zeros(len(sample_benign))])

# Try 100 different threshold values and pick the best one
best_threshold = 0
best_f1 = 0
for threshold in np.linspace(anomaly_scores.min(), anomaly_scores.max(), 100):
    preds = (sample_scores < threshold).astype(int)
    f1 = fbeta_score(sample_labels, preds, beta=0.5)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.4f}")
print(f"Best F1 on calibration sample: {best_f1:.4f}")

# ── 10. Apply calibrated threshold to full test set ───────────────
y_pred_calibrated = (anomaly_scores < best_threshold).astype(int)

# ── 11. Evaluate calibrated results ───────────────────────────────
print("\n--- Isolation Forest Results (Calibrated) ---")
print(classification_report(y_test, y_pred_calibrated, target_names=["Benign", "Attack"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_calibrated))

# Save the threshold so we can use it later
threshold_path = "src/detection/models/iso_forest_threshold.npy"
np.save(threshold_path, best_threshold)
print(f"\nThreshold saved to {threshold_path}")

# ── 9. Save model and scaler ───────────────────────────────────────
# joblib is the standard way to save sklearn models
# We save the scaler too because when new data comes in we need
# to scale it the same way we scaled the training data
os.makedirs("src/detection/models", exist_ok=True)
joblib.dump(iso_forest, "src/detection/models/isolation_forest.pkl")
joblib.dump(scaler, "src/detection/models/scaler.pkl")
print("\nModel and scaler saved to src/detection/models/")

