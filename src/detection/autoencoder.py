import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import os


# ── 1. Define the Autoencoder architecture ────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super(Autoencoder, self).__init__()

        # Change architecture - bigger bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # forward() defines how data flows through the network
        # PyTorch calls this automatically during training
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ── 2. Load and prepare data ───────────────────────────────────────
df = pd.read_csv("data/processed/cleaned.csv")

cols_to_drop = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "RST Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"
]

X = df.drop(columns=cols_to_drop + ["Label", "is_attack"])
y = df["is_attack"]

# Train only on benign traffic
benign_mask = y == 0
X_train = X[benign_mask]
X_test = X

# Scale features
scaler = joblib.load("src/detection/models/scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 3. Convert to PyTorch tensors ─────────────────────────────────
# PyTorch works with tensors not numpy arrays
# float32 is standard for neural network training
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)

# ── 4. Create DataLoader ───────────────────────────────────────────
# DataLoader handles batching automatically
# Instead of feeding all 2.2M rows at once we feed 256 rows at a time
# This is called mini-batch gradient descent
batch_size = 256
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True  # shuffle so the model doesn't memorize row order
)

# ── 5. Initialize model, loss function, and optimizer ─────────────
input_dim = X_train_scaled.shape[1]
model = Autoencoder(input_dim=input_dim)

# MSE loss = mean squared error between input and reconstruction
# Perfect for autoencoders since we're comparing original vs reconstructed
criterion = nn.MSELoss()

# Adam optimizer — standard choice for deep learning
# lr = learning rate: how big each weight update step is
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Model input dimension: {input_dim}")
print(f"Training on {len(X_train_tensor)} benign rows")
print(f"Batch size: {batch_size}")
print(f"Batches per epoch: {len(train_loader)}")

# ── 6. Training loop ───────────────────────────────────────────────
# An epoch = one full pass through the training data
# We run 20 epochs — enough to learn normal patterns without overfitting
epochs = 50

print("\nStarting training...")
for epoch in range(epochs):
    model.train()  # puts model in training mode (enables dropout etc if used)
    total_loss = 0

    for batch_input, batch_target in train_loader:
        # Step 1: forward pass — get reconstruction
        reconstruction = model(batch_input)

        # Step 2: compute loss — how different is reconstruction from input?
        loss = criterion(reconstruction, batch_target)

        # Step 3: zero gradients from previous batch
        # PyTorch accumulates gradients by default so we reset each batch
        optimizer.zero_grad()

        # Step 4: backward pass — compute gradients
        loss.backward()

        # Step 5: update weights using computed gradients
        optimizer.step()

        total_loss += loss.item()

    # Print progress every 5 epochs
    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs} — Average Loss: {avg_loss:.6f}")

print("Training complete.")

# ── 7. Compute reconstruction errors on test set ──────────────────
model.eval()  # puts model in evaluation mode — disables dropout etc
with torch.no_grad():  # disables gradient tracking — saves memory during inference
    X_test_reconstructed = model(X_test_tensor)

# Compute reconstruction error for each row
# Mean squared error between original and reconstructed features
reconstruction_errors = torch.mean(
    (X_test_tensor - X_test_reconstructed) ** 2, dim=1
).numpy()

# ── 8. Calibrate threshold using labeled sample ───────────────────
from sklearn.metrics import fbeta_score

attack_errors = reconstruction_errors[y.values == 1]
benign_errors = reconstruction_errors[y.values == 0]

np.random.seed(42)
sample_attack = np.random.choice(attack_errors, size=min(1000, len(attack_errors)), replace=False)
sample_benign = np.random.choice(benign_errors, size=1000, replace=False)

sample_errors = np.concatenate([sample_attack, sample_benign])
sample_labels = np.concatenate([np.ones(len(sample_attack)), np.zeros(len(sample_benign))])

best_threshold = 0
best_f1 = 0
for threshold in np.linspace(reconstruction_errors.min(), reconstruction_errors.max(), 100):
    preds = (sample_errors > threshold).astype(int)
    f1 = fbeta_score(sample_labels, preds, beta=0.5)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest threshold: {best_threshold:.6f}")
print(f"Best F1 on calibration sample: {best_f1:.4f}")

# ── 9. Evaluate on full test set ──────────────────────────────────
y_pred = (reconstruction_errors > best_threshold).astype(int)

print("\n--- Autoencoder Results ---")
print(classification_report(y.values, y_pred, target_names=["Benign", "Attack"]))
print("Confusion Matrix:")
print(confusion_matrix(y.values, y_pred))

# ── 10. Save model and threshold ──────────────────────────────────
os.makedirs("src/detection/models", exist_ok=True)
torch.save(model.state_dict(), "src/detection/models/autoencoder.pt")
np.save("src/detection/models/autoencoder_threshold.npy", best_threshold)
print("\nAutoencoder and threshold saved.")