import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load data ===
data = np.load("../data/so101_dynamics.npz")
q = data["q"][:, :5]
dq = data["dq"][:, :5]
ddq = data["ddq"][:, :5]
tau = data["tau"][:, :5]

# === Inputs and Target Approximation
X_np = np.hstack([q, dq])  # shape (N, 10)
y_np = np.zeros((len(q), 25))  # shape (N, 25)

# Better M estimation via least-squares for each row
for i in range(len(q)):
    try:
        M_i, _, _, _ = np.linalg.lstsq(ddq[i].reshape(5, 1), tau[i].reshape(5, 1), rcond=None)
        M_full = M_i @ ddq[i].reshape(1, 5)  # Reconstruct full M
        y_np[i] = M_full.flatten()
    except:
        y_np[i] = np.zeros(25)

# === Normalize targets
y_mean = y_np.mean(axis=0)
y_std = y_np.std(axis=0) + 1e-8
y_np_norm = (y_np - y_mean) / y_std

# === Torch dataset
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np_norm, dtype=torch.float32)
dataset = TensorDataset(X, y)

# === Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# === Model (Smaller + Dropout)
class MassMatrixNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 25)
        )

    def forward(self, x):
        return self.net(x)

model = MassMatrixNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# === Training loop with early stopping
EPOCHS = 50
train_losses = []
val_losses = []

best_val_loss = float("inf")
patience = 30
wait = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * xb.size(0)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_val_loss += loss_fn(pred, yb).item() * xb.size(0)

    train_loss = total_train_loss / len(train_loader.dataset)
    val_loss = total_val_loss / len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"[{epoch:04d}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # === Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        torch.save(model.state_dict(), "../models/so101_mass_matrix_net_best.pt")
    else:
        wait += 1
        if wait > patience:
            print("Early stopping triggered.")
            break

# === Save final model + stats
torch.save(model.state_dict(), "../models/so101_mass_matrix_net_last.pt")
np.savez("../models/so101_mass_matrix_norm_stats.npz", mean=y_mean, std=y_std)

# === Plot losses
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Mass Matrix Estimation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../plots/mass_matrix_loss_curve.png")
plt.show()
