import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load data ===
data = np.load("../data/so101_dynamics.npz")
q = data["q"][:, :5]     # Exclude joint 6
dq = data["dq"][:, :5]
ddq = data["ddq"][:, :5]
tau = data["tau"][:, :5]

# === Prepare dataset ===
X_np = np.hstack([q, dq, ddq])   # Shape: (N, 15)
y_np = tau                       # Shape: (N, 5)

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

dataset = TensorDataset(X, y)

# === Split into train and validation ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# === Model definition ===
class TauNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.net(x)

model = TauNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# === Training loop ===
EPOCHS = 200
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item() * xb.size(0)

    train_losses.append(train_loss / len(train_loader.dataset))
    val_losses.append(val_loss / len(val_loader.dataset))

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"[{epoch:04d}] Train: {train_losses[-1]:.6f} | Val: {val_losses[-1]:.6f}")

# === Save model ===
torch.save(model.state_dict(), "../models/tau_net.pt")
print("✔ Saved model as tau_net.pt")

# === Plot losses ===
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Tau Prediction: Train vs Val Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../plots/tau_net_loss_curve.png")
plt.show()
