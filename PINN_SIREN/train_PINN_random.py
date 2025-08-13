import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# === Load dataset (for q0/qT sampling and normalizers) ===
data = np.load("../data/so101_dynamics.npz")
q_data = data["q"][:, :5]

# === f model ===
class FModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
    def forward(self, x): return self.net(x)

f_model = FModel().to(device)
f_model.load_state_dict(torch.load("../models/tau_net.pt"))
f_model.eval()

# === M model ===
class MModel(nn.Module):
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
    def forward(self, x): return self.net(x)

mass_model = MModel().to(device)
mass_model.load_state_dict(torch.load("../models/so101_mass_matrix_net_best.pt"))
mass_model.eval()

stats = np.load("../models/so101_mass_matrix_norm_stats.npz")
y_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
y_std = torch.tensor(stats["std"], dtype=torch.float32, device=device)

# === SIREN trajectory model ===
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=5.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights(is_first)
    def init_weights(self, is_first):
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class TrajectoryPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(11, 64, is_first=True),
            SineLayer(64, 64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    def forward(self, t, q0, qT):
        B, T = t.shape[:2]
        q0_expand = q0.unsqueeze(1).expand(-1, T, -1)
        qT_expand = qT.unsqueeze(1).expand(-1, T, -1)
        inp = torch.cat([t, q0_expand, qT_expand], dim=-1)
        return self.net(inp.view(-1, 11)).view(B, T, 5)

model = TrajectoryPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

# === Training settings
EPOCHS = 3000
BATCH_SIZE = 64
boundary_weight = 10.0
residual_weight = 1e-3  # Encourage residuals to not be zero
q_scale = 1.0
tau_scale = 10.0

# === Time ===
t = np.linspace(0, 1, 200, dtype=np.float32).reshape(-1, 1)
t_tensor = torch.tensor(t, dtype=torch.float32, device=device).repeat(BATCH_SIZE, 1, 1)
t_tensor_flat = t_tensor[:, :, 0]
dt_vis = float(t[1, 0] - t[0, 0])

physics_losses = []
boundary_losses = []
total_losses = []

# === Pick one trajectory for visualization ===
q_min = torch.tensor(np.min(q_data, axis=0), dtype=torch.float32, device=device)
q_max = torch.tensor(np.max(q_data, axis=0), dtype=torch.float32, device=device)

from random import randint
fixed_idx = randint(0, len(q_data) - 200)
#q0_vis = torch.tensor(q_data[fixed_idx], dtype=torch.float32, device=device)
#qT_vis = torch.tensor(q_data[fixed_idx + 199], dtype=torch.float32, device=device)
q0_vis = torch.rand(5, device=device) * (q_max - q_min) + q_min
qT_vis = torch.rand(5, device=device) * (q_max - q_min) + q_min

t_vis = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0)
#q_base_vis = q0_vis + 0.5 * (1 - torch.cos(np.pi * t_vis[0])) * (qT_vis - q0_vis) #cos interpolation
#q_base_vis = q0_vis + t_vis[0] * (qT_vis - q0_vis) #linear interpolation


with torch.no_grad():
    q_pred_before = model(t_vis, q0_vis.unsqueeze(0), qT_vis.unsqueeze(0))

    dq_pred_before = torch.gradient(q_pred_before, spacing=(dt_vis,), dim=1)[0]
    ddq_pred_before = torch.gradient(dq_pred_before, spacing=(dt_vis,), dim=1)[0]
    f_in_before = torch.cat([q_pred_before, dq_pred_before, ddq_pred_before], dim=2)
    f_out_before = f_model(f_in_before.view(-1, 15)).view(1, -1, 5)
    M_in_before = torch.cat([q_pred_before, dq_pred_before], dim=2)
    M_flat_before = mass_model(M_in_before.view(-1, 10)) * y_std + y_mean
    M_before = M_flat_before.view(1, -1, 5, 5)
    tau_pred_before = torch.matmul(M_before, ddq_pred_before.unsqueeze(-1)).squeeze(-1) + f_out_before
    physics_loss_before = torch.mean(tau_pred_before ** 2).item()

q_pred_before = q_pred_before.detach().clone()

# === Training loop ===
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    q0_list, qT_list = [], []
    for _ in range(BATCH_SIZE):
        q0 = torch.rand(5, device=device) * (q_max - q_min) + q_min
        qT = torch.rand(5, device=device) * (q_max - q_min) + q_min

        q0_list.append(q0)
        qT_list.append(qT)

    q0_batch = torch.stack(q0_list)
    qT_batch = torch.stack(qT_list)

    residuals = model(t_tensor, q0_batch, qT_batch)
    q_pred = model(t_tensor, q0_batch, qT_batch)

    dq_pred = torch.gradient(q_pred, spacing=(dt_vis,), dim=1)[0]
    ddq_pred = torch.gradient(dq_pred, spacing=(dt_vis,), dim=1)[0]

    smoothness_loss = torch.mean(ddq_pred**2)
    smoothness_weight = 1e-5  # You can tune this


    f_in = torch.cat([q_pred, dq_pred, ddq_pred], dim=2)
    f_out = f_model(f_in.view(-1, 15)).view(BATCH_SIZE, -1, 5)

    M_in = torch.cat([q_pred, dq_pred], dim=2)
    M_flat = mass_model(M_in.view(-1, 10)) * y_std + y_mean
    M = M_flat.view(BATCH_SIZE, -1, 5, 5)

    tau_pred = torch.matmul(M, ddq_pred.unsqueeze(-1)).squeeze(-1) + f_out

    physics_loss = torch.mean(tau_pred**2) / tau_scale
    boundary_loss = (
        F.mse_loss(q_pred[:, 0], q0_batch) +
        F.mse_loss(q_pred[:, -1], qT_batch)
    ) / q_scale
    residual_magnitude = torch.mean(residuals**2)

    total_loss = physics_loss + boundary_weight * boundary_loss + residual_weight * residual_magnitude + smoothness_weight * smoothness_loss
    total_loss.backward()
    optimizer.step()

    physics_losses.append(physics_loss.item())
    boundary_losses.append(boundary_loss.item())
    total_losses.append(total_loss.item())

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"[{epoch:05d}] Physics: {physics_loss.item():.3e}, Boundary: {boundary_loss.item():.3e}, Residual Mag: {residual_magnitude.item():.3e}, Total: {total_loss.item():.3e}")

torch.save(model.state_dict(), "../models/trajectory_pinn_trained.pt")

# === Post-training visualization ===
with torch.no_grad():
    q_pred_after = model(t_vis, q0_vis.unsqueeze(0), qT_vis.unsqueeze(0))

    dq_pred_after = torch.gradient(q_pred_after, spacing=(dt_vis,), dim=1)[0]
    ddq_pred_after = torch.gradient(dq_pred_after, spacing=(dt_vis,), dim=1)[0]
    f_in_after = torch.cat([q_pred_after, dq_pred_after, ddq_pred_after], dim=2)
    f_out_after = f_model(f_in_after.view(-1, 15)).view(1, -1, 5)
    M_in_after = torch.cat([q_pred_after, dq_pred_after], dim=2)
    M_flat_after = mass_model(M_in_after.view(-1, 10)) * y_std + y_mean
    M_after = M_flat_after.view(1, -1, 5, 5)
    tau_pred_after = torch.matmul(M_after, ddq_pred_after.unsqueeze(-1)).squeeze(-1) + f_out_after
    physics_loss_after = torch.mean(tau_pred_after.detach() ** 2).item()

fig, axes = plt.subplots(1, 5, figsize=(20, 3))
for joint in range(5):
    ax = axes[joint]
    ax.plot(t, q_pred_before[0, :, joint].cpu(), '--', label="Before")
    ax.plot(t, q_pred_after[0, :, joint].cpu(), label="After")
    ax.plot(0, q0_vis[joint].cpu(), 'ro')
    ax.plot(1, qT_vis[joint].cpu(), 'ro')
    ax.set_title(f"Joint {joint}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    ax.legend()

# Add physics losses to the first subplot
axes[0].text(0.02, 0.92, f"Phys. Loss (Before): {physics_loss_before:.1e}", transform=axes[0].transAxes, color='red')
axes[0].text(0.02, 0.82, f"Phys. Loss (After):  {physics_loss_after:.1e}", transform=axes[0].transAxes, color='green')

plt.tight_layout()
plt.savefig("../plots/trajectories_showcase.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(total_losses, label="Total Loss")
plt.plot(physics_losses, label="Physics Loss")
plt.plot(boundary_losses, label="Boundary Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PINN Loss Breakdown (Batch Size 64)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../plots/trajectory_loss_batch64.png")
plt.show()