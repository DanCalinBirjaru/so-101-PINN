import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# === Load dataset ===
data = np.load("../data/so101_dynamics.npz")
q_data = data["q"][:, :5]  # Only first 5 joints
dq_data = data["dq"][:, :5]
ddq_data = data["ddq"][:, :5]

# === f model (nonlinear dynamics) ===
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
f_model.load_state_dict(torch.load("../models/tau_net.pt", map_location=device))
f_model.eval()

# === M model (mass matrix) ===
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
mass_model.load_state_dict(torch.load("../models/so101_mass_matrix_net_best.pt", map_location=device))
mass_model.eval()

# Load normalization stats for mass matrix
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

# Load trained PINN model
pinn_model = TrajectoryPINN().to(device)
pinn_model.load_state_dict(torch.load("../models/trajectory_pinn_trained.pt", map_location=device))
pinn_model.eval()

# === Physics Loss Computation ===
def compute_physics_loss(q_pred, dt=0.005):
    """
    Compute physics loss: ||τ||² where τ = M(q)q̈ + f(q,q̇,q̈)
    """
    q_tensor = torch.tensor(q_pred, dtype=torch.float32, device=device)
    
    # Compute derivatives using finite differences
    dq_pred = torch.gradient(q_tensor, spacing=(dt,), dim=0)[0]
    ddq_pred = torch.gradient(dq_pred, spacing=(dt,), dim=0)[0]
    
    # Prepare inputs for f and M models
    f_in = torch.cat([q_tensor, dq_pred, ddq_pred], dim=1)
    f_out = f_model(f_in)
    
    M_in = torch.cat([q_tensor, dq_pred], dim=1)
    M_flat = mass_model(M_in) * y_std + y_mean
    M = M_flat.view(-1, 5, 5)
    
    # Compute τ = M(q)q̈ + f(q,q̇,q̈)
    tau_pred = torch.matmul(M, ddq_pred.unsqueeze(-1)).squeeze(-1) + f_out
    
    # Physics loss is the magnitude of required external torques
    physics_loss = torch.mean(tau_pred ** 2).item()
    
    return physics_loss

# === Trajectory Generation Methods ===

def generate_linear_interpolation(q0, qT, T=200):
    """Linear interpolation between start and end configurations"""
    t = np.linspace(0, 1, T)
    q_traj = np.outer((1 - t), q0) + np.outer(t, qT)
    return q_traj

def generate_cosine_interpolation(q0, qT, T=200):
    """Cosine interpolation for smoother trajectories"""
    t = np.linspace(0, 1, T)
    # Use cosine interpolation: q(t) = q0 + 0.5 * (1 - cos(π*t)) * (qT - q0)
    interpolation_factor = 0.5 * (1 - np.cos(np.pi * t))
    q_traj = np.outer((1 - interpolation_factor), q0) + np.outer(interpolation_factor, qT)
    return q_traj

def generate_pid_trajectory(q0, qT, T=200, dt=0.005, kp=100.0, kd=10.0):
    """
    Generate trajectory using PID controller simulation
    Simple second-order system: q̈ = kp*(target - q) - kd*q̇
    """
    t_span = np.linspace(0, (T-1)*dt, T)
    
    def pid_dynamics(state, t):
        """
        state = [q1, q2, q3, q4, q5, dq1, dq2, dq3, dq4, dq5]
        """
        q = state[:5]
        dq = state[5:]
        
        # Smooth target transition (sigmoid-like)
        progress = min(1.0, t / (t_span[-1] * 0.8))  # Reach target by 80% of time
        target = q0 + progress * (qT - q0)
        
        # PID control: ddq = kp*(target - q) - kd*dq
        ddq = kp * (target - q) - kd * dq
        
        return np.concatenate([dq, ddq])
    
    # Initial state: [q0, 0_velocities]
    initial_state = np.concatenate([q0, np.zeros(5)])
    
    # Integrate
    try:
        solution = odeint(pid_dynamics, initial_state, t_span)
        q_traj = solution[:, :5]  # Extract positions
        return q_traj
    except:
        # Fallback to cosine interpolation if PID integration fails
        print("PID integration failed, using cosine interpolation as fallback")
        return generate_cosine_interpolation(q0, qT, T)

def generate_pinn_trajectory(q0, qT, T=200):
    """Generate trajectory using trained PINN model"""
    t = np.linspace(0, 1, T).reshape(-1, 1)
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0)
    q0_tensor = torch.tensor(q0, dtype=torch.float32, device=device).unsqueeze(0)
    qT_tensor = torch.tensor(qT, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        q_pred = pinn_model(t_tensor, q0_tensor, qT_tensor)
        return q_pred[0].cpu().numpy()

# === Benchmarking ===
print("Starting trajectory benchmarking...")

# Sample configurations from data
q_min = np.min(q_data, axis=0)
q_max = np.max(q_data, axis=0)

N_TRAJECTORIES = 10000
T_POINTS = 200

physics_losses = {
    'PINN': [],
    'Linear': [],
    'Cosine': [],
    'PID': []
}

# Progress tracking
for i in range(N_TRAJECTORIES):
    if i % 100 == 0:
        print(f"Processing trajectory {i}/{N_TRAJECTORIES}")
    
    # Sample random start and end configurations
    q0 = np.random.uniform(q_min, q_max)
    qT = np.random.uniform(q_min, q_max)
    
    try:
        # Generate trajectories using different methods
        traj_linear = generate_linear_interpolation(q0, qT, T_POINTS)
        traj_cosine = generate_cosine_interpolation(q0, qT, T_POINTS)
        traj_pid = generate_pid_trajectory(q0, qT, T_POINTS)
        traj_pinn = generate_pinn_trajectory(q0, qT, T_POINTS)
        
        # Compute physics losses
        physics_losses['Linear'].append(compute_physics_loss(traj_linear))
        physics_losses['Cosine'].append(compute_physics_loss(traj_cosine))
        physics_losses['PID'].append(compute_physics_loss(traj_pid))
        physics_losses['PINN'].append(compute_physics_loss(traj_pinn))
        
    except Exception as e:
        print(f"Error processing trajectory {i}: {e}")
        continue

# Convert to numpy arrays for easier handling
for key in physics_losses:
    physics_losses[key] = np.array(physics_losses[key])

# === Results and Statistics ===
print("\n" + "="*60)
print("TRAJECTORY BENCHMARKING RESULTS")
print("="*60)

for method in ['PINN', 'Linear', 'Cosine', 'PID']:
    losses = physics_losses[method]
    if len(losses) > 0:
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        median_loss = np.median(losses)
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        
        print(f"\n{method} Method:")
        print(f"  Physics Loss = {mean_loss:.3e} ± {std_loss:.3e}")
        print(f"  Median = {median_loss:.3e}")
        print(f"  Min = {min_loss:.3e}, Max = {max_loss:.3e}")
        print(f"  Samples = {len(losses)}")

# === Plotting ===

# Set style
plt.style.use('default')
sns.set_palette("husl")

# 1. Individual distribution plots for each method
methods_to_plot = ['PINN', 'PID', 'Cosine', 'Linear']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, method in enumerate(methods_to_plot):
    if len(physics_losses[method]) > 0:
        ax = axes[i]
        losses = physics_losses[method]
        
        # Calculate statistics
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        median_loss = np.median(losses)
        
        # Create histogram
        n, bins, patches = ax.hist(losses, bins=40, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for mean and median
        ax.axvline(mean_loss, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_loss:.3e}')
        ax.axvline(median_loss, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_loss:.3e}')
        
        # Add statistics text box
        stats_text = f'Mean = {mean_loss:.3e}\nStd = {std_loss:.3e}\nMedian = {median_loss:.3e}\nSamples = {len(losses)}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, weight='bold')
        
        ax.set_xlabel('Physics Loss')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{method} Method - Physics Loss Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set appropriate scale based on data range
        if np.max(losses) / np.min(losses) > 100:
            ax.set_yscale('log')

plt.tight_layout()
plt.savefig("../plots/trajectory_benchmark_individual_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Combined comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Box plot with statistics
data_for_box = []
labels_for_box = []
stats_for_box = []

for method in ['PINN', 'PID', 'Cosine', 'Linear']:
    if len(physics_losses[method]) > 0:
        data_for_box.append(physics_losses[method])
        labels_for_box.append(method)
        mean_val = np.mean(physics_losses[method])
        std_val = np.std(physics_losses[method])
        stats_for_box.append(f'{mean_val:.2e}±{std_val:.2e}')

box_plot = ax1.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
box_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
for patch, color in zip(box_plot['boxes'], box_colors[:len(box_plot['boxes'])]):
    patch.set_facecolor(color)

# Add statistics annotations below box plots
for i, (label, stats) in enumerate(zip(labels_for_box, stats_for_box)):
    ax1.text(i+1, ax1.get_ylim()[0], stats, ha='center', va='top', 
             fontsize=9, weight='bold', rotation=45)

ax1.set_ylabel('Physics Loss')
ax1.set_title('Physics Loss Distribution Comparison')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Combined histogram with statistics
for i, method in enumerate(['PINN', 'PID', 'Cosine', 'Linear']):
    if len(physics_losses[method]) > 0:
        losses = physics_losses[method]
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        ax2.hist(losses, bins=30, alpha=0.6, label=f'{method}\n({mean_loss:.2e}±{std_loss:.2e})', 
                color=colors[i])

ax2.set_xlabel('Physics Loss')
ax2.set_ylabel('Frequency')
ax2.set_title('Physics Loss Histograms with Statistics')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../plots/trajectory_benchmark_distributions_combined.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Statistical summary bar chart
fig, ax = plt.subplots(figsize=(10, 6))

methods = []
means = []
stds = []

for method in ['PINN', 'PID', 'Cosine', 'Linear']:
    if len(physics_losses[method]) > 0:
        methods.append(method)
        means.append(np.mean(physics_losses[method]))
        stds.append(np.std(physics_losses[method]))

x_pos = np.arange(len(methods))
bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8)

# Color bars
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_xlabel('Method')
ax.set_ylabel('Physics Loss')
ax.set_title('Mean Physics Loss by Trajectory Generation Method\n(Error bars show ±1 standard deviation)')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Add value labels on bars
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std, f'{mean:.2e}\n±{std:.2e}', 
            ha='center', va='bottom', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig("../plots/trajectory_benchmark_summary.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Detailed statistical comparison table
print("\n" + "="*80)
print("DETAILED STATISTICAL COMPARISON")
print("="*80)
print(f"{'Method':<10} {'Mean':<12} {'Std':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
print("-" * 80)

for method in ['PINN', 'PID', 'Cosine', 'Linear']:
    if len(physics_losses[method]) > 0:
        losses = physics_losses[method]
        print(f"{method:<10} {np.mean(losses):<12.3e} {np.std(losses):<12.3e} "
              f"{np.median(losses):<12.3e} {np.min(losses):<12.3e} {np.max(losses):<12.3e}")

# 4. Performance improvement analysis
if len(physics_losses['PINN']) > 0:
    print(f"\nPERFORMACE IMPROVEMENTS (vs PINN):")
    print("-" * 40)
    pinn_mean = np.mean(physics_losses['PINN'])
    
    for method in ['Linear', 'Cosine', 'PID']:
        if len(physics_losses[method]) > 0:
            method_mean = np.mean(physics_losses[method])
            improvement = (method_mean - pinn_mean) / pinn_mean * 100
            print(f"{method}: {improvement:+.1f}% (lower is better)")

print(f"\nBenchmarking complete! Results saved to ../plots/")
print(f"Total trajectories processed: {len(physics_losses['PINN'])}")