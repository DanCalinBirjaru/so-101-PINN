import sympy as sp
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, LagrangesMethod

print('Lagrangian')
# Generalized coordinates (joint angles) as functions of time
q1, q2, q3, q4, q5 = dynamicsymbols('q1 q2 q3 q4 q5')

# Define symbols for link lengths, masses, inertias, and gravity
L1, L2, L3, L4, L5 = sp.symbols('L1 L2 L3 L4 L5', positive=True)
m1, m2, m3, m4, m5 = sp.symbols('m1 m2 m3 m4 m5', positive=True)
I1, I2, I3, I4, I5 = sp.symbols('I1 I2 I3 I4 I5', positive=True)
g = sp.symbols('g', positive=True)  # gravitational acceleration

# Define reference frames for each link
N = ReferenceFrame('N')                    # Inertial/base frame
A = N.orientnew('A', 'Axis', (q1, N.z))    # Base yaw rotation by q1 about z
B = A.orientnew('B', 'Axis', (q2, A.y))    # Shoulder pitch by q2 about A.y
C = B.orientnew('C', 'Axis', (q3, B.y))    # Elbow pitch by q3 about B.y
D = C.orientnew('D', 'Axis', (q4, C.x))    # Wrist roll by q4 about C.x
E = D.orientnew('E', 'Axis', (q5, D.y))    # Wrist pitch by q5 about D.y

# Define key points: joint locations and center-of-mass locations
O = Point('O')               # origin (base)
O.set_vel(N, 0)              # base is fixed
P1 = O.locatenew('P1', L1 * A.z)    # position of shoulder joint (link1 along A.z)
P2 = P1.locatenew('P2', L2 * B.x)   # elbow joint (link2 along B.x)
P3 = P2.locatenew('P3', L3 * C.x)   # wrist roll joint (link3 along C.x)
P4 = P3.locatenew('P4', L4 * D.x)   # wrist pitch joint (link4 along D.x)
P5 = P4.locatenew('P5', L5 * E.x)   # end-effector (link5 along E.x)
# Center of mass for each link (assume at midpoint of each link)
C1 = O.locatenew('C1', L1/2 * A.z)
C2 = P1.locatenew('C2', L2/2 * B.x)
C3 = P2.locatenew('C3', L3/2 * C.x)
C4 = P3.locatenew('C4', L4/2 * D.x)
C5 = P4.locatenew('C5', L5/2 * E.x)

# Linear velocities of each link's CoM (in inertial frame N)
v_C1 = C1.vel(N)
v_C2 = C2.vel(N)
v_C3 = C3.vel(N)
v_C4 = C4.vel(N)
v_C5 = C5.vel(N)

# Angular velocities of each link (in inertial frame)
omega1 = A.ang_vel_in(N)
omega2 = B.ang_vel_in(N)
omega3 = C.ang_vel_in(N)
omega4 = D.ang_vel_in(N)
omega5 = E.ang_vel_in(N)

# Define inertia dyadic for each link (treat each as a sphere or cylinder for simplicity)
I1_dyadic = inertia(A, I1, I1, I1)
I2_dyadic = inertia(B, I2, I2, I2)
I3_dyadic = inertia(C, I3, I3, I3)
I4_dyadic = inertia(D, I4, I4, I4)
I5_dyadic = inertia(E, I5, I5, I5)

# Kinetic energy (translational + rotational)
T_trans = sp.Rational(1, 2) * (m1 * v_C1.dot(v_C1) + m2 * v_C2.dot(v_C2) + 
                               m3 * v_C3.dot(v_C3) + m4 * v_C4.dot(v_C4) + 
                               m5 * v_C5.dot(v_C5))
T_rot  = sp.Rational(1, 2) * (omega1.dot(I1_dyadic.dot(omega1)) + 
                              omega2.dot(I2_dyadic.dot(omega2)) + 
                              omega3.dot(I3_dyadic.dot(omega3)) + 
                              omega4.dot(I4_dyadic.dot(omega4)) + 
                              omega5.dot(I5_dyadic.dot(omega5)))
T = sp.simplify(T_trans + T_rot)

# Potential energy (gravity, with N.z as vertical axis)
h1 = C1.pos_from(O).dot(N.z)  # vertical height of link1 CoM
h2 = C2.pos_from(O).dot(N.z)
h3 = C3.pos_from(O).dot(N.z)
h4 = C4.pos_from(O).dot(N.z)
h5 = C5.pos_from(O).dot(N.z)
V = m1*g*h1 + m2*g*h2 + m3*g*h3 + m4*g*h4 + m5*g*h5

# Lagrangian L = T - V
L = T - V

# Formulate Lagrange’s equations
coords = [q1, q2, q3, q4, q5]
lagrangian_method = LagrangesMethod(L, coords)
lagrangian_method.form_lagranges_equations()  # derive equations of motion
M_matrix = lagrangian_method.mass_matrix      # 5x5 mass matrix M(q)
forcing_vec = lagrangian_method.forcing       # 5x1 forcing vector (includes gravity/Coriolis effects)

# Create numerical functions for M(q) and f(q, qdot) using lambdify
M_func = sp.lambdify((q1, q2, q3, q4, q5, 
                      m1, m2, m3, m4, m5, 
                      L1, L2, L3, L4, L5, 
                      I1, I2, I3, I4, I5, g), 
                     M_matrix, 'numpy')
f_func = sp.lambdify((q1, q2, q3, q4, q5, 
                      q1.diff(), q2.diff(), q3.diff(), q4.diff(), q5.diff(),
                      m1, m2, m3, m4, m5, 
                      L1, L2, L3, L4, L5, 
                      I1, I2, I3, I4, I5, g), 
                     forcing_vec, 'numpy')

# Example parameter values (dummy placeholders; replace with actual arm parameters)
param_values = {
    m1: 2.0, m2: 2.0, m3: 1.0, m4: 1.0, m5: 0.5,        # link masses (kg)
    L1: 0.5, L2: 0.5, L3: 0.5, L4: 0.3, L5: 0.2,        # link lengths (m)
    I1: 0.02, I2: 0.02, I3: 0.01, I4: 0.005, I5: 0.002, # link inertias about CoM (kg·m^2)
    g: 9.81                                            # gravity (m/s^2)
}
# For convenience, pack parameters into a tuple in the order expected by M_func and f_func
params = tuple(param_values[sym] for sym in (m1, m2, m3, m4, m5, 
                                             L1, L2, L3, L4, L5, 
                                             I1, I2, I3, I4, I5, g))

print('calculations')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Data ===
data = np.load("/home/dan/Code/so-101-PINN/data/so101_dynamics.npz")
q = data["q"]
dq = data["dq"]
ddq = data["ddq"]
tau_true = data["tau"][:, :5]

# === Load TauNet (learned f) ===
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

# === Load MassMatrixNet ===
class MassMatrixNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 25)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tau_model = TauNet().to(device)
tau_model.load_state_dict(torch.load("/home/dan/Code/so-101-PINN/models/tau_net.pt", map_location=device))
tau_model.eval()

M_model = MassMatrixNet().to(device)
M_model.load_state_dict(torch.load("/home/dan/Code/so-101-PINN/models/so101_mass_matrix_net_best.pt", map_location=device))
M_model.eval()

# === Symbolic Tau and M ===
def symbolic_tau(qi, dqi, ddqi):
    Mi = np.array(M_func(*qi[:5], *params))                # (5,5)
    fi = np.array(f_func(*qi[:5], *dqi[:5], *params)).flatten()  # (5,)
    return Mi @ ddqi[:5] + fi

def symbolic_M(qi):
    return np.array(M_func(*qi[:5], *params))  # (5,5)

# === Estimate M from network
def learned_M(qi, dqi):
    inp = torch.tensor(np.hstack([qi[:5], dqi[:5]]), dtype=torch.float32).unsqueeze(0).to(device)
    M_flat = M_model(inp).cpu().numpy().flatten()
    return M_flat.reshape(5, 5)

# === Evaluate
N = 200
idx = np.random.choice(len(q), N, replace=False)

tau_symb, tau_learned = [], []
M_symb, M_learned = [], []

with torch.no_grad():
    for i in idx:
        qi, dqi, ddqi = q[i], dq[i], ddq[i]

        # Tau
        tau_symb.append(symbolic_tau(qi, dqi, ddqi))
        inp_tau = torch.tensor(np.hstack([qi[:5], dqi[:5], ddqi[:5]]), dtype=torch.float32).to(device)
        tau_learned.append(tau_model(inp_tau).cpu().numpy())

        # M
        M_symb.append(symbolic_M(qi))
        M_learned.append(learned_M(qi, dqi))

tau_symb = np.array(tau_symb)
tau_learned = np.array(tau_learned)
tau_true_subset = tau_true[idx]

M_symb = np.array(M_symb)
M_learned = np.array(M_learned)

# === Errors
mae_symb_tau = np.mean(np.abs(tau_symb - tau_true_subset), axis=0)
mae_learned_tau = np.mean(np.abs(tau_learned - tau_true_subset), axis=0)
mae_M = np.mean(np.abs(M_symb - M_learned), axis=0)

# === Plot τ error
joint_labels = [f"Joint {i}" for i in range(5)]
x = np.arange(5)
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, mae_symb_tau, width, label="Symbolic τ")
plt.bar(x + width/2, mae_learned_tau, width, label="Learned τ")
plt.xticks(x, joint_labels)
plt.ylabel("Mean Absolute Error [Nm]")
plt.title("Torque Prediction Error: Symbolic vs Learned")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("tau_error_comparison.png")
plt.show()

# === Plot M error
plt.figure(figsize=(6, 5))
sns.heatmap(mae_M, annot=True, fmt=".2f", cmap="viridis", xticklabels=joint_labels, yticklabels=joint_labels)
plt.title("Mean Absolute Error in Estimated Mass Matrix M(q)")
plt.xlabel("Column")
plt.ylabel("Row")
plt.tight_layout()
plt.savefig("mass_matrix_error_heatmap.png")
plt.show()
