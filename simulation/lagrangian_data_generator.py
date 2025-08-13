import pybullet as p
import pybullet_data
import time
import numpy as np
from tqdm import tqdm

# === CONFIG ===
STEPS = 10000
SAVE_EVERY = 10
RESET_EVERY = 2000
DT = 1.0 / 240.0
MAX_TORQUE = 100  # Additional torques on top of motor control
TORQUE_HOLD_STEPS = 3000
POSITION_HOLD_STEPS = 100  # How long to hold each target position
URDF_PATH = "../arm/SO101/so101_new_calib.urdf"
OUTPUT_FILE = "../data/so101_dynamics.npz"

# === CONNECT & SETUP ===
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=0.7, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.3])

# Set realistic physics parameters
p.setPhysicsEngineParameter(
    fixedTimeStep=DT,
    numSolverIterations=50,
    numSubSteps=4,
    contactBreakingThreshold=0.001,
    erp=0.2,
    contactERP=0.2,
    frictionERP=0.2
)

# === LOAD ROBOT ===
arm_id = p.loadURDF(URDF_PATH, useFixedBase=True)

# Get all movable joints
joint_indices = []
joint_names = []
for i in range(p.getNumJoints(arm_id)):
    joint_info = p.getJointInfo(arm_id, i)
    if joint_info[2] != p.JOINT_FIXED:
        joint_indices.append(i)
        joint_names.append(joint_info[1].decode("utf-8"))

n_joints = len(joint_indices)
print(f"Found {n_joints} movable joints: {joint_names}")

# === JOINT LIMITS ===
joint_info = [p.getJointInfo(arm_id, j) for j in joint_indices]
lower_limits = np.array([info[8] for info in joint_info])
upper_limits = np.array([info[9] for info in joint_info])

print(f"Lower limits (deg): {np.degrees(lower_limits)}")
print(f"Upper limits (deg): {np.degrees(upper_limits)}")

# === STARTING POSITION ===
start_angles = np.zeros(n_joints)
for i, j in enumerate(joint_indices):
    p.resetJointState(arm_id, j, start_angles[i], 0.0)

# === MOTOR CONTROL PARAMETERS ===
# Keep motors active with PID control
position_gains = np.ones(n_joints) * 100.0  # P gain
velocity_gains = np.ones(n_joints) * 10.0   # D gain
max_forces = np.ones(n_joints) * 50.0       # Maximum motor force

# Scale gains for different joint sizes
gain_scales = np.array([1.0, 1.0, 0.8, 0.6, 0.4, 0.2])[:n_joints]
position_gains *= gain_scales
velocity_gains *= gain_scales
max_forces *= gain_scales

print(f"Position gains: {position_gains}")
print(f"Velocity gains: {velocity_gains}")
print(f"Max forces: {max_forces}")

# === BUFFERS ===
q_list, dq_list, ddq_list, tau_list, target_list = [], [], [], [], []
prev_dq = np.zeros(n_joints)
current_tau = np.zeros(n_joints)
current_targets = start_angles.copy()

# === SIMULATION LOOP ===
for step in tqdm(range(STEPS), desc="Collecting SO-101 data with active motors"):
    # Reset periodically
    if step % RESET_EVERY == 0:
        #print(f"\nResetting at step {step}")
        for i, j in enumerate(joint_indices):
            p.resetJointState(arm_id, j, start_angles[i], 0.0)
        prev_dq = np.zeros(n_joints)
        current_targets = start_angles.copy()
    
    # Change target positions periodically
    if step % POSITION_HOLD_STEPS == 0:
        # Generate new random target positions within joint limits
        target_range = 0.3  # Limit range to 30% of joint limits for safety
        mid_points = (upper_limits + lower_limits) / 2
        ranges = (upper_limits - lower_limits) * target_range
        
        current_targets = mid_points + np.random.uniform(-ranges/2, ranges/2)
        current_targets = np.clip(current_targets, lower_limits, upper_limits)
        
        #print(f"Step {step}: New targets (deg) = {np.degrees(current_targets)}")
    
    # Change additional torques periodically (on top of motor control)
    if step % TORQUE_HOLD_STEPS == 0:
        base_torque = np.random.uniform(-MAX_TORQUE, MAX_TORQUE, size=n_joints)
        torque_scales = np.array([1.0, 1.0, 0.8, 0.6, 0.4, 0.2])[:n_joints]
        current_tau = base_torque * torque_scales
        
        #print(f"Step {step}: Additional torques = {current_tau}")
    
    # Apply position control with PID (motors stay active)
    for i, j in enumerate(joint_indices):
        p.setJointMotorControl2(
            arm_id, j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=current_targets[i],
            positionGain=position_gains[i],
            velocityGain=velocity_gains[i],
            force=max_forces[i]
        )
    
    # Apply additional torques on top of motor control
    # Note: This creates interesting dynamics as the motors fight against the disturbances
    for i, j in enumerate(joint_indices):
        # Get current motor torque and add our disturbance
        p.setJointMotorControl2(
            arm_id, j,
            controlMode=p.TORQUE_CONTROL,
            force=current_tau[i]
        )
    
    # Actually, let's use a hybrid approach - position control with additional torque
    for i, j in enumerate(joint_indices):
        p.setJointMotorControl2(
            arm_id, j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=current_targets[i],
            positionGain=position_gains[i],
            velocityGain=velocity_gains[i],
            force=max_forces[i] + abs(current_tau[i])  # Add disturbance to max force
        )
    
    # Step simulation
    p.stepSimulation()
    time.sleep(DT)
    
    # Get joint states
    q = np.array([p.getJointState(arm_id, j)[0] for j in joint_indices])
    dq = np.array([p.getJointState(arm_id, j)[1] for j in joint_indices])
    tau_measured = np.array([p.getJointState(arm_id, j)[3] for j in joint_indices])
    
    # Calculate acceleration
    ddq = (dq - prev_dq) / DT
    prev_dq = dq.copy()
    
    # Save data every SAVE_EVERY steps
    if step % SAVE_EVERY == 0:
        q_list.append(q)
        dq_list.append(dq)
        ddq_list.append(ddq)
        tau_list.append(tau_measured)  # Use measured torque (includes motor response)
        target_list.append(current_targets.copy())
        
        # Debug print
        #if step % (SAVE_EVERY * 20) == 0:
            #print(f"\nStep {step}:")
            #print(f"  Targets (deg): {np.degrees(current_targets)}")
            #print(f"  Actual (deg): {np.degrees(q)}")
            #print(f"  Error (deg): {np.degrees(current_targets - q)}")
            #print(f"  Velocities (deg/s): {np.degrees(dq)}")
            #print(f"  Measured torques: {tau_measured}")

# === SAVE ===
if q_list:
    np.savez(OUTPUT_FILE, 
             q=np.array(q_list), 
             dq=np.array(dq_list), 
             ddq=np.array(ddq_list), 
             tau=np.array(tau_list),
             targets=np.array(target_list))  # Include target positions
    print(f"\n✔ Saved {len(q_list)} samples to {OUTPUT_FILE}")
else:
    print("\n❌ No data collected!")

p.disconnect()

# === ANALYSIS ===
if q_list:
    q_array = np.array(q_list)
    dq_array = np.array(dq_list)
    tau_array = np.array(tau_list)
    target_array = np.array(target_list)
    
    print(f"\n=== DATA ANALYSIS ===")
    print(f"Total samples: {len(q_list)}")
    
    # Position tracking analysis
    position_errors = target_array - q_array
    rms_errors = np.sqrt(np.mean(position_errors**2, axis=0))
    print(f"RMS tracking errors (deg): {np.degrees(rms_errors)}")
    
    # Movement analysis
    position_ranges = np.max(q_array, axis=0) - np.min(q_array, axis=0)
    velocity_ranges = np.max(np.abs(dq_array), axis=0)
    torque_ranges = np.max(np.abs(tau_array), axis=0)
    