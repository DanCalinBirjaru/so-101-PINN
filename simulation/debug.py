import pybullet as p
import pybullet_data
import numpy as np
import time

# === CONFIG ===
URDF_PATH = "/arm/SO101/so101_new_calib.urdf"
JOINT_TO_MOVE = 0         # choose 0–5 manually
TORQUE = 1
FREQ = 0.5
DURATION = 5.0
DT = 1.0 / 240.0

# === SETUP ===
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(DT)
p.resetDebugVisualizerCamera(0.7, 45, -30, [0, 0, 0.3])

# === LOAD ROBOT ===
arm = p.loadURDF(URDF_PATH, useFixedBase=True)
joint_indices = [i for i in range(p.getNumJoints(arm)) if p.getJointInfo(arm, i)[2] != p.JOINT_FIXED]
n_joints = len(joint_indices)

# === DISABLE ALL DEFAULT MOTORS ===
for j in joint_indices:
    p.setJointMotorControl2(bodyIndex=arm, jointIndex=j, controlMode=p.VELOCITY_CONTROL, force=0)

# === RESET STATE ===
for j in joint_indices:
    p.resetJointState(arm, j, targetValue=0.0, targetVelocity=0.0)

# === APPLY TORQUE TO A SINGLE JOINT ===
print(f"Applying sinusoidal torque to joint {JOINT_TO_MOVE} (joint index {joint_indices[JOINT_TO_MOVE]})")

steps = int(DURATION / DT)
for t in range(steps):
    tau = TORQUE * np.sin(2 * np.pi * FREQ * t * DT)
    for i, j in enumerate(joint_indices):
        applied_tau = tau if i == JOINT_TO_MOVE else 0.0
        p.setJointMotorControl2(
            bodyIndex=arm,
            jointIndex=j,
            controlMode=p.TORQUE_CONTROL,
            force=applied_tau
        )
    p.stepSimulation()
    time.sleep(DT)

p.disconnect()
