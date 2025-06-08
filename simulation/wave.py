import pybullet as p
import time
import numpy as np

# control
_STEPS = 1000
_INFO = False

# connect to simulation
p.connect(p.GUI)

p.resetDebugVisualizerCamera(
    cameraDistance=0.7,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.3]
)

p.resetSimulation()
p.setGravity(0, 0, -9.81)

# load the robot arm
arm_id = p.loadURDF("/arm/SO101/so101_new_calib.urdf", useFixedBase=True)
num_joints = p.getNumJoints(arm_id)

# get movable joint indices
joint_indices = [i for i in range(num_joints)
                 if p.getJointInfo(arm_id, i)[2] != p.JOINT_FIXED]

# upright pose: hand lifted near head, palm out
upright_pose = np.array([1.5, -1.2, 0.4, -1.5, 1.2, 0.0])[:len(joint_indices)]

# move to upright pose
for t in range(_STEPS):
    alpha = t / _STEPS
    current_joint_angles = alpha * upright_pose

    p.setJointMotorControlArray(
        bodyUniqueId=arm_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=current_joint_angles.tolist(),
        forces=[50] * len(joint_indices),
        positionGains=[0.05] * len(joint_indices)
    )

    p.stepSimulation()
    time.sleep(1. / 240.)

# waving: oscillate elbow (joint 3) and flick wrist (last joint)
elbow_joint = joint_indices[3]
wrist_joint = joint_indices[-1]

elbow_amplitude = 0.5  # side-to-side wave
wrist_amplitude = 0.3  # flick
wave_speed = 2
wave_cycles = 4
wave_steps = 240  # steps per cycle

for t in range(wave_cycles * wave_steps):
    angle_offset = np.sin(2 * np.pi * wave_speed * t / wave_steps)

    target_angles = upright_pose.copy()
    target_angles[3] += elbow_amplitude * angle_offset
    target_angles[-1] += wrist_amplitude * angle_offset * 0.5

    p.setJointMotorControlArray(
        bodyUniqueId=arm_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=target_angles.tolist(),
        forces=[50] * len(joint_indices),
        positionGains=[0.05] * len(joint_indices)
    )

    p.stepSimulation()
    time.sleep(1. / 240.)

# disconnect
p.disconnect()
