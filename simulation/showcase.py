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

# you define these — example: 6 joints
start_joint_angles = np.array([0, 0, 0, 0, 180, 0])          # all joints at 0
end_joint_angles = np.array([-0.5, -0.4, 0.4, -1.2, 1.4, 0.5])[:len(joint_indices)]  # customize this!

# check if user provided fewer angles than joints
if len(end_joint_angles) != len(joint_indices):
    raise ValueError(f"Expected {len(joint_indices)} joint angles, got {len(end_joint_angles)}.")

# get movable joint indices
joint_indices = [i for i in range(num_joints)
                 if p.getJointInfo(arm_id, i)[2] != p.JOINT_FIXED]

# simulate interpolation from start to end joint angles
for t in range(_STEPS):
    alpha = t / _STEPS
    current_joint_angles = (1 - alpha) * start_joint_angles + alpha * end_joint_angles

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

# disconnect
p.disconnect()
