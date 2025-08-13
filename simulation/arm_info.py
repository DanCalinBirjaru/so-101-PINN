import pybullet as p
import pybullet_data
import time

URDF_PATH = "/arm/SO101/so101_new_calib.urdf"

# === SETUP ===
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)

# Load robot
arm_id = p.loadURDF(URDF_PATH, useFixedBase=True)

# === Print Joint Info ===
print("\n=== SO-101 Joint Info ===")
joint_type_map = {
    0: "REVOLUTE",
    1: "PRISMATIC",
    2: "SPHERICAL",
    3: "PLANAR",
    4: "FIXED"
}

for i in range(p.getNumJoints(arm_id)):
    info = p.getJointInfo(arm_id, i)
    name = info[1].decode("utf-8")
    joint_type = joint_type_map.get(info[2], str(info[2]))
    lower_limit = info[8]
    upper_limit = info[9]
    
    print(f"[{i}] Name: {name:20s} | Type: {joint_type:9s} | Limits: [{lower_limit:6.2f}, {upper_limit:6.2f}]")

print("\n🕵️ Review the printed joint info above and send it here so I can fix the torque control logic.")

# Keep GUI open
while p.isConnected():
    time.sleep(0.1)

p.disconnect()
