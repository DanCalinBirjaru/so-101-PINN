from robot import robot
from path_planner import path_planner
import time


robot_port = '/dev/ttyACM0'
robot_id = 'SO101'
model_path = '../models/trajectory_pinn_trained.pt'


# ================================================

def main():
    r = robot(port = robot_port, id = robot_id)
    r.default_pose()
    #r.max_range_demo()
   
    p = path_planner(r, model_path)
    
    sp = [0, 0, 0, 0, 0]
    tp = [-2, -0.5, -0.2, 0.2, 0.2]
    p.generate_path(current_pos = sp, target_pos = tp, time = 10)
    p.play_path()
    time.sleep(2)
    r.default_pose()
    time.sleep(1)
    r.interface.disconnect()

if __name__ == "__main__":
    main()