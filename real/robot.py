from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from utils import rads_to_positions, servo_names

import time

class robot:
    def __init__(self, port, id):
        self.port = port
        self.id = id

        self.config = SO101FollowerConfig(
            port = self.port,
            id = self.id
            )
        
        self.interface = SO101Follower(self.config)
        self.interface.connect()

    def default_pose(self):
        action = {'shoulder_pan.pos' : 0,
                        'shoulder_lift.pos': -100,
                        'elbow_flex.pos' : 100,
                        'wrist_flex.pos' : 40,
                        'wrist_roll.pos' : 0,
                        'gripper.pos' : 0}
        
        self.interface.send_action(action)

    def mid_pose(self):
        action = {'shoulder_pan.pos' : 0,
                        'shoulder_lift.pos': 0,
                        'elbow_flex.pos' : 0,
                        'wrist_flex.pos' : 0,
                        'wrist_roll.pos' : 0,
                        'gripper.pos' : 0}
        
        self.interface.send_action(action)

    def move_joints(self, servos, positions, radians = False):
        # if it is not a list, make it a list
        if type(servos) != list:
            servos = [servos]

        if type(positions) != list:
            positions = [positions]

        # if user provides radians, we convert to positions
        if radians == True:
            for i in range(len(positions)):
                positions[i] = rads_to_positions(servos[i], positions[i])

        action = {'shoulder_pan.pos' : 0,
                        'shoulder_lift.pos': 0,
                        'elbow_flex.pos' : 0,
                        'wrist_flex.pos' : 0,
                        'wrist_roll.pos' : 0,
                        'gripper.pos' : 0}
        
        for i, name in enumerate(servos):
            action[name + '.pos'] = positions[i]
            self.interface.send_action(action)

    def range_max(self, servo):
        self.move_joints(servo, 100)

    def range_min(self, servo):
        self.move_joints(servo, -100)

    def max_range_demo(self):
        for name in servo_names:
            self.mid_pose()
            time.sleep(2)

            self.range_max(name)
            time.sleep(2)

            self.range_min(name)
            time.sleep(2)
        
        self.default_pose()    