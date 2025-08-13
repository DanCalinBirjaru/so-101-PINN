servo_names = ['shoulder_pan',
                'shoulder_lift',
                'elbow_flex',
                'wrist_flex',
                'wrist_roll',
                'gripper']

# hard coded from gathered data
position_per_radians = [106.46919400788714, 
                         112.74139521757158, 
                         155.44088739140835, 
                         180.78292741458384, 
                         113.80540519772887, 
                         128.32544715654694]

def rads_to_positions(_servo_name, rads):
    servo_index = servo_names.index(_servo_name)
    rad_per_position = position_per_radians[servo_index]

    position = rads * rad_per_position

    return position