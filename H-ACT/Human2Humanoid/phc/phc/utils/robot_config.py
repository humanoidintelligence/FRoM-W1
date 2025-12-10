import torch

class RobotConfig:
    
    class Extend:
        extend = True
        extend_link_name = ["left_hand_link", "right_hand_link", "head_link"]
        extend_parent_idx = [19, 33, 0]
        extend_local_translation = [[0.2, 0, 0], [0.2, 0, 0], [0, 0, 0.45]]
        extend_local_rotation = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
        
    ROBOT_ROTATION_AXIS = torch.tensor([
        [0, 1, 0], # l_hip_paw
        [1, 0, 0], # l_hip_roll
        [0, 0, 1], # l_hip_yitch
        
        [0, 1, 0], # l_knee
        [0, 1, 0], # l_ankle_pitch
        [1, 0, 0], # l_ankle_roll
        
        [0, 1, 0], # r_hip_paw
        [1, 0, 0], # r_hip_roll
        [0, 0, 1], # r_hip_yitch
        
        [0, 1, 0], # r_knee
        [0, 1, 0], # r_ankle_pitch
        [1, 0, 0], # r_ankle_roll
        
        [0, 0, 1], # waist
        [1, 0, 0], # waist_roll
        [0, 1, 0], # waist_pitch
        
        [0, 1, 0], # l_shoulder_yaw
        
        [0, 1, 0], # l_shoulder_pitch
        [1, 0, 0], # l_roll_pitch
        [0, 0, 1], # l_yaw_pitch
        
        [0, 1, 0], # l_elbow
        
        [0, 1, 0], # r_shoulder_pitch
        [1, 0, 0], # r_roll_pitch
        [0, 0, 1], # r_yaw_pitch
        
        [0, 1, 0], # r_elbow
    ])
    xml_file = "/home/fudan/Workspace/h2h-G1-GR1-H1/legged_gym/resources/robots/g1/xml/g1_23dof_waist.xml"
    
    JOINT_NUM = len(ROBOT_ROTATION_AXIS)
    
# robotConfig = RobotConfig()