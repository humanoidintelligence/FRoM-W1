import torch
from .smpl_parser import SMPL_BONE_ORDER_NAMES

class G1Config:
    class Extend:
        extend = True
        extend_link_name = ["left_hand_site", "right_hand_site", "head_link"]
        extend_parent_idx = [19, 26, 15]
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
        
        [0, 0, 1], # waist_yaw
        [1, 0, 0], # waist_roll
        [0, 1, 0], # waist_pitch
        
        [0, 1, 0], # l_shoulder_pitch
        [1, 0, 0], # l_roll_pitch
        [0, 0, 1], # l_yaw_pitch
        
        [0, 1, 0], # l_elbow

        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        
        [0, 1, 0], # r_shoulder_pitch
        [1, 0, 0], # r_roll_pitch
        [0, 0, 1], # r_yaw_pitch
        
        [0, 1, 0], # r_elbow

        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    xml_file = "assets/robot/g1/g1_29dof.xml"
    
    JOINT_NUM = len(ROBOT_ROTATION_AXIS)
    ROBOT_JOINT_NAMES = [
                        'pelvis', 
                        'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 
                        'left_knee_link', 
                        'left_ankle_pitch_link', 'left_ankle_roll_link',
                        
                        'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 
                        'right_knee_link', 
                        'right_ankle_pitch_link', 'right_ankle_roll_link',
                        
                        'waist_yaw_link', 'waist_roll_link', 'torso_link',
                        
                        'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 
                        'left_elbow_link', 
                        'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
                        
                        'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 
                        'right_elbow_link', 
                        'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link',

                    ]
    
    ROBOT_JOINT_PICK = [
                        'pelvis', 
                        'left_hip_pitch_link', 
                        'left_knee_link', 
                        'left_ankle_pitch_link', 
                        
                        'right_hip_pitch_link', 
                        'right_knee_link', 
                        'right_ankle_pitch_link',
                        
                        # 'torso_link',
                        
                        'left_shoulder_pitch_link', 
                        'left_elbow_link', 
                        # 'left_wrist_yaw_link',
                        
                        'right_shoulder_pitch_link', 
                        'right_elbow_link', 
                        # 'right_wrist_yaw_link',
                        
                        'left_hand_site',
                        'right_hand_site',
                        'head_link',
            ]
    
    SMPL_JOINT_PICK = [
                    "Pelvis", 
                    "L_Hip",  "L_Knee", "L_Ankle",  
                    "R_Hip", "R_Knee", "R_Ankle", 
                    # "Spine", 
                    "L_Shoulder", "L_Elbow", #"L_Hand", 
                    "R_Shoulder", "R_Elbow", #"R_Hand",
                    "L_Hand", "R_Hand",
                    "Head"
                    ]
    
    WRIST_PICK = ['left_wrist_yaw_link', 'right_wrist_yaw_link',]
    SMPL_WRIST_PICK = ['L_Hand', 'R_Hand']
    
    def __init__(self):
        if self.Extend.extend:
            self.ROBOT_JOINT_NAMES = self.ROBOT_JOINT_NAMES + self.Extend.extend_link_name
        self.robot_joint_pick_idx = [self.ROBOT_JOINT_NAMES.index(j) for j in self.ROBOT_JOINT_PICK]
        self.smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in self.SMPL_JOINT_PICK]
        self.wrist_pick_idx = [self.ROBOT_JOINT_NAMES.index(j) for j in self.WRIST_PICK]
        self.smpl_wrist_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in self.SMPL_WRIST_PICK]

class H1Config:
    class Extend:
        extend = True
        extend_link_name = ["left_hand_link", "right_hand_link", "head_link"]
        extend_parent_idx = [15, 19, 1]
        extend_local_translation = [[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.75]]
        extend_local_rotation = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
        
    ROBOT_ROTATION_AXIS = torch.tensor([
        [0, 0, 1], # l_hip_yaw
        [1, 0, 0], # l_hip_roll
        [0, 1, 0], # l_hip_pitch
        
        [0, 1, 0], # kneel
        [0, 1, 0], # ankle
        
        [0, 0, 1], # r_hip_yaw
        [1, 0, 0], # r_hip_roll
        [0, 1, 0], # r_hip_pitch
        
        [0, 1, 0], # kneel
        [0, 1, 0], # ankle
        
        [0, 0, 1], # torso
        
        [0, 1, 0], # l_shoulder_pitch
        [1, 0, 0], # l_roll_pitch
        [0, 0, 1], # l_yaw_pitch
        
        [0, 1, 0], # l_elbow
        
        [0, 1, 0], # r_shoulder_pitch
        [1, 0, 0], # r_roll_pitch
        [0, 0, 1], # r_yaw_pitch
        
        [0, 1, 0], # r_elbow
    ])
    xml_file = "assets/robot/h1/h1.xml"
    
    JOINT_NUM = len(ROBOT_ROTATION_AXIS)
    ROBOT_JOINT_NAMES = [
                        'pelvis', 
                    'left_hip_yaw_link', 'left_hip_roll_link','left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
                    'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
                    'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 
                    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'
                    ]
    
    ROBOT_JOINT_PICK = [
        'pelvis', "left_knee_link", "left_ankle_link",  'right_knee_link', 'right_ankle_link', "left_shoulder_roll_link", "left_elbow_link", "left_hand_link", "right_shoulder_roll_link", "right_elbow_link", "right_hand_link"
    ]
    
    SMPL_JOINT_PICK = [
                    "Pelvis",  "L_Knee", "L_Ankle",  "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"
                    ]
    
    WRIST_PICK = []
    SMPL_WRIST_PICK = []
    
    def __init__(self):
        if self.Extend.extend:
            self.ROBOT_JOINT_NAMES = self.ROBOT_JOINT_NAMES + self.Extend.extend_link_name
        self.robot_joint_pick_idx = [self.ROBOT_JOINT_NAMES.index(j) for j in self.ROBOT_JOINT_PICK]
        self.smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in self.SMPL_JOINT_PICK]
        self.wrist_pick_idx = [self.ROBOT_JOINT_NAMES.index(j) for j in self.WRIST_PICK]
        self.smpl_wrist_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in self.SMPL_WRIST_PICK]

class H121DOFConfig:
    class Extend:
        extend = True
        extend_link_name = ["left_hand_link", "right_hand_link", "head_link"]
        extend_parent_idx = [15, 20, 0]
        extend_local_translation = [[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.75]]
        extend_local_rotation = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
        
    ROBOT_ROTATION_AXIS = torch.tensor([
        [0, 0, 1], # l_hip_yaw
        [1, 0, 0], # l_hip_roll
        [0, 1, 0], # l_hip_pitch
        
        [0, 1, 0], # kneel
        [0, 1, 0], # ankle
        
        [0, 0, 1], # r_hip_yaw
        [1, 0, 0], # r_hip_roll
        [0, 1, 0], # r_hip_pitch
        
        [0, 1, 0], # kneel
        [0, 1, 0], # ankle
        
        [0, 0, 1], # torso
        
        [0, 1, 0], # l_shoulder_pitch
        [1, 0, 0], # l_roll_pitch
        [0, 0, 1], # l_yaw_pitch
        
        [0, 1, 0], # l_elbow
        [1, 0, 0], # l_wrist_roll
        
        [0, 1, 0], # r_shoulder_pitch
        [1, 0, 0], # r_roll_pitch
        [0, 0, 1], # r_yaw_pitch
        
        [0, 1, 0], # r_elbow
        [1, 0, 0], # r_wrist_roll
    ])
    xml_file = "assets/robot/h1/h1_21dof.xml"
    
    JOINT_NUM = len(ROBOT_ROTATION_AXIS)
    ROBOT_JOINT_NAMES = [
                        'pelvis', 
                    'left_hip_yaw_link', 'left_hip_roll_link','left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
                    'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
                    'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_link',
                    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_link',
                    ]
    
    ROBOT_JOINT_PICK = [
        'pelvis', "left_knee_link", "left_ankle_link",  'right_knee_link', 'right_ankle_link', "left_shoulder_roll_link", "left_elbow_link", "left_hand_link", "right_shoulder_roll_link", "right_elbow_link", "right_hand_link"
    ]
    
    SMPL_JOINT_PICK = [
                    "Pelvis",  "L_Knee", "L_Ankle",  "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"
                    ]
    
    WRIST_PICK = ['left_wrist_link', 'right_wrist_link']
    SMPL_WRIST_PICK = ['L_Hand', 'R_Hand']
    
    def __init__(self):
        if self.Extend.extend:
            self.ROBOT_JOINT_NAMES = self.ROBOT_JOINT_NAMES + self.Extend.extend_link_name
        self.robot_joint_pick_idx = [self.ROBOT_JOINT_NAMES.index(j) for j in self.ROBOT_JOINT_PICK]
        self.smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in self.SMPL_JOINT_PICK]
        self.wrist_pick_idx = [self.ROBOT_JOINT_NAMES.index(j) for j in self.WRIST_PICK]
        self.smpl_wrist_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in self.SMPL_WRIST_PICK]

g1Config = G1Config()
h1Config = H1Config()
h121Config = H121DOFConfig()
