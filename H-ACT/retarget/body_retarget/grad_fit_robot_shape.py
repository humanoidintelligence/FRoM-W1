from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_parser import (
    SMPL_Parser,
)

import joblib
import torch
from torch.autograd import Variable
from smpl_parser import SMPL_BONE_ORDER_NAMES
from robot import Humanoid_Batch
from robot_config import robotConfig

robot = Humanoid_Batch(cfg=robotConfig)
joint_nums = robot.cfg.JOINT_NUM
device = torch.device("cpu")

# 1,1,J,4
pose_aa_robot = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], joint_nums, axis=2), 1, axis=1)
pose_aa_robot = torch.from_numpy(pose_aa_robot).float()

dof_pos = torch.zeros((1, joint_nums))
pose_aa_robot = robot.cfg.ROBOT_ROTATION_AXIS * dof_pos[..., None]

root_trans = torch.zeros((1, 1, 3))    

###### prepare SMPL default pause for gr1
pose_aa_stand = np.zeros((1, 72))
rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
pose_aa_stand[:, :3] = rotvec
pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Shoulder')] = sRot.from_euler("xyz", [0, 0, -np.pi/2],  degrees = False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Shoulder')] = sRot.from_euler("xyz", [0, 0, np.pi/2],  degrees = False).as_rotvec()

# only H1 and G1
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Elbow')] = sRot.from_euler("xyz", [0, -np.pi/2, 0],  degrees = False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Elbow')] = sRot.from_euler("xyz", [0, np.pi/2, 0],  degrees = False).as_rotvec()

pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

smpl_parser_n = SMPL_Parser(model_path=r"smpl", gender="neutral")


###### Shape fitting
trans = torch.zeros([1, 3]) + torch.tensor([0, 0.22, 0])
beta = torch.zeros([1, 10])
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans)
print(joints[:, 0], joints.size())
offset = joints[:, 0] - trans
root_trans_offset = trans + offset


pose_aa_robot = torch.cat([torch.zeros((1, 1, 3)), pose_aa_robot, torch.zeros(1, len(robot.cfg.Extend.extend_parent_idx), 3)], dim=1)
fk_return = robot.fk_batch(pose_aa_robot[None, ], root_trans_offset[None, 0:1])

shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
scale = Variable(0.01 * torch.ones([1]).to(device), requires_grad=True)
optimizer_shape = torch.optim.Adam([shape_new, scale],lr=0.1)

for iteration in range(1000):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
    root_pos = joints[:, 0]
    joints = (joints - joints[:, 0]) * scale + root_pos
    diff = fk_return.global_translation[:, :, robot.cfg.robot_joint_pick_idx] - joints[:, robot.cfg.smpl_joint_pick_idx]
    loss_g = diff.norm(dim = -1).mean() 
    loss = loss_g
    if iteration % 100 == 0:
        print(iteration, loss.item() * 1000, root_pos)

    optimizer_shape.zero_grad()
    loss.backward()
    optimizer_shape.step()
    
import ipdb; ipdb.set_trace()
joblib.dump((shape_new.detach(), scale), r"shape_optimized_v1.pkl")
joblib.dump(joints[:, robot.cfg.smpl_joint_pick_idx], r"smpl_position")
joblib.dump(fk_return.global_translation[0, :, robot.cfg.robot_joint_pick_idx], r"fk_position")
