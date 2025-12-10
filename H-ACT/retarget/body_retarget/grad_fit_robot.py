      
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from .smpl_parser import (
    SMPL_Parser,
    SMPL_BONE_ORDER_NAMES
)
import joblib
from .robot import Humanoid_Batch
from .robot_config import g1Config, h1Config, h121Config
from torch.autograd import Variable
from .utils import axis_angle_to_matrix
from tqdm import tqdm
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed

SMPL_PARENT = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]
device = torch.device("cpu")

SMOOTH_WEIGHT = 1e-3
MAX_ITER_PATIENCE = 100
MIN_LR = 1e-5
INIT_LR = 1e-1
WEIGHT_DECAY = 1e-5

smpl_parser_n = SMPL_Parser(model_path="models/smpl", gender="neutral")
smpl_parser_n.to(device)

shape_new, scale = joblib.load("assets/beta/shape_optimized_g1.pkl")
shape_new = shape_new.to(device)
scale = scale.to(device)

robot = Humanoid_Batch(cfg=g1Config, device = device)

def get_joint_global_rot(smpl_pose: torch.Tensor) -> torch.Tensor:
    """
    Args:
        smpl_pose: (t, 72), axis-angle pose vectors
    Returns:
        global_rot_mats: (t, 24, 3, 3), global rotation matrix of each joint
    """
    t = smpl_pose.shape[0]
    pose_axis_angle = smpl_pose.view(t, 24, 3)  # (t, 24, 3)
    # pose_axis_angle = torch.zeros_like(pose_axis_angle)
    local_rot_mats = axis_angle_to_matrix(pose_axis_angle)  # (t, 24, 3, 3)

    global_rot_mats = torch.zeros_like(local_rot_mats)  # (t, 24, 3, 3)

    for i in range(24):
        parent = SMPL_PARENT[i]
        if parent == -1:
            global_rot_mats[:, i] = local_rot_mats[:, i]
        else:
            global_rot_mats[:, i] = torch.matmul(global_rot_mats[:, parent], local_rot_mats[:, i])

    return global_rot_mats

def geodesic_loss(R1, R2):
    R_diff = R1 @ R2.transpose(-1, -2)
    cos = ((R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]) - 1) / 2
    cos = torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)

def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if 'mocap_framerate' in  entry_data:
        framerate = entry_data['mocap_framerate']
    elif 'mocap_frame_rate' in  entry_data:
        framerate = entry_data['mocap_frame_rate']
    else:
        print(f"ERROR LOAD {data_path}: {entry_data.keys()}")
        return 


    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    pose_aa[:, :3] = (sRot.from_quat([0.5, 0.5, 0.5, 0.5]) * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": 30 # framerate
    }
    
def load_smplx_data(data_path):
    motion_data = np.load(data_path)
    frame_num = len(motion_data)
    return {
        "pose_aa": np.concatenate([motion_data[:, :66], np.zeros((frame_num, 6))], axis=1),
        "trans": motion_data[:, 309:309+3],
        "betas": motion_data[:, 312:],
        "fps": 30
    }
    
def process_data(amass_data: dict, robot: str="G1", smpl_parser_n=smpl_parser_n, device=device):
    if robot == "G1":
        shape_new, scale = joblib.load("assets/beta/shape_optimized_g1.pkl")
        shape_new = shape_new.to(device)
        scale = scale.to(device)

        robot = Humanoid_Batch(cfg=g1Config, device = device)
    elif robot == "H1":
        shape_new, scale = joblib.load("assets/beta/shape_optimized_h1.pkl")
        shape_new = shape_new.to(device)
        scale = scale.to(device)

        robot = Humanoid_Batch(cfg=h1Config, device = device)
    elif robot == "H121":
        shape_new, scale = joblib.load("assets/beta/shape_optimized_h121.pkl")
        shape_new = shape_new.to(device)
        scale = scale.to(device)

        robot = Humanoid_Batch(cfg=h121Config, device = device)
    else:
        raise ValueError(f"robot {robot} not supported")
    if amass_data is None:
        raise Exception("Error with data_path")

    skip = int(amass_data['fps'] // 30)
    trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device)

    if len(trans) >= 1000:
        raise ValueError(f"Too long!{len(trans)}")
    
    N = trans.shape[0]
    pose_aa_walk = torch.from_numpy(
        amass_data['pose_aa'][::skip]
    ).float().to(device)

    verts, joints = smpl_parser_n.get_joints_verts(
        pose_aa_walk, torch.zeros((1, 10)).to(device), trans
    )
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset

    dof_pos = torch.zeros((1, N, robot.cfg.JOINT_NUM, 1)).to(device)
    gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)    

    dof_pos_new = Variable(dof_pos, requires_grad=True)
    # optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)
    optimizer_pose = torch.optim.Adam([dof_pos_new], lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pose, 'min', patience=10, factor=0.5, min_lr=MIN_LR)

    patience_counter = 0
    for iteration in range(1000):
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
        smpl_rot = get_joint_global_rot(pose_aa_walk)
        root_pos = joints[:, 0:1]
        joints = (joints - root_pos) * scale + root_pos
        
        pose_aa_robot_new = torch.cat(
            [gt_root_rot[None, :, None], robot.cfg.ROBOT_ROTATION_AXIS[None, ].to(device) * dof_pos_new, torch.zeros((1, N, len(robot.cfg.Extend.extend_parent_idx), 3)).to(device)],
            axis=2
        ).to(device)
        
        fk_return = robot.fk_batch(pose_aa_robot_new, root_trans_offset[None, ])
        diff = fk_return['global_translation_extend'][:, :, robot.cfg.robot_joint_pick_idx] - joints[:, robot.cfg.smpl_joint_pick_idx]
        # rot_diff
        if robot.cfg.wrist_pick_idx == []:
            loss_rot = 0
        else:
            loss_rot = geodesic_loss(fk_return['global_rotation_mat_extend'][0, :, robot.cfg.wrist_pick_idx], smpl_rot[:, robot.cfg.smpl_wrist_pick_idx]).mean()
        # loss_rot = so3_log_map(diff_rot.view(-1, 3, 3)).abs().mean()

        loss_g = diff.norm(dim=-1).mean()
        loss_smooth = torch.norm(dof_pos_new[:, 1:] - dof_pos_new[:, :-1], p=2)
        loss = loss_g + loss_smooth * SMOOTH_WEIGHT + loss_rot * 0.01
        # print(loss_g.item(), loss_rot.item())
        
        scheduler.step(loss)
        if optimizer_pose.param_groups[0]['lr'] <= MIN_LR:
                patience_counter += 1
        if patience_counter >= MAX_ITER_PATIENCE:
            # print(f"Early stopping at epoch {iteration}")
            break

        optimizer_pose.zero_grad()
        loss.backward()
        optimizer_pose.step()
        dof_pos_new.data.clamp_(robot.joints_range[:, 0, None], robot.joints_range[:, 1, None])
        
    dof_pos_new.data.clamp_(robot.joints_range[:, 0, None], robot.joints_range[:, 1, None])
    
    pose_aa_robot_new = torch.cat([gt_root_rot[None, :, None], robot.cfg.ROBOT_ROTATION_AXIS[None, ].to(device) * dof_pos_new, torch.zeros((1, N, len(robot.cfg.Extend.extend_parent_idx), 3)).to(device)], axis = 2)
    fk_return = robot.fk_batch(pose_aa_robot_new, root_trans_offset[None, ])

    root_trans_offset_dump = root_trans_offset.clone()
    root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08

    result = {
            "body_names": robot.model_names,
            "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy()[:, [2, 0, 1]],
            "pose_aa": pose_aa_robot_new.squeeze().cpu().detach().numpy(),
            "dof": dof_pos_new.squeeze().detach().cpu().numpy(),
            "root_rot": (sRot.from_rotvec(gt_root_rot.cpu().numpy())).as_quat(),
            "fps": 30,
        }
    
    print(geodesic_loss(fk_return['global_rotation_mat_extend'][-1, :, robot.cfg.wrist_pick_idx], smpl_rot[:, robot.cfg.smpl_wrist_pick_idx])[-1])
    torch.cuda.empty_cache()
    return result

if __name__ == "__main__":
    device = torch.device("cuda:0")

    SMOOTH_WEIGHT = 1e-3
    MAX_ITER_PATIENCE = 100
    MIN_LR = 1e-5
    INIT_LR = 1e-1
    WEIGHT_DECAY = 1e-5
    
    smpl_parser_n = SMPL_Parser(model_path="smpl", gender="neutral")
    smpl_parser_n.to(device)

    shape_new, scale = joblib.load("shape_optimized_v1.pkl")
    shape_new = shape_new.to(device)
    scale = scale.to(device)

    robot = Humanoid_Batch(cfg=robotConfig, device = device)
    data = process_data("test.npz", smpl_parser_n=smpl_parser_n, robot=robot, device=device)
    joblib.dump(data, "../test.pkl")


    