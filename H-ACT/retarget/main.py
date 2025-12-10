from body_retarget import process_data, load_amass_data
from hand_retarget import retarget_from_rotvec
from utils import pos2smpl, feats2joints, set_fps

import numpy as np
import torch
import joblib
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SMPLX_OUTPUT_PATH = ""
def process(file_path, out_path):
    data = torch.from_numpy(np.load(file_path)[0])
    smplx_pos = feats2joints(data)
    smpl_dict = pos2smpl(smplx_pos)
    np.savez(SMPLX_OUTPUT_PATH, **smpl_dict)
    # feats2joints(): from 623 to 52 joint 3D position
    # pos2smpl(): from 52 joints 3D position to smplx dict
    
    amass_data = load_amass_data(SMPLX_OUTPUT_PATH) # load file: {"pose", "trans", "root_rot"}
    robot_data = process_data(amass_data) # retarget body: from smpl dict to h1/g1

    hand_data = retarget_from_rotvec(amass_data['poses'][:, 66:], hand_type="inspire") # retarget hand: from smplx dict to inspire/dex
    robot_data.update({"hand_pose": hand_data})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(robot_data, out_path)

if __name__ == "__main__":
    for file in os.listdir("data/623"):
        if file.endswith(".npy"):
            src = os.path.join("data/623", file)
            smplx_path = os.path.join("data/smplx", file.replace(".npy", ".npz"))
            out_path = os.path.join("data/output", file.replace(".npy", ".pkl"))

            data = torch.from_numpy(np.load(src)[0])
            smplx_pos = feats2joints(data)
            smpl_dict = pos2smpl(smplx_pos)
            # smpl_dict = np.load("assets/data/smplx_output_new.npz")
            np.savez(smplx_path, **smpl_dict)
            amass_data = load_amass_data(smplx_path)
    
            robot_data = process_data(amass_data, "G1") # available robot: H1, G1, H121(H1 19dof and 2dof from wrist)
            hand_data = retarget_from_rotvec(smpl_dict['poses'][:, 66:], hand_type="dex3") # available hand: inspire, dex3
            robot_data.update({"hand_pose": hand_data})
            robot_data = {"motion 0": set_fps(robot_data, 60)}
            print(robot_data["motion 0"]["dof"].shape)
            joblib.dump(robot_data, out_path)
    