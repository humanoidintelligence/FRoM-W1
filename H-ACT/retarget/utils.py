import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as RRR
from scipy.spatial.transform import Slerp, Rotation
import torch

# Just body + hands
SMPLX_BODY_BONES = [
    0.0, 0.0, 0.0,
    0.5170168002319281, -0.8242100032985323, -0.23102233883152715, 
    -0.512304434814126, -0.8413885532775784, -0.17207354978177053, 
    -0.02437497773056777, 0.9695524519707751, -0.24366724621983335, 
    0.14288347087678838, -0.9894350743911933, -0.02454577494747777, 
    -0.12097257675135349, -0.9915877731218379, -0.04603552044909732, 
    0.07139785084809115, 0.9964372191853292, -0.04488946394335843, 
    -0.1068580852217544, -0.9911794574602667, -0.07838739406769607, 
    0.037463744766768226, -0.9980925365672985, -0.04906839793124481, 
    -0.18712964862763354, 0.8626998767553236, 0.4698202708876958, 
    0.3377931257951294, -0.41456247034725274, 0.8450051002130262, 
    -0.280974511193582, -0.4222565009396134, 0.8618309733862207, 
    -0.07214764875054516, 0.9796091746262603, -0.18751123472321926, 
    0.4777726280335685, 0.8753267554837316, -0.07440551686905858, 
    -0.4876135185030301, 0.8622442708939044, -0.1369951437529178, 
    0.15142019675854623, 0.980315860563546, 0.12669891612292816, 
    0.893997681220286, 0.4328173176921888, -0.11591878850292976, 
    -0.8812978143728383, 0.4598559036751545, -0.10884180449877767, 
    0.9497887030343014, -0.2696638608488966, -0.15869072748629895, 
    -0.9864596334606464, -0.13276201561508488, -0.0962889713373828, 
    0.9957332535001641, 0.09175938778346185, -0.0097685002419672, 
    -0.9979510055981831, -0.01814620594149265, -0.06135491475672568, 

    0.979013914973406, -0.08347063703479499, 0.1859145365295892, 
    0.937273333777617, -0.3467544834018322, 0.03576876102645852, 
    0.2869455751332447, -0.9519127179350296, -0.10734771544361472, 
    0.997673199674327, -0.05770646697777134, -0.0363038279860881, 
    0.8571090613714573, -0.4927288101477402, -0.15027218305931436, 
    0.2096253556938126, -0.9745204567044462, -0.07978881428511288, 
    0.8767591768335594, -0.15167126233745332, -0.4563868593455223, 
    0.6181938694276978, -0.7366989056151588, -0.27406211423227006, 
    0.19749001987155582, -0.9802660678836584, 0.00866534996093828, 
    0.9587978214343472, -0.09118837586522949, -0.2690561668702549, 
    0.8245334278178931, -0.561913262517145, -0.06631298454108067, 
    0.17779474826415056, -0.9834795563269039, 0.034002271631543936, 
    0.7923224859706394, -0.35051300148479847, 0.4993649207266538, 
    0.7155250288863992, -0.5833297869956893, 0.38438214219983013, 
    0.9145366817955072, -0.31839305248875593, 0.24949500940906333, 
    -0.9747752340922156, -0.1149937712057153, 0.19128376929633997, 
    -0.9372687226424627, -0.34676575045923524, 0.03578035907507009, 
    -0.28697258195778236, -0.9519091605769129, -0.1073070588452662, 
    -0.995575101591021, -0.08735863389908446, -0.03461936941703828, 
    -0.8571087601731756, -0.4927273340727708, -0.1502787407679196, 
    -0.2096067577522239, -0.9745256905234924, -0.07977374749403984, 
    -0.8679292043983766, -0.18659095522836916, -0.4603069627523407, 
    -0.6181859178569569, -0.7367155216182728, -0.2740353833505526, 
    -0.1974885004914507, -0.9802661457948039, 0.008691124236957214, 
    -0.9545784196988403, -0.123663959821445, -0.2710849785851166, 
    -0.8245284713480929, -0.5619206702250122, -0.06631184208589805, 
    -0.1777912321623163, -0.9834778453715315, 0.03407007663749788, 
    -0.7566963314167896, -0.41299253981089856, 0.5068011761467633, 
    -0.7155269290181653, -0.583322873330875, 0.3843890969912958, 
    -0.9145924604074775, -0.31827331389302604, 0.2494433141815671,]

class HybrIKJointsToRotmat:
    def __init__(self):     
        # TODO: last finger ?         
        self.num_nodes = 52
                         #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21]
        self.naive_hybrik = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                             # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21]
        self.parents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  9,  9,  12, 13, 14, 16, 17, 18, 19, 
                #       22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 
                        20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
                #       37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
                        21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50]

                #    -1: multiple children
                #    -2: no child
                #    [0,  1, 2, 3, 4, 5, 6, 7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.child = [-1, 4, 5, 6, 7, 8, 9, 10, 11, -1, -2, -2, 15, 16, 17, -2, 18, 19, 20, 21, -1, -1,
                #     22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 
                      23, 24, -2, 26, 27, -2, 29, 30, -2, 32, 33, -2, 35, 36, -2,
                #     37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
                      38, 39, -2, 41, 42, -2, 44, 45, -2, 47, 48, -2, 50, 51, -2           
                      ]
        self.bones = np.reshape(np.array(SMPLX_BODY_BONES), [52, 3])[:self.num_nodes]

    def multi_child_rot(self, t, p,
                        pose_global_parent):
        """
        t: B x 3 x child_num
        p: B x 3 x child_num
        pose_global_parent: B x 3 x 3
        """
        m = np.matmul(t, np.transpose(np.matmul(np.linalg.inv(pose_global_parent), p), [0, 2, 1]))
        u, s, vt = np.linalg.svd(m)       
        r = np.matmul(np.transpose(vt, [0, 2, 1]), np.transpose(u, [0, 2, 1]))

        err_det_mask = (np.linalg.det(r) < 0.0).reshape(-1, 1, 1)
        id_fix = np.reshape(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
                            [1, 3, 3])
        r_fix = np.matmul(np.transpose(vt, [0, 2, 1]),
                          np.matmul(id_fix,
                                    np.transpose(u, [0, 2, 1])))
        r = r * (1.0 - err_det_mask) + r_fix * err_det_mask
        return r, np.matmul(pose_global_parent, r)

    def single_child_rot(self, t, p, pose_global_parent, twist=None):
        """
        t: B x 3 x 1
        p: B x 3 x 1
        pose_global_parent: B x 3 x 3
        twist: B x 2 if given, default to None
        """
        p_rot = np.matmul(np.linalg.inv(pose_global_parent), p)
        cross = np.cross(t, p_rot, axisa=1, axisb=1, axisc=1)
        sina = np.linalg.norm(cross, axis=1, keepdims=True) / (np.linalg.norm(t, axis=1, keepdims=True) *
                                                               np.linalg.norm(p_rot, axis=1, keepdims=True))
        cross = cross / np.linalg.norm(cross, axis=1, keepdims=True)
        cosa = np.sum(t * p_rot, axis=1, keepdims=True) / (np.linalg.norm(t, axis=1, keepdims=True) *
                                                           np.linalg.norm(p_rot, axis=1, keepdims=True))
        sina = np.reshape(sina, [-1, 1, 1])
        cosa = np.reshape(cosa, [-1, 1, 1])
        skew_sym_t = np.stack([0.0 * cross[:, 0], -cross[:, 2], cross[:, 1],
                               cross[:, 2], 0.0 * cross[:, 0], -cross[:, 0],
                               -cross[:, 1], cross[:, 0], 0.0 * cross[:, 0]], 1)
        skew_sym_t = np.reshape(skew_sym_t, [-1, 3, 3])
        dsw_rotmat = np.reshape(np.eye(3), [1, 3, 3]
                                ) + sina * skew_sym_t + (1.0 - cosa) * np.matmul(skew_sym_t,
                                                                                 skew_sym_t)
        if twist is not None:
            skew_sym_t = np.stack([0.0 * t[:, 0], -t[:, 2], t[:, 1],
                                   t[:, 2], 0.0 * t[:, 0], -t[:, 0],
                                   -t[:, 1], t[:, 0], 0.0 * t[:, 0]], 1)
            skew_sym_t = np.reshape(skew_sym_t, [-1, 3, 3])
            sina = np.reshape(twist[:, 1], [-1, 1, 1])
            cosa = np.reshape(twist[:, 0], [-1, 1, 1])
            dtw_rotmat = np.reshape(np.eye(3), [1, 3, 3]
                                    ) + sina * skew_sym_t + (1.0 - cosa) * np.matmul(skew_sym_t,
                                                                                     skew_sym_t)
            dsw_rotmat = np.matmul(dsw_rotmat, dtw_rotmat)
        return dsw_rotmat, np.matmul(pose_global_parent, dsw_rotmat)

    def __call__(self, joints, twist=None):
        """
        joints: B x N x 3
        twist: B x N x 2 if given, default to None
        """
        expand_dim = False
        if len(joints.shape) == 2:
            expand_dim = True
            joints = np.expand_dims(joints, 0)
            if twist is not None:
                twist = np.expand_dims(twist, 0)
        assert (len(joints.shape) == 3)
        batch_size = np.shape(joints)[0]
        joints_rel = joints - joints[:, self.parents]
        joints_hybrik = 0.0 * joints_rel
        pose_global = np.zeros([batch_size, self.num_nodes, 3, 3])
        pose = np.zeros([batch_size, self.num_nodes, 3, 3])

        for i in range(self.num_nodes):
            if i == 0:
                joints_hybrik[:, 0] = joints[:, 0]
            else:
                joints_hybrik[:, i] = np.matmul(pose_global[:, self.parents[i]],
                                                np.reshape(self.bones[i], [1, 3, 1])).reshape(-1, 3) + \
                                      joints_hybrik[:, self.parents[i]]

            if self.child[i] == -2:
                pose[:, i] = pose[:, i] + np.eye(3).reshape(1, 3, 3)
                pose_global[:, i] = pose_global[:, self.parents[i]]
                continue

            if i == 0:
                r, rg = self.multi_child_rot(np.transpose(self.bones[[1, 2, 3]].reshape(1, 3, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [1, 2, 3]], [0, 2, 1]),
                                             np.eye(3).reshape(1, 3, 3))

            elif i == 9:
                r, rg = self.multi_child_rot(np.transpose(self.bones[[12, 13, 14]].reshape(1, 3, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [12, 13, 14]], [0, 2, 1]),
                                             pose_global[:, self.parents[9]])

            # Doney 粗选
            elif i == 16:
                axis = np.cross(joints_rel[:, 18], joints_rel[:, 20], axisa=1, axisb=1, axisc=1)
                axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-6)
                r, rg = self.multi_child_rot(np.transpose(np.concatenate([self.bones[[18]], np.array([[0, -1, 0]])], axis=0).reshape(1, 2, 3), [0, 2, 1]),
                                np.transpose(np.concatenate([joints_rel[:, [18]], axis[:, None]], axis=1), [0, 2, 1]),
                                pose_global[:, self.parents[16]])

            elif i == 17:
                axis = np.cross(joints_rel[:, 19], joints_rel[:, 21], axisa=1, axisb=1, axisc=1)
                axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-6)
                r, rg = self.multi_child_rot(np.transpose(np.concatenate([self.bones[[19]], np.array([[0, 1, 0]])], axis=0).reshape(1, 2, 3), [0, 2, 1]),
                                np.transpose(np.concatenate([joints_rel[:, [19]], axis[:, None]], axis=1), [0, 2, 1]),
                                pose_global[:, self.parents[17]])

            elif i == 20:
                r, rg = self.multi_child_rot(np.transpose(self.bones[[22, 25, 28, 31, 34]].reshape(1, 5, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [22, 25, 28, 31, 34]], [0, 2, 1]),
                                             pose_global[:, self.parents[20]])

            elif i == 21:
                r, rg = self.multi_child_rot(np.transpose(self.bones[[37, 40, 43, 46, 49]].reshape(1, 5, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [37, 40, 43, 46, 49]], [0, 2, 1]),
                                             pose_global[:, self.parents[21]])
            else:
                p = joints_rel[:, self.child[i]]
                if self.naive_hybrik[i] == 0:
                    p = joints[:, self.child[i]] - joints_hybrik[:, i]
                twi = None
                if twist is not None:
                    twi = twist[:, i]
                r, rg = self.single_child_rot(self.bones[self.child[i]].reshape(1, 3, 1),
                                              p.reshape(-1, 3, 1),
                                              pose_global[:, self.parents[i]],
                                              twi)
            pose[:, i] = r
            pose_global[:, i] = rg

        # callback

        if expand_dim:
            pose = pose[0]
        return pose

def joints2bones(joints, parents):
    ''' Decompose joints location to bone length and direction.

        Parameters
        ----------
        joints: torch.tensor Bx24x3
    '''
    assert joints.shape[1] == parents.shape[0]
    bone_dirs = torch.zeros_like(joints)
    bone_lens = torch.zeros_like(joints[:, :, :1])

    for c_id in range(parents.shape[0]):
        p_id = parents[c_id]
        if p_id == -1:
            # Parent node
            bone_dirs[:, c_id] = joints[:, c_id]
        else:
            # Child node
            # (B, 3)
            diff = joints[:, c_id] - joints[:, p_id]
            length = torch.norm(diff, dim=1, keepdim=True) + 1e-8
            direct = diff / length

            bone_dirs[:, c_id] = direct
            bone_lens[:, c_id] = length

    return bone_dirs, bone_lens

def bones2joints(bone_dirs, bone_lens, parents):
    ''' Recover bone length and direction to joints location.

        Parameters
        ----------
        bone_dirs: torch.tensor 1x24x3
        bone_lens: torch.tensor Bx24x1
    '''
    batch_size = bone_lens.shape[0]
    joints = torch.zeros_like(bone_dirs).expand(batch_size, -1, 3)

    for c_id in range(parents.shape[0]):
        p_id = parents[c_id]
        if p_id == -1:
            # Parent node
            joints[:, c_id] = bone_dirs[:, c_id]
        else:
            # Child node
            joints[:, c_id] = joints[:, p_id] + bone_dirs[:, c_id] * bone_lens[:, c_id]

    return joints

def pos2smpl(joints: np.ndarray):
    """
    joints: Batch(t), 52, 3
    """
    jts2rot_hybrik = HybrIKJointsToRotmat()

    trans = joints[:, 0]
    if joints is np.ndarray:
        joints = torch.from_numpy(joints)

    parents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  9,  9,  12, 13, 14, 16, 17, 18, 
               19,  20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
               21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50]
    parents = torch.tensor(parents)

    bone_dirs, bone_lens = joints2bones(joints, parents)
    bone_lens = torch.ones_like(bone_lens)
    joints = bones2joints(bone_dirs, bone_lens, parents)
    joints = joints.numpy()

    pose = jts2rot_hybrik(joints)

    aa = np.array([RRR.from_matrix(pose[i]).as_rotvec() for i in range(len(pose))]) # [B, 52,3]
    l_elbow = RRR.from_rotvec(aa[:,18,:]).as_euler('xyz')
    r_elbow = RRR.from_rotvec(aa[:,19,:]).as_euler('xyz')
    l_elbow[:, [0, 2]] = 0
    r_elbow[:, [0, 2]] = 0


    aa[:, 18, :] = RRR.from_euler('xyz', l_elbow).as_rotvec()
    aa[:, 19, :] = RRR.from_euler('xyz', r_elbow).as_rotvec()
    data_dict = {
        'trans': trans, # np.zeros((len(aa), 3)),
        'poses': aa.reshape(len(aa), -1),
        'gender': 'neutral',
        'mocap_framerate': 20,
        'betas': np.zeros((10))  
    }

    return data_dict

# 623 Feature to 52, 3 Position
def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    # data = torch.Tensor(data)
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions

def denormalize(features):
    mean = np.load("assets/meta/mean.npy")
    std = np.load("assets/meta/std.npy")
    mean = torch.tensor(mean).to(features)
    std = torch.tensor(std).to(features)

    features = features * std + mean
    return features

def feats2joints(features):
    features = denormalize(features)
    return recover_from_ric(features, 52)

def set_fps(data_dict, target_fps=None):
    if target_fps is None:
        return data_dict
    
    curr_fps = data_dict["fps"]
    if curr_fps == target_fps:
        return data_dict
    
    n_frames = len(data_dict["dof"])
    t_orig = np.linspace(0, 1, n_frames)
    
    n_new = int(round(n_frames * target_fps / curr_fps))
    t_new = np.linspace(0, 1, n_new)

    def linear_resample(arr):
        arr = np.asarray(arr)
        return np.vstack([
            np.interp(t_new, t_orig, arr[:, i]) for i in range(arr.shape[1])
        ]).T.astype(np.float32)

    data_dict["dof"] = linear_resample(data_dict["dof"])
    data_dict["root_trans_offset"] = linear_resample(data_dict["root_trans_offset"])
    data_dict["hand_pose"] = linear_resample(data_dict["hand_pose"])

    if "pose_aa" in data_dict:
        tmp = []
        for i in range(data_dict["pose_aa"].shape[1]):
            tmp.append(linear_resample(data_dict["pose_aa"][:, i]))
        data_dict["pose_aa"] = np.stack(tmp, axis=1)

    rots = Rotation.from_quat(data_dict["root_rot"])
    slerp = Slerp(t_orig, rots)
    new_rots = slerp(t_new).as_quat()
    data_dict["root_rot"] = new_rots.astype(np.float32)

    data_dict["fps"] = target_fps
    
    return data_dict

if __name__ == "__main__":
    data = torch.from_numpy(np.load(f"data/623/{motion_name}.npy")[0])
    smplx_pos = feats2joints(data)
    smpl_dict = pos2smpl(smplx_pos)

