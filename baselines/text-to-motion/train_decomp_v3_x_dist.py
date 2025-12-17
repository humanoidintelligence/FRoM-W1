import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainDecompOptions
from utils.plot_script import *

from networks.modules import *
from networks.trainers import DecompTrainerV3Dist
from data.dataset import MotionDatasetV2Dist
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator

def setup(rank, world_size):
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前GPU
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def plot_t2m(data, kinematic_chain, mean, std, fps, radius, save_dir):
    # data = train_dataset.inv_transform(data)
    data = data * std + mean
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


def main(rank, world_size, opt):
    """主训练函数"""
    # 设置分布式训练
    setup(rank, world_size)
    
    # 只在主进程上创建目录
    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
        os.makedirs(opt.eval_dir, exist_ok=True)
        os.makedirs(opt.log_dir, exist_ok=True)
    
    # 等待主进程创建完目录
    dist.barrier()
    
    # 设置设备
    opt.device = torch.device(f"cuda:{rank}")
    opt.gpu_id = rank
    torch.autograd.set_detect_anomaly(True)

    if opt.dataset_name == 't2m-x':
        opt.data_root = './dataset/HumanML3D-X'  # HumanML3D
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 52 # 22
        opt.max_motion_length = 300 # TODO(pli): maybe 400 for motion-x
        dim_pose = 623 # 263
        radius = 4
        fps = 30 # 20
        kinematic_chain = paramUtil.t2m_x_kinematic_chain
    # elif opt.dataset_name == 'kit':
    #     opt.data_root = './dataset/KIT-ML'
    #     opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    #     opt.text_dir = pjoin(opt.data_root, 'texts')
    #     opt.joints_num = 21
    #     radius = 240 * 8
    #     fps = 12.5
    #     dim_pose = 251
    #     opt.max_motion_length = 196
    #     kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    movement_enc = MovementConvEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)

    # 将模型移到GPU并用DDP包装
    movement_enc = movement_enc.to(rank)
    movement_dec = movement_dec.to(rank)
    
    movement_enc = DDP(movement_enc, device_ids=[rank])
    movement_dec = DDP(movement_dec, device_ids=[rank])

    # 只在主进程上打印参数信息
    if rank == 0:
        all_params = 0
        pc_mov_enc = sum(param.numel() for param in movement_enc.module.parameters())
        print(movement_enc.module)
        print("Total parameters of prior net: {}".format(pc_mov_enc))
        all_params += pc_mov_enc

        pc_mov_dec = sum(param.numel() for param in movement_dec.module.parameters())
        print(movement_dec.module)
        print("Total parameters of posterior net: {}".format(pc_mov_dec))
        all_params += pc_mov_dec
        print(f"Total parameters: {all_params}")

    # 创建数据集和数据加载器
    train_dataset = MotionDatasetV2Dist(opt, mean, std, train_split_file)
    val_dataset = MotionDatasetV2Dist(opt, mean, std, val_split_file)

    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 调整batch_size以适应多卡训练
    per_gpu_batch_size = opt.batch_size // world_size
    if per_gpu_batch_size < 1:
        per_gpu_batch_size = 1

    train_loader = DataLoader(
        train_dataset, 
        batch_size=per_gpu_batch_size, 
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=per_gpu_batch_size, 
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    # 创建训练器并开始训练
    trainer = DecompTrainerV3Dist(opt, movement_enc, movement_dec)
    trainer.train(train_loader, val_loader, kinematic_chain, train_dataset.mean, train_dataset.std, fps, radius, plot_t2m)
    
    # 清理分布式环境
    cleanup()
    
if __name__ == '__main__':
    parser = TrainDecompOptions()
    opt = parser.parse()
    
    # 设置保存路径
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)
    
    # 获取世界大小和当前进程排名
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    # 调用主函数
    main(rank, world_size, opt)