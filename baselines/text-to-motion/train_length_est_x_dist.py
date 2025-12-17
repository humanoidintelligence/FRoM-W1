import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from os.path import join as pjoin

from options.train_options import TrainLenEstOptions

from networks.modules import *
from networks.trainers import LengthEstTrainerDist, collate_fn
from data.dataset import Text2MotionDatasetDist
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

def main(rank, world_size, opt):
    """主训练函数"""
    # 设置分布式训练
    setup(rank, world_size)
    
    # 只在主进程上创建目录
    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
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
        dim_pose = 623 # 263
    # elif opt.dataset_name == 'kit':
    #     opt.data_root = './dataset/KIT-ML'
    #     opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    #     opt.text_dir = pjoin(opt.data_root, 'texts')
    #     opt.joints_num = 21
    #     dim_pose = 251
    else:
        raise KeyError('Dataset Does Not Exist')

    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    opt.max_motion_length = 300 # 196 for humanml3d， 300 for humanml3d-x
    num_classes = 300 // opt.unit_length
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    if opt.estimator_mod == 'bigru':
        estimator = MotionLenEstimatorBiGRU(dim_word, dim_pos_ohot, 512, num_classes)
    else:
        raise Exception('Estimator Mode is not Recognized!!!')

    # 将模型移到GPU并用DDP包装
    estimator = estimator.to(rank)
    estimator = DDP(estimator, device_ids=[rank])

    # 只在主进程上打印参数信息
    if rank == 0:
        pc_est = sum(param.numel() for param in estimator.module.parameters())
        print(estimator.module)
        print("Total parameters of posterior net: {}".format(pc_est))

    # 创建数据集
    train_dataset = Text2MotionDatasetDist(opt, mean, std, train_split_file, w_vectorizer)
    val_dataset = Text2MotionDatasetDist(opt, mean, std, val_split_file, w_vectorizer)

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
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=per_gpu_batch_size, 
        sampler=val_sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    # 创建训练器并开始训练
    trainer = LengthEstTrainerDist(opt, estimator)
    trainer.train(train_loader, val_loader)
    
    # 清理分布式环境
    cleanup()
    
if __name__ == '__main__':
    parser = TrainLenEstOptions()
    opt = parser.parse()
    
    # 设置保存路径
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)
    
    # 获取世界大小和当前进程排名
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    # 调用主函数
    main(rank, world_size, opt)