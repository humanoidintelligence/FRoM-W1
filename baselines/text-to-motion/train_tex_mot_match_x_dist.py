import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from os.path import join as pjoin
import torch
from options.train_options import TrainTexMotMatchOptions

from networks.modules import *
from networks.trainers import TextMotionMatchTrainerDist
from data.dataset import Text2MotionDatasetV2Dist, collate_fn, custom_collate_fn
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator


def setup(rank, world_size):
    """初始化分布式训练环境"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前GPU
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def build_models(opt, rank, dim_pose, dim_word, dim_pos_ohot):
    movement_enc = MovementConvEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=dim_word,
                                  pos_size=dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)
    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)
    
    if not opt.is_continue:
        checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
                               map_location=f"cuda:{rank}")
        movement_enc.load_state_dict(checkpoint['movement_enc'])
    
    return text_enc, motion_enc, movement_enc

def main(rank, world_size, opt):
    """主训练函数"""
    # 设置分布式训练
    setup(rank, world_size)
    
    # 只在主进程上创建目录
    if rank == 0:
        opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
        opt.model_dir = pjoin(opt.save_root, 'model')
        opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)
        opt.eval_dir = pjoin(opt.save_root, 'eval')

        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.eval_dir, exist_ok=True)
        os.makedirs(opt.log_dir, exist_ok=True)
    
    # 等待主进程创建完目录
    dist.barrier()
    
    # 设置设备
    opt.device = torch.device(f"cuda:{rank}")
    opt.gpu_id = rank
    torch.autograd.set_detect_anomaly(True)

    if opt.dataset_name == 't2m-x':
        opt.data_root = './dataset/HumanML3D-X' # HumanML3D
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 52 # 22
        opt.max_motion_length = 300 # 196 for humanml 20fps, 300 for humanml-x 30fps
        dim_pose = 623 # 623
        # num_classes = 200 // opt.unit_length
        meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.comp_name, 'meta')
    elif opt.dataset_name == 't2m-x-noise':
        opt.data_root = './dataset/HumanML3D-X-Noise' # HumanML3D
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 52 # 22
        opt.max_motion_length = 300 # 196 for humanml 20fps, 300 for humanml-x 30fps
        dim_pose = 623 # 623
        # num_classes = 200 // opt.unit_length
        opt.dataset_name = 't2m-x'
        meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.comp_name, 'meta')
    elif opt.dataset_name == 't2m-x-rephrase':
        opt.data_root = './dataset/HumanML3D-X-Rephrase' # HumanML3D
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 52 # 22
        opt.max_motion_length = 300 # 196 for humanml 20fps, 300 for humanml-x 30fps
        dim_pose = 623 # 623
        # num_classes = 200 // opt.unit_length
        opt.dataset_name = 't2m-x'
        meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.comp_name, 'meta')
    # elif opt.dataset_name == 'kit':
    #     opt.data_root = './dataset/KIT-ML'
    #     opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    #     opt.text_dir = pjoin(opt.data_root, 'texts')
    #     opt.joints_num = 21
    #     radius = 240 * 8
    #     fps = 12.5
    #     dim_pose = 251
    #     opt.max_motion_length = 196
    #     num_classes = 200 // opt.unit_length
    #     meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'Comp_v6_KLD005', 'meta')
    else:
        raise KeyError('Dataset Does Not Exist')

    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)

    mean = np.load(pjoin(meta_root, 'mean.npy'))
    std = np.load(pjoin(meta_root, 'std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    # 构建模型
    text_encoder, motion_encoder, movement_encoder = build_models(opt, rank, dim_pose, dim_word, dim_pos_ohot)

    # 将模型移到GPU并用DDP包装
    text_encoder = text_encoder.to(rank)
    motion_encoder = motion_encoder.to(rank)
    movement_encoder = movement_encoder.to(rank)
    
    text_encoder = DDP(text_encoder, device_ids=[rank])
    motion_encoder = DDP(motion_encoder, device_ids=[rank])
    movement_encoder = DDP(movement_encoder, device_ids=[rank])

    # 只在主进程上打印参数信息
    if rank == 0:
        pc_text_enc = sum(param.numel() for param in text_encoder.module.parameters())
        print(text_encoder.module)
        print("Total parameters of text encoder: {}".format(pc_text_enc))
        pc_motion_enc = sum(param.numel() for param in motion_encoder.module.parameters())
        print(motion_encoder.module)
        print("Total parameters of motion encoder: {}".format(pc_motion_enc))
        print("Total parameters: {}".format(pc_motion_enc + pc_text_enc))

    # 创建数据集
    train_dataset = Text2MotionDatasetV2Dist(opt, mean, std, train_split_file, w_vectorizer)
    val_dataset = Text2MotionDatasetV2Dist(opt, mean, std, val_split_file, w_vectorizer)

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
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=per_gpu_batch_size, 
        sampler=val_sampler,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True
    )

    # 创建训练器并开始训练
    trainer = TextMotionMatchTrainerDist(opt, text_encoder, motion_encoder, movement_encoder)
    trainer.train(train_loader, val_loader)
    
    # 清理分布式环境
    cleanup()

if __name__ == '__main__':
    parser = TrainTexMotMatchOptions()
    opt = parser.parse()
    
    # 获取世界大小和当前进程排名
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    # 调用主函数
    main(rank, world_size, opt)