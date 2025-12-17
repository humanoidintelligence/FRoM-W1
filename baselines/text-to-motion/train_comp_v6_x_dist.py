import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *

from networks.modules import *
from networks.trainers import CompTrainerV6Dist
from data.dataset import Text2MotionDatasetDist
from scripts.motion_process import *
# from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator
import random 


def setup(rank, world_size):
    """初始化分布式训练环境"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前GPU
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def plot_t2m(data, save_dir, captions, ep_curves=None, mean=None, std=None, kinematic_chain=None, fps=None, radius=None, joints_num=None):
    data = data * std + mean
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)
        if ep_curves is not None:
            ep_curve = ep_curves[i]
            plt.plot(ep_curve)
            plt.title(caption)
            save_path = pjoin(save_dir, '%02d.png' % (i))
            plt.savefig(save_path)
            plt.close()

def loadDecompModel(opt, rank):
    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    if not opt.is_continue:
        checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
                                map_location=f"cuda:{rank}")
        movement_enc.load_state_dict(checkpoint['movement_enc'])
        movement_dec.load_state_dict(checkpoint['movement_dec'])

    return movement_enc, movement_dec

def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)

    seq_posterior = TextDecoder(text_size=text_size,
                                input_size=opt.dim_att_vec + opt.dim_movement_latent * 2,
                                output_size=opt.dim_z,
                                hidden_size=opt.dim_pos_hidden,
                                n_layers=opt.n_layers_pos)

    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    return text_encoder, seq_prior, seq_posterior, seq_decoder, att_layer

def set_seed(seed=42):
    """
    设置随机种子以保证结果可复现
    
    参数:
        seed (int): 随机种子值，默认为42
    """
    # Python内置随机数生成器
    random.seed(seed)
    
    # NumPy随机数生成器
    np.random.seed(seed)
    
    # PyTorch随机数生成器
    torch.manual_seed(seed)
    
    # 如果使用CUDA（GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        
        # 额外的CUDA设置以确保确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置Python哈希种子（用于字典等数据结构的哈希）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"随机种子已设置为: {seed}")

def main(rank, world_size, opt):
    set_seed(seed=42)
    
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
        radius = 4
        fps = 30 # 20
        opt.max_motion_length = 300
        opt.dim_pose = 623 # 263
        kinematic_chain = paramUtil.t2m_x_kinematic_chain # t2m_kinematic_chain
    # elif opt.dataset_name == 'kit':
    #     opt.data_root = './dataset/KIT-ML'
    #     opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    #     opt.text_dir = pjoin(opt.data_root, 'texts')
    #     opt.joints_num = 21
    #     radius = 240 * 8
    #     fps = 12.5
    #     opt.dim_pose = 251
    #     opt.max_motion_length = 196
    #     kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    opt.dim_word = 300
    opt.dim_pos_ohot = len(POS_enumerator)
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')
    
    # 加载分解模型
    movement_enc, movement_dec = loadDecompModel(opt, rank)

    # 构建主模型
    text_encoder, seq_prior, seq_posterior, seq_decoder, att_layer = build_models(opt)
    
    # 将模型移到GPU并用DDP包装
    movement_enc = movement_enc.to(rank)
    movement_dec = movement_dec.to(rank)
    text_encoder = text_encoder.to(rank)
    seq_prior = seq_prior.to(rank)
    seq_posterior = seq_posterior.to(rank)
    seq_decoder = seq_decoder.to(rank)
    att_layer = att_layer.to(rank)
    
    movement_enc = DDP(movement_enc, device_ids=[rank])
    movement_dec = DDP(movement_dec, device_ids=[rank])
    text_encoder = DDP(text_encoder, device_ids=[rank])
    seq_prior = DDP(seq_prior, device_ids=[rank])
    seq_posterior = DDP(seq_posterior, device_ids=[rank])
    seq_decoder = DDP(seq_decoder, device_ids=[rank])
    att_layer = DDP(att_layer, device_ids=[rank])

    # 只在主进程上打印参数信息
    if rank == 0:
        all_params = 0
        for name, model in [("Text Encoder", text_encoder), 
                           ("Sequence Prior", seq_prior),
                           ("Sequence Posterior", seq_posterior),
                           ("Sequence Decoder", seq_decoder),
                           ("Attention Layer", att_layer),
                           ("Movement Encoder", movement_enc),
                           ("Movement Decoder", movement_dec)]:
            model_params = sum(param.numel() for param in model.module.parameters())
            print(f"{name}: {model_params} parameters")
            all_params += model_params
        print(f"Total parameters: {all_params}")

    # 创建数据集和数据加载器
    train_dataset = Text2MotionDatasetDist(opt, mean, std, train_split_file, w_vectorizer)
    val_dataset = Text2MotionDatasetDist(opt, mean, std, val_split_file, w_vectorizer)

    # # 创建分布式采样器
    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # # 调整batch_size以适应多卡训练
    per_gpu_batch_size = opt.batch_size // world_size
    if per_gpu_batch_size < 1:
        per_gpu_batch_size = 1

    # train_loader = DataLoader(
    #     train_dataset, 
    #     batch_size=per_gpu_batch_size, 
    #     sampler=train_sampler,
    #     num_workers=8,
    #     pin_memory=True,
    #     drop_last=True
    # )
    
    # val_loader = DataLoader(
    #     val_dataset, 
    #     batch_size=per_gpu_batch_size, 
    #     sampler=val_sampler,
    #     num_workers=8,
    #     pin_memory=True,
    #     drop_last=True
    # )

    # 创建训练器并开始训练
    trainer = CompTrainerV6Dist(opt, text_encoder, seq_prior, seq_decoder, att_layer, movement_dec,
                               mov_enc=movement_enc, seq_post=seq_posterior)
    
    trainer.train(train_dataset, val_dataset, per_gpu_batch_size, train_dataset.mean, train_dataset.std, 
                  kinematic_chain, fps, radius, opt.joints_num,
                  plot_t2m)
    
    # 清理分布式环境
    cleanup()

if __name__ == '__main__':
    parser = TrainCompOptions()
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