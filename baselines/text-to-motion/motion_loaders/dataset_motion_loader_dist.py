from data.dataset import Text2MotionDatasetV2Dist, collate_fn
from utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader, DistributedSampler
from utils.get_opt import get_opt
import torch.distributed as dist

def get_dataset_motion_loader(opt_path, batch_size, device, rank=0, world_size=1):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        if dist.is_available() and dist.is_initialized():
            print(f'Rank {rank}: Loading dataset {opt.dataset_name} ...')
        else:
            print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test.txt')
        dataset = Text2MotionDatasetV2Dist(opt, mean, std, split_file, w_vectorizer)
        
        # 创建分布式采样器
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True
            )
            shuffle = False  # 使用sampler时不需要在DataLoader中shuffle
        else:
            sampler = None
            shuffle = True
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=4, 
            drop_last=True,
            collate_fn=collate_fn, 
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True  # 添加pin_memory以提高数据传输效率
        )
    else:
        raise KeyError('Dataset not Recognized !!')

    if dist.is_available() and dist.is_initialized():
        print(f'Rank {rank}: Ground Truth Dataset Loading Completed!!!')
    else:
        print('Ground Truth Dataset Loading Completed!!!')
    
    return dataloader, dataset