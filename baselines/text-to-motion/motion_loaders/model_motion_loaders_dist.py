from torch.utils.data import DataLoader, Dataset, DistributedSampler
from utils.get_opt import get_opt
from motion_loaders.comp_v6_model_dataset_dist import CompV6GeneratedDataset
from utils.word_vectorizer import WordVectorizer
import numpy as np
from torch.utils.data._utils.collate import default_collate
import torch.distributed as dist


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class MMGeneratedDataset(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            if len(motion) < self.opt.max_motion_length:
                motion = np.concatenate([motion,
                                         np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
                                         ], axis=0)
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        # print(m_lens)
        # print(sort_indx)
        # print(m_lens[sort_indx])
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens



def get_motion_loader(opt_path, batch_size, ground_truth_dataset, mm_num_samples, mm_num_repeats, device, rank=0, world_size=1):
    opt = get_opt(opt_path, device)

    # Currently the configurations of two datasets are almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
    else:
        raise KeyError('Dataset not recognized!!')
    
    if dist.is_available() and dist.is_initialized():
        print(f'Rank {rank}: Generating {opt.name} ...')
    else:
        print('Generating %s ...' % opt.name)

    if 'v6' in opt.name:
        dataset = CompV6GeneratedDataset(opt, ground_truth_dataset, w_vectorizer, mm_num_samples, mm_num_repeats)
    else:
        raise KeyError('Dataset not recognized!!')

    mm_dataset = MMGeneratedDataset(opt, dataset, w_vectorizer)

    # 为motion_loader创建分布式采样器
    if world_size > 1:
        motion_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # 对于生成的数据，通常不需要shuffle
            drop_last=True
        )
        motion_shuffle = False
    else:
        motion_sampler = None
        motion_shuffle = False

    motion_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        drop_last=True, 
        num_workers=4,
        sampler=motion_sampler,
        shuffle=motion_shuffle,
        pin_memory=True
    )

    # 为mm_motion_loader创建分布式采样器
    # 注意：mm_motion_loader通常batch_size=1，需要特殊处理
    if world_size > 1:
        mm_sampler = DistributedSampler(
            mm_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False  # 多模态评估可能需要保留所有样本
        )
        mm_shuffle = False
    else:
        mm_sampler = None
        mm_shuffle = False

    mm_motion_loader = DataLoader(
        mm_dataset, 
        batch_size=1, 
        num_workers=1,
        sampler=mm_sampler,
        shuffle=mm_shuffle,
        pin_memory=True
    )

    if dist.is_available() and dist.is_initialized():
        print(f'Rank {rank}: Generated Dataset Loading Completed!!!')
    else:
        print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader