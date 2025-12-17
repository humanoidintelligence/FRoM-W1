import os
import codecs as cs
import concurrent
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join as pjoin

import numpy as np
import spacy
import torch
from torch import distributed as dist
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path

# class Text2MotionDataset(data.Dataset):
#     """Dataset for Text2Motion generation task.

#     """
#     def __init__(self, opt, mean, std, split_file, times=1, w_vectorizer=None, eval_mode=False):
#         self.opt = opt
#         self.max_length = 20
#         self.times = times
#         self.w_vectorizer = w_vectorizer
#         self.eval_mode = eval_mode
#         min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

#         joints_num = opt.joints_num

#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, 'r') as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())

#         new_name_list = []
#         length_list = []
#         for name in tqdm(id_list):
#             try:
#                 motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
#                 if (len(motion)) < min_motion_len or (len(motion) >= 200):
#                     continue
#                 text_data = []
#                 flag = False
#                 with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
#                     for line in f.readlines():
#                         text_dict = {}
#                         line_split = line.strip().split('#')
#                         caption = line_split[0]
#                         tokens = line_split[1].split(' ')
#                         f_tag = float(line_split[2])
#                         to_tag = float(line_split[3])
#                         f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                         to_tag = 0.0 if np.isnan(to_tag) else to_tag

#                         text_dict['caption'] = caption
#                         text_dict['tokens'] = tokens
#                         if f_tag == 0.0 and to_tag == 0.0:
#                             flag = True
#                             text_data.append(text_dict)
#                         else:
#                             n_motion = motion[int(f_tag*20) : int(to_tag*20)]
#                             if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
#                                 continue
#                             new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
#                             while new_name in data_dict:
#                                 new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
#                             data_dict[new_name] = {'motion': n_motion,
#                                                     'length': len(n_motion),
#                                                     'text':[text_dict]}
#                             new_name_list.append(new_name)
#                             length_list.append(len(n_motion))

#                 if flag:
#                     data_dict[name] = {'motion': motion,
#                                        'length': len(motion),
#                                        'text':text_data}
#                     new_name_list.append(name)
#                     length_list.append(len(motion))
#             except:
#                 # Some motion may not exist in KIT dataset
#                 pass


#         name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

#         if opt.is_train:
#             # root_rot_velocity (B, seq_len, 1)
#             std[0:1] = std[0:1] / opt.feat_bias
#             # root_linear_velocity (B, seq_len, 2)
#             std[1:3] = std[1:3] / opt.feat_bias
#             # root_y (B, seq_len, 1)
#             std[3:4] = std[3:4] / opt.feat_bias
#             # ric_data (B, seq_len, (joint_num - 1)*3)
#             std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
#             # rot_data (B, seq_len, (joint_num - 1)*6)
#             std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
#                         joints_num - 1) * 9] / 1.0
#             # local_velocity (B, seq_len, joint_num*3)
#             std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
#                                                                                        4 + (joints_num - 1) * 9: 4 + (
#                                                                                                    joints_num - 1) * 9 + joints_num * 3] / 1.0
#             # foot contact (B, seq_len, 4)
#             std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
#                                                               4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

#             assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
#             np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
#             np.save(pjoin(opt.meta_dir, 'std.npy'), std)

#         self.mean = mean
#         self.std = std
#         self.length_arr = np.array(length_list)
#         self.data_dict = data_dict
#         self.name_list = name_list

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def real_len(self):
#         return len(self.data_dict)

#     def __len__(self):
#         return self.real_len() * self.times

#     def __getitem__(self, item):
#         idx = item % self.real_len()
#         data = self.data_dict[self.name_list[idx]]
#         motion, m_length, text_list = data['motion'], data['length'], data['text']
#         # Randomly select a caption
#         text_data = random.choice(text_list)
#         caption = text_data['caption']

#         max_motion_length = self.opt.max_motion_length
#         if m_length >= self.opt.max_motion_length:
#             idx = random.randint(0, len(motion) - max_motion_length)
#             motion = motion[idx: idx + max_motion_length]
#         else:
#             padding_len = max_motion_length - m_length
#             D = motion.shape[1]
#             padding_zeros = np.zeros((padding_len, D))
#             motion = np.concatenate((motion, padding_zeros), axis=0)

#         assert len(motion) == max_motion_length
#         "Z Normalization"
#         motion = (motion - self.mean) / self.std

#         if self.eval_mode:
#             tokens = text_data['tokens']
#             if len(tokens) < self.opt.max_text_len:
#                 # pad with "unk"
#                 tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
#                 sent_len = len(tokens)
#                 tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
#             else:
#                 # crop
#                 tokens = tokens[:self.opt.max_text_len]
#                 tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
#                 sent_len = len(tokens)
#             pos_one_hots = []
#             word_embeddings = []
#             for token in tokens:
#                 word_emb, pos_oh = self.w_vectorizer[token]
#                 pos_one_hots.append(pos_oh[None, :])
#                 word_embeddings.append(word_emb[None, :])
#             pos_one_hots = np.concatenate(pos_one_hots, axis=0)
#             word_embeddings = np.concatenate(word_embeddings, axis=0)
#             return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length
#         return caption, motion, m_length

class Text2MotionDatasetDist(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer=None, eval_mode=False):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 28
        self.pointer = 0
        self.eval_mode = eval_mode
        min_motion_len = 60 if 't2m' in self.opt.dataset_name else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # 使用多进程并行加载数据
        results = self._parallel_load_data(id_list, opt.motion_dir, opt.text_dir, min_motion_len)
        
        new_name_list = []
        length_list = []
        
        for name, result in zip(id_list, results):
            if result is not None:
                motion_data_list, flag = result
                
                # 处理没有时间标签的数据
                if flag:
                    data_dict[name] = {
                        'motion': motion_data_list[0]['motion'],
                        'length': motion_data_list[0]['length'],
                        'text': motion_data_list[0]['text']
                    }
                    new_name_list.append(name)
                    length_list.append(motion_data_list[0]['length'])
                
                # 处理有时间标签的数据
                for motion_data in motion_data_list[1:]:  # 第一个是原始motion，后面的是片段
                    new_name = motion_data['name']
                    data_dict[new_name] = {
                        'motion': motion_data['motion'],
                        'length': motion_data['length'],
                        'text': motion_data['text']
                    }
                    new_name_list.append(new_name)
                    length_list.append(motion_data['length'])

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            if dist.get_rank() == 0:
                import shutil
                source_dir = "./checkpoints/t2mx/text_mot_match/meta/"
                target_dir = opt.meta_dir
                
                # 复制 mean.npy
                shutil.copy2(pjoin(source_dir, 'mean.npy'), pjoin(target_dir, 'mean.npy'))
                # 复制 std.npy
                shutil.copy2(pjoin(source_dir, 'std.npy'), pjoin(target_dir, 'std.npy'))
            
            # # root_rot_velocity (B, seq_len, 1)
            # std[0:1] = std[0:1] / opt.feat_bias
            # # root_linear_velocity (B, seq_len, 2)
            # std[1:3] = std[1:3] / opt.feat_bias
            # # root_y (B, seq_len, 1)
            # std[3:4] = std[3:4] / opt.feat_bias
            # # ric_data (B, seq_len, (joints_num - 1)*3)
            # std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # # rot_data (B, seq_len, (joints_num - 1)*6)
            # std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
            #             joints_num - 1) * 9] / 1.0
            # # local_velocity (B, seq_len, joints_num*3)
            # std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
            #                                                                            4 + (joints_num - 1) * 9: 4 + (
            #                                                                                        joints_num - 1) * 9 + joints_num * 3] / 1.0
            # # foot contact (B, seq_len, 4)
            # std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
            #                                                   4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            # assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            
            # if dist.get_rank() == 0:
            #     np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            #     np.save(pjoin(opt.meta_dir, 'std.npy'), std)
            
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        
        self.check_length_consistency()
        # print ("Good Job!")
        # exit(0)
        
    def check_length_consistency(self):
        """
        检查每个样本的 data['length'] 和 self.length_arr 中的长度是否一致
        返回不一致的样本索引和详细信息
        """
        inconsistent_samples = []
        
        for idx in range(len(self.name_list)):
            # 获取当前样本的数据
            data = self.data_dict[self.name_list[idx]]
            motion, m_length, text_list = data['motion'], data['length'], data['text']
            
            # 检查长度是否一致
            if m_length != self.length_arr[idx]:
                inconsistent_samples.append({
                    'index': idx,
                    'name': self.name_list[idx],
                    'data_length': m_length,
                    'length_arr_value': self.length_arr[idx],
                    'difference': abs(m_length - self.length_arr[idx])
                })
        
        # 输出检查结果
        if not inconsistent_samples:
            print("✓ 所有样本的 data['length'] 和 length_arr 一致！")
            return True
        else:
            print(f"✗ 发现 {len(inconsistent_samples)} 个不一致的样本：")
            for sample in inconsistent_samples:
                print(f"  索引 {sample['index']}: {sample['name']} - "
                    f"data['length']={sample['data_length']}, "
                    f"length_arr={sample['length_arr_value']}, "
                    f"差异={sample['difference']}")
            exit(0)
        

    def _load_single_item(self, args):
        """加载单个数据项的函数"""
        name, motion_dir, text_dir, min_motion_len = args
        name_set = set()
        try:
            motion_path = pjoin(motion_dir, name + '.npy')
            text_path = pjoin(text_dir, name + '.txt')
            
            # 检查文件是否存在
            if not Path(motion_path).exists() or not Path(text_path).exists():
                return None
            
            motion = np.load(motion_path)
            if len(motion) < min_motion_len or len(motion) >= 300:
                return None
            
            text_data = []
            flag = False
            motion_data_list = []
            
            with cs.open(text_path, 'r') as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens
                    
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        try:
                            n_motion = motion[int(f_tag*30) : int(to_tag*30)]
                            if len(n_motion) < min_motion_len or len(n_motion) >= 300:
                                continue
                            
                            # 生成唯一的新名称
                            import random
                            import string
                            new_name = random.choice(string.ascii_uppercase) + '_' + name
                            while new_name in name_set:
                                new_name = random.choice(string.ascii_uppercase) + '_' + name
                            name_set.add(new_name)
                            
                            motion_data_list.append({
                                'name': new_name,
                                'motion': n_motion,
                                'length': len(n_motion),
                                'text': [text_dict]
                            })
                        except Exception as e:
                            # 记录错误但继续处理其他行
                            continue
            
            # 添加原始motion数据（如果没有时间标签的文本）
            if flag:
                motion_data_list.insert(0, {
                    'name': name,
                    'motion': motion,
                    'length': len(motion),
                    'text': text_data
                })
            
            return motion_data_list, flag
            
        except Exception as e:
            # 可以在这里添加日志记录
            return None

    def _parallel_load_data(self, id_list, motion_dir, text_dir, min_motion_len):
        """使用线程池并行加载数据"""
        results = []
        
        num_threads = min(mp.cpu_count() * 2, len(id_list))  # I/O密集型，可以多用一些线程
        
        print(f"Using {num_threads} threads to load text-motion data...")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            future_to_name = {
                executor.submit(self._load_single_item, (name, motion_dir, text_dir, min_motion_len)): name 
                for name in id_list
            }
            
            # 使用tqdm显示进度
            from tqdm import tqdm
            for future in tqdm(as_completed(future_to_name), total=len(id_list), desc="Loading text-motion pairs"):
                result = future.result()
                results.append(result)
        
        return results

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print(f"Pointer Pointing at {self.pointer}, length {length}")
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def real_len(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data['caption']

        max_motion_length = self.opt.max_motion_length
        if m_length >= self.opt.max_motion_length:
            idx = random.randint(0, len(motion) - max_motion_length)
            motion = motion[idx: idx + max_motion_length]
        else:
            padding_len = max_motion_length - m_length
            D = motion.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            motion = np.concatenate((motion, padding_zeros), axis=0)

        assert len(motion) == max_motion_length
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if self.eval_mode:
            tokens = text_data['tokens']
            if len(tokens) < self.opt.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.opt.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length
        return caption, motion, m_length
