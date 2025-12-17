import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data._utils.collate import default_collate
import time

# import spacy
def custom_collate_fn(batch):
    try:
        batch.sort(key=lambda x: x[3], reverse=True)
        return default_collate(batch)
    except RuntimeError as e:
        print(f"Collate error: {e}")
        
        # 检查哪个样本的形状与其他样本不一致
        if len(batch) > 1:
            print("检查批次中各个样本的形状...")
            
            # 假设所有样本都是相同结构的数据（如图像、标签等组成的元组）
            num_items = len(batch[0])  # 每个样本包含的数据项数量
            
            for item_idx in range(num_items):
                print(f"\n检查第 {item_idx} 个数据项:")
                
                # 收集所有样本中该数据项的形状
                shapes = []
                for sample_idx, sample in enumerate(batch):
                    if hasattr(sample[item_idx], 'shape'):
                        shapes.append((sample_idx, sample[item_idx].shape))
                    else:
                        shapes.append((sample_idx, type(sample[item_idx])))
                
                print(f"各个样本的形状: {shapes}")
                
                # 检查形状是否一致
                if len(shapes) > 1 and all(hasattr(sample[item_idx], 'shape') for sample in batch):
                    first_shape = shapes[0][1]
                    inconsistent_indices = []
                    
                    for sample_idx, shape in shapes:
                        if shape != first_shape:
                            inconsistent_indices.append(sample_idx)
                    
                    if inconsistent_indices:
                        print(f"发现不一致的形状！")
                        print(f"样本 {0} 的形状: {first_shape}")
                        for idx in inconsistent_indices:
                            print(f"样本 {idx} 的形状: {shapes[idx][1]}")
        
        # 打印详细信息
        print("\n" + "* " * 30)
        print("批次详细信息:")
        for i, sample in enumerate(batch):
            print(f"\n样本 {i}:")
            for j, item in enumerate(sample):
                if hasattr(item, 'shape'):
                    print(f"  数据项 {j}: 形状 {item.shape}, 类型 {type(item)}")
                else:
                    print(f"  数据项 {j}: 值 {item}, 类型 {type(item)}")
        print("* " * 30)
        
        # 可以选择跳过这个批次或者退出
        # return None  # 跳过这个批次
        exit(0)

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 28 # 20 before
        self.pointer = 0
        min_motion_len = 60 if 't2m' in self.opt.dataset_name else 24 # 40 before

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 300):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
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
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 300):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

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

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx+self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


class Text2MotionDatasetDist(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 28
        self.pointer = 0
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
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joints_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joints_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joints_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

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

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

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

        assert m_length >= self.max_length, f"item {item}: m_length {m_length} < self.max_length {self.max_length}\n{self.length_arr[idx]}"
        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                    
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx+self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 28
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 60 if 't2m' in self.opt.dataset_name else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 300):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
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
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 300):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

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

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class Text2MotionDatasetV2Dist(data.Dataset):
    # parallel version.
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 28
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 60 if 't2m' in self.opt.dataset_name else 24

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        print (f"All {len(id_list)} samples.")
        
        # 使用多进程并行加载数据
        self.data_dict, self.name_list, self.length_list = self._parallel_load_data(
            id_list, opt.motion_dir, opt.text_dir, min_motion_len
        )

        # 排序
        self.name_list, self.length_list = zip(*sorted(zip(self.name_list, self.length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(self.length_list)
        self.reset_max_len(self.max_length)

    def _load_single_item(self, args):
        """加载单个数据项的函数"""
        name, motion_dir, text_dir, min_motion_len = args
        try:
            motion_path = pjoin(motion_dir, name + '.npy')
            text_path = pjoin(text_dir, name + '.txt')
            
            if not Path(motion_path).exists() or not Path(text_path).exists():
                return None

            motion = np.load(motion_path)
            if len(motion) < min_motion_len or len(motion) >= 300:
                return None

            text_data = []
            new_name_list = []
            length_list = []
            data_dict = {}
            flag = False
            name_set = set()
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
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in name_set:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            name_set.add(new_name)
                            
                            data_dict[new_name] = {
                                'motion': n_motion,
                                'length': len(n_motion),
                                'text': [text_dict]
                            }
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))
                        except Exception as e:
                            print(f"Error processing line: {line_split}, name: {name}")
                            continue

            if flag:
                data_dict[name] = {
                    'motion': motion,
                    'length': len(motion),
                    'text': text_data
                }
                new_name_list.append(name)
                length_list.append(len(motion))

            return data_dict, new_name_list, length_list

        except Exception as e:
            print(f"Error loading {name}: {e}")
            return None

    def _parallel_load_data(self, id_list, motion_dir, text_dir, min_motion_len):
        """使用线程池并行加载数据"""
        all_data_dict = {}
        all_name_list = []
        all_length_list = []
        
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
                if result is not None:
                    data_dict, name_list, length_list = result
                    all_data_dict.update(data_dict)
                    all_name_list.extend(name_list)
                    all_length_list.extend(length_list)
        
        return all_data_dict, all_name_list, all_length_list

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

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
        for idx, token in enumerate(tokens):
            try:
                _, _ = token.split('/')
            except:
                # print (f"item: {token}")
                pos_one_hots.append(pos_one_hots[-1])
                word_embeddings.append(word_embeddings[-1])
                continue
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


'''For use of training baseline'''
class Text2MotionDatasetBaseline(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 28
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 60 if 't2m' in self.opt.dataset_name else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 300):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
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
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 300):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

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

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == 'single':
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx: s_idx + m_length]
        tgt_motion = motion[s_idx: s_idx + self.max_length]

        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate([src_motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx+self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


import multiprocessing as mp
from pathlib import Path

class MotionDatasetV2Dist(data.Dataset):
    # parallel version.
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # 使用多进程并行加载数据
        self.data = self._parallel_load_data(id_list, opt.motion_dir)
        
        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def _load_single_motion(self, args):
        """加载单个运动数据的函数"""
        name, motion_dir, window_size = args
        try:
            motion_path = pjoin(motion_dir, name + '.npy')
            if not Path(motion_path).exists():
                return None
                
            motion = np.load(motion_path)
            if motion.shape[0] < window_size:
                return None
                
            return motion, motion.shape[0] - window_size
        except Exception as e:
            # 可以在这里添加日志记录
            return None

    def _parallel_load_data(self, id_list, motion_dir):
        """使用线程池并行加载数据"""
        data = []
        lengths = []
        
        num_threads = min(mp.cpu_count() * 2, len(id_list))  # I/O密集型，可以多用一些线程
        
        print(f"Using {num_threads} threads to load data...")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            future_to_name = {
                executor.submit(self._load_single_motion, (name, motion_dir, self.opt.window_size)): name 
                for name in id_list
            }
            
            # 使用tqdm显示进度
            from tqdm import tqdm
            for future in tqdm(as_completed(future_to_name), total=len(id_list), desc="Loading motions"):
                result = future.result()
                if result is not None:
                    motion, length = result
                    data.append(motion)
                    lengths.append(length)
        
        self.lengths = lengths
        return data
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx+self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion

class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
                self.data_dict.append({'caption':line.strip(), "tokens":tokens})

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))


    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

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

        return word_embeddings, pos_one_hots, caption, sent_len