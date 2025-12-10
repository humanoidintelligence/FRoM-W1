import random
from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin
from rich.progress import track

import numpy as np
from torch.utils.data import Dataset

class Text2MotionDatasetToken(Dataset):
    def __init__(
        self,
        motion_feat_path,
        text_path,
        cot_path,
        split_path,
        split,
        mean,
        std,
        min_motion_length,
        max_motion_length,
        unit_length,
        fps,
        debug=False,
        **kwargs,
    ):
        # init
        cot_path = cot_path if cot_path != '' else None
        split_file = pjoin(split_path, f'{split}.txt')
        
        self.mean = mean
        self.std = std        
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.fps = fps
            
        # load id list
        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())
            print (f"Id list num: {len(self.id_list)}")
                
        if debug:
            enumerator = enumerate(self.id_list)
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading Dataset {split_file}",
                ))
            maxdata = 1e10
            subset = ''
            
        # load motion
        new_name_list = []
        length_list = []
        data_dict = {}
        for idx, name in enumerator:
            if len(new_name_list) >= maxdata:
                break
            
            try:
                motion = np.load(pjoin(motion_feat_path, name + '.npy'), allow_pickle=True)
            except Exception as e:
                print (f"Loading data error: {name}. Skipped.")
                continue
            
            if np.isnan(motion).any():
                print (f"Found nan value: {name}. skiped.")
                continue
            
            if (len(motion)) <  self.min_motion_length or (len(motion) >= self.max_motion_length):
                print (f"Length out of range: {name}. curr: {len(motion)} min: {self.min_motion_length} max: {self.max_motion_length}. skiped.")
                continue
            
            data_dict[name] = {'motion': motion,
                            'length': len(motion),
                            'name': name}
            new_name_list.append(name)
            length_list.append(len(motion))
            
        self.data_dict = data_dict
        self.name_list = new_name_list
    
    def __len__(self):
        return len(self.name_list)  
        
    def __getitem__(self, idx):
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, "_".join(tokens), word_embeddings, pos_one_hots, all_captions, cot, task
        return name, motion, m_length, None, None, None, None, None, None, None, None, None, None


def unit_test():
    test_dataset = Text2MotionDatasetToken(
        motion_feat_path = 'datasets/motionx/data/motion_data/vectors_623',
        text_path = 'datasets/motionx/data/texts/semantic_labels',
        cot_path = 'datasets/motionx/data/texts/cot/v3',
        split_path = 'datasets/motionx/data/split',
        split = 'test',
        mean = 0.0,
        std = 1.0,
        min_motion_length = 40,
        max_motion_length = 400,
        unit_length = 4,
        fps = 30,
        debug=True,
    )
    print (len(test_dataset))
    print (test_dataset[10])
    
if __name__ == "__main__":
    unit_test()