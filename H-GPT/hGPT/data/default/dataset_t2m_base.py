import os
import rich
import random
import pickle
import codecs as cs
from os.path import join as pjoin
from rich.progress import track

import numpy as np
from torch.utils.data import Dataset

class Text2MotionDatasetBase(Dataset):
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
        
        # load motion, text, cot
        data_dict = {}
        new_name_list = []
        length_list = []
        
        for idx, name in enumerator:
            if len(new_name_list) >= maxdata:
                break
            
            # load motion
            try:
                motion = np.load(pjoin(motion_feat_path, name + ".npy"), allow_pickle=True)
            except Exception as e:
                print (f"Loading data error: {name}. Skipped.")
                continue
            
            if np.isnan(motion).any():
                print (f"Found nan value: {name}. skiped.")
                continue
            
            if (len(motion)) < self.min_motion_length or (len(motion)
                                                            >= self.max_motion_length):
                print (f"Length out of range: {name}. curr: {len(motion)} min: {self.min_motion_length} max: {self.max_motion_length}. skiped.")
                continue
            
            # load cot
            cot_list = []
            if cot_path != None:
                cot_file = pjoin(cot_path, name + '.txt')
                if not os.path.exists(cot_file):
                    print (f"No such a cot file: {name}. skipped.")
                    continue
                else:
                    with open(cot_file, 'r') as f:
                        cot_list = f.readlines()
                        
            # load text
            text_data = []
            flag = False
            with cs.open(pjoin(text_path, name + '.txt'), "r") as f:
                lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    text_dict = {}
                    line_split = line.split('#')
                    try:
                        caption = line_split[0]
                        t_tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        assert f_tag <= to_tag, "f_tag > t_tag"
                    except:
                        print (f"Error load text: {name}. skipped.")
                        continue
                    
                    text_dict['caption'] = caption
                    text_dict['tokens'] = t_tokens
                    
                    # TODO: fix len(cot_list) != len(lines)
                    if cot_path == None:
                        text_dict['cot'] = ''
                    elif len(cot_list) != len(lines):
                        print (f"cot != text lines: {name}. {len(cot_list)} vs {len(lines)}")
                        continue
                    else:
                        text_dict['cot'] = cot_list[line_idx]
                        
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        motion_new = motion[int(f_tag *
                                                self.fps):int(to_tag * self.fps)]
                        if (len(motion_new)
                            ) < self.min_motion_length or (
                                len(motion_new) >= self.min_motion_length):
                            continue
                        
                        new_name = random.choice(
                            'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        while new_name in new_name_list:
                            new_name = random.choice(
                                'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        name_count = 1
                        while new_name in data_dict:
                            new_name += '_' + name_count
                            name_count += 1
                            
                        data_dict[new_name] = {
                            'motion': motion_new,
                            "length": len(motion_new),
                            'text': [text_dict]
                        }
                        new_name_list.append(new_name)
                        length_list.append(len(motion_new))

            if flag:
                data_dict[name] = {
                    'motion': motion,
                    "length": len(motion),
                    'text': text_data,
                }
                new_name_list.append(name)
                length_list.append(len(motion))
            
        # sort data
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.data_dict = data_dict
        self.name_list = name_list
        self.length_list = length_list
        
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data["caption"]
        tokens = text_data["tokens"]
        cot = text_data["cot"]
        
        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        # Crop the motions in to times of unit_length, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
            
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # Z Normalization
        motion = (motion - self.mean) / self.std
        
        # return name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, "_".join(tokens), word_embeddings, pos_one_hots, all_captions, cot, task
        return name, motion, m_length, None, None, caption, None, "_".join(tokens), None, None, all_captions, cot, None

def unit_test():
    test_dataset = Text2MotionDatasetBase(
        motion_feat_path = 'datasets/motionx/data/motion_data/vectors_623',
        text_path = 'datasets/motionx/data/texts/semantic_labels',
        cot_path = 'datasets/motionx/data/texts/cot/v3',
        split_path = 'datasets/motionx/data/split',
        split = 'train',
        mean = 0.0,
        std = 1.0,
        min_motion_length = 40,
        max_motion_length = 400,
        unit_length = 4,
        fps = 30,
        debug=False,
    )
    print (len(test_dataset))
    print (test_dataset[10])
    
if __name__ == "__main__":
    unit_test()