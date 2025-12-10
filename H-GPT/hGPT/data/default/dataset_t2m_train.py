import os
import random
import json
import pickle
import spacy
import rich
import codecs as cs
from os.path import join as pjoin
from rich.progress import track

import numpy as np
from torch.utils.data import Dataset
    
class Text2MotionDatasetTrain(Dataset):
    def __init__(
        self,
        motion_token_path,
        text_path,
        cot_path,
        split_path,
        split,
        mean,
        std,
        unit_length,
        fps,
        task_path,
        std_text,
        debug=False,
        **kwargs,
    ):
        # init
        cot_path = cot_path if cot_path != '' else None
        split_file = pjoin(split_path, f'{split}.txt')
        
        self.mean = mean
        self.std = std        
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

        new_name_list = []
        data_dict = {}

        for i, name in enumerator:
            if len(new_name_list) > maxdata:
                break
            
            # load motion token
            try:
                m_token_list = np.load(pjoin(motion_token_path, f'{name}.npy'), allow_pickle=True)
            except Exception as e:
                print (f"Loading data error: {name}. Skipped.")
                continue

            if np.isnan(m_token_list).any() or m_token_list.shape[0] == 0:
                print (f"found nan or none: {name}. skiped.")
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
            with cs.open(pjoin(text_path, name + '.txt')) as f:
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
                        m_token_list_new = [
                            tokens[int(f_tag * fps / unit_length
                                        ):int(to_tag * fps /
                                                unit_length)]
                            for tokens in m_token_list
                            if int(f_tag * fps / unit_length) <
                            int(to_tag * fps / unit_length)
                        ]
                        if len(m_token_list_new) == 0:
                            print (f"new motion token list is []: {name}. skipped.")
                            continue

                        new_name = '%s_%f_%f' % (name, f_tag,
                                                    to_tag)
                        data_dict[new_name] = {
                            'm_token_list': m_token_list_new,
                            'text': [text_dict]
                        }
                        new_name_list.append(new_name)

            if flag:
                data_dict[name] = {
                    'm_token_list': m_token_list,
                    'text': text_data
                }
                new_name_list.append(name)
        
        self.data_dict = data_dict
        self.name_list = new_name_list

        # texts
        self.std_text = std_text
        self.nlp = spacy.load('en_core_web_sm')

        # tasks
        self.instructions = json.load(open(task_path, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])

    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, idx):
        data_idx = idx % len(self.name_list)
        task_idx = random.randint(0, len(self.tasks) - 1)
        
        name = self.name_list[data_idx]
        data = self.data_dict[name]
        m_token_list, text_list = data['m_token_list'], data['text']
        
        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data['caption']
        cot = text_data["cot"]
        
        if self.std_text:
            doc = self.nlp(caption)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN'
                        or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)
            caption = ' '.join(word_list)
        
        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]

        m_tokens_len = m_tokens.shape[0]
        task = self.tasks[task_idx]
        
        # name, motion, m_length, m_tokens, m_tokens_len, caption, sent_len, "_".join(tokens), word_embeddings, pos_one_hots, all_captions, cot, task
        return name, None, None, m_tokens, m_tokens_len, caption, None, None, None, None, all_captions, cot, task


def unit_test():
    test_dataset = Text2MotionDatasetTrain(
        motion_token_path = 'datasets/motionx/data/TOKENS',
        text_path = 'datasets/motionx/data/texts/semantic_labels',
        cot_path = 'datasets/motionx/data/texts/cot/v3',
        split_path = 'datasets/motionx/data/split',
        split = 'test',
        mean = 0.0,
        std = 1.0,
        unit_length = 4,
        fps = 30,
        task_path = './datasets/motionx/data/instructions/template_pretrain_orig.json',    
        std_text = True,
        debug=True,
    )
    print (len(test_dataset))
    print (test_dataset[10])
    
if __name__ == "__main__":
    unit_test()