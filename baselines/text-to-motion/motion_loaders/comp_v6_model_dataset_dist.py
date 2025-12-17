import torch
from networks.modules import *
from networks.trainers import CompTrainerV6Dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from os.path import join as pjoin
from tqdm import tqdm
import torch.distributed as dist
import numpy as np


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


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator


class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats, rank=0, world_size=1):
        assert mm_num_samples < len(dataset)
        
        if dist.is_available() and dist.is_initialized():
            print(f'Rank {rank}: {opt.model_dir}')
        else:
            print(opt.model_dir)

        # 创建分布式数据加载器
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
            
        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=shuffle, sampler=sampler)
        
        # 只在主进程加载模型，然后广播到其他进程
        if rank == 0:
            text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
            trainer = CompTrainerV6Dist(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
            epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
            
            # 保存模型状态到CPU，然后广播
            model_states = {
                'text_enc': text_enc.state_dict(),
                'seq_pri': seq_pri.state_dict(), 
                'seq_dec': seq_dec.state_dict(),
                'att_layer': att_layer.state_dict(),
                'mov_enc': mov_enc.state_dict(),
                'mov_dec': mov_dec.state_dict(),
                'len_estimator': len_estimator.state_dict(),
                'trainer_epoch': epoch,
                'schedule_len': schedule_len
            }
        else:
            model_states = None
        
        # 广播模型状态到所有进程
        if world_size > 1:
            model_states = dist.broadcast(model_states, src=0)
            
            # 其他进程重建模型
            text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
            trainer = CompTrainerV6Dist(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
            
            # 加载模型状态
            text_enc.load_state_dict(model_states['text_enc'])
            seq_pri.load_state_dict(model_states['seq_pri'])
            seq_dec.load_state_dict(model_states['seq_dec'])
            att_layer.load_state_dict(model_states['att_layer'])
            mov_enc.load_state_dict(model_states['mov_enc'])
            mov_dec.load_state_dict(model_states['mov_dec'])
            len_estimator.load_state_dict(model_states['len_estimator'])
            epoch = model_states['trainer_epoch']
            schedule_len = model_states['schedule_len']
        
        if dist.is_available() and dist.is_initialized():
            print(f'Rank {rank}: Loading model: Epoch {epoch:03d} Schedule_len {schedule_len:03d}')
        else:
            print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
            
        trainer.eval_mode()
        trainer.to(opt.device)
        
        # 分布式选择多模态样本索引
        if world_size > 1:
            if rank == 0:
                mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
                mm_idxs = np.sort(mm_idxs)
            else:
                mm_idxs = np.zeros(mm_num_samples, dtype=np.int64)
            
            # 广播多模态索引到所有进程
            mm_idxs = torch.from_numpy(mm_idxs).to(opt.device)
            dist.broadcast(mm_idxs, src=0)
            mm_idxs = mm_idxs.cpu().numpy()
        else:
            mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
            mm_idxs = np.sort(mm_idxs)

        generated_motion = []
        mm_generated_motions = []
        min_mov_length = 15 if opt.dataset_name == 't2m' else 6

        with torch.no_grad():
            # 每个进程只处理自己分配到的数据
            for i, data in enumerate(dataloader):
                # 全局索引
                if world_size > 1:
                    global_idx = i * world_size + rank
                else:
                    global_idx = i
                    
                # 检查是否超出数据集范围
                if global_idx >= len(dataset):
                    break
                    
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                # 计算多模态样本的全局索引
                mm_num_now = len(mm_generated_motions)
                is_mm = False
                if mm_num_now < mm_num_samples:
                    # 找到当前多模态样本对应的全局索引
                    target_mm_idx = mm_idxs[mm_num_now]
                    if global_idx == target_mm_idx:
                        is_mm = True

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                          m_lens[0]//opt.unit_length, opt.dim_pose)
                    if t == 0:
                        sub_dict = {
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item(),
                            'cap_len': cap_lens[0].item(),
                            'caption': caption[0],
                            'tokens': tokens
                        }
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                        
                if is_mm:
                    mm_generated_motions.append({
                        'caption': caption[0],
                        'tokens': tokens,
                        'cap_len': cap_lens[0].item(),
                        'mm_motions': mm_motions
                    })

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        
        if dist.is_available() and dist.is_initialized():
            print(f'Rank {rank}: Generated {len(generated_motion)} motions, {len(mm_generated_motions)} multi-modal motions')


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']
        
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)