import os 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

is_multimodality = args.is_multimodality
print (f"is_multimodality: {is_multimodality}")

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, args.batch_size, w_vectorizer,is_multimodality=is_multimodality)

if args.dataname == 'kit':
    vq_base_dir = './dataset/KIT-ML'
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
elif args.dataname == 't2m':
    vq_base_dir = './dataset/HumanML3D'
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
elif args.dataname == 'humanml3d_x':
    vq_base_dir = './dataset/HumanML3D-X'
    dataset_opt_path = 'checkpoints/humanml3d_x/Dev_X_30FPS_Dist_Comp_v6_KLD01_BS1024/opt.txt'
elif args.dataname == 'humanml3d_x_noise':
    vq_base_dir = './dataset/HumanML3D-X-Noise'
    dataset_opt_path = 'checkpoints/humanml3d_x/Dev_X_30FPS_Dist_Comp_v6_KLD01_BS1024/opt.txt'
elif args.dataname == 'humanml3d_x_rephrase':
    vq_base_dir = './dataset/HumanML3D-X-Rephrase'
    dataset_opt_path = 'checkpoints/humanml3d_x/Dev_X_30FPS_Dist_Comp_v6_KLD01_BS1024/opt.txt'
args.vq_dir = os.path.join(vq_base_dir, args.vq_name)

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'), args)
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root='./clip-ViT-B-32')  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()


fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
gt_fid_list = []

repeat_time = 3

for i in range(repeat_time):
    print (f"repeat_time: {i} ...")
    gt_fid, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, writer, logger = eval_trans.evaluation_transformer_test(args.out_dir, 
                                 val_loader, net, trans_encoder, logger, writer, i, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, 
                                 best_multi=0, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False, savegif=False, save=False, savenpy=(i==0), is_multimodality=is_multimodality)
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
    gt_fid_list.append(gt_fid)
    if is_multimodality:
        multi.append(multimodality)

print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)
print('gt_fid: ', sum(gt_fid_list)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
gt_fid_list = np.array(gt_fid_list)

if is_multimodality:
    print('multi: ', sum(multi)/repeat_time)
    multi = np.array(multi)

msg_final = f"gt-FID. {np.mean(gt_fid_list):.3f} ± {np.std(gt_fid_list)*1.96/np.sqrt(repeat_time):.3f}, FID. {np.mean(fid):.3f} ± {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f} ± {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}± {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f} ± {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f} ± {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f} ± {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"

if is_multimodality:
    msg_final += f" multi: {np.mean(multi):.3f} ± {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"

logger.info(msg_final)