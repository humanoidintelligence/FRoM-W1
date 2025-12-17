import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Manager
import numpy as np
import os
import sys
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
from datasets import get_dataset_motion_loader, get_motion_loader
from models import MotionTransformer
from utils.get_opt import get_opt
from utils.metrics import *
from datasets import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from utils import paramUtil
from utils.utils import *
from trainers import DDPMTrainer

from os.path import join as pjoin
import sys
import random
import os

def build_models(opt, dim_pose):
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder

torch.multiprocessing.set_sharing_strategy('file_system')


def evaluate_matching_score(motion_loaders, file, eval_wrapper):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file, eval_wrapper):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, eval_wrapper, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, file, eval_wrapper, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluate_replication(replication_id, device_id, opt_path, dataset_opt_path, 
                        batch_size, mm_num_samples, mm_num_repeats, results_queue, diversity_times, mm_num_times):
    """单个replication的评估函数，在指定GPU上运行"""
    # try:
    # 设置设备
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    
    # 加载数据和模型
    gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, batch_size, device)
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    opt = get_opt(opt_path, device)
    encoder = build_models(opt, opt.dim_pose)
    trainer = DDPMTrainer(opt, encoder)
    
    eval_motion_loaders = {
        'text2motion': lambda: get_motion_loader(
            opt,
            batch_size,
            trainer,
            gt_dataset,
            mm_num_samples,
            mm_num_repeats
        )
    }
    
    # 执行评估
    motion_loaders = {}
    mm_motion_loaders = {}
    motion_loaders['ground truth'] = gt_loader
    
    for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
        motion_loader, mm_motion_loader = motion_loader_getter()
        motion_loaders[motion_loader_name] = motion_loader
        mm_motion_loaders[motion_loader_name] = mm_motion_loader

    # 使用临时文件记录日志
    os.makedirs('./tmp', exist_ok=True)
    with open(f'./tmp/eval_rep_{replication_id}_gpu_{device_id}.log', 'w+') as f:
        print(f'==================== Replication {replication_id} on GPU {device_id} ====================', file=f, flush=True)
        print(f'Time: {datetime.now()}', file=f, flush=True)
        
        mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f, eval_wrapper)
        
        print(f'Time: {datetime.now()}', file=f, flush=True)
        fid_score_dict = evaluate_fid(gt_loader, acti_dict, f, eval_wrapper)
        
        print(f'Time: {datetime.now()}', file=f, flush=True)
        div_score_dict = evaluate_diversity(acti_dict, f, eval_wrapper, diversity_times)
        
        print(f'Time: {datetime.now()}', file=f, flush=True)
        mm_score_dict = evaluate_multimodality(mm_motion_loaders, f, eval_wrapper, mm_num_times)
        
        print(f'!!! DONE Replication {replication_id} on GPU {device_id} !!!', file=f, flush=True)

    # 将结果发送到主进程
    results = {
        'replication_id': replication_id,
        'Matching Score': mat_score_dict,
        'R_precision': R_precision_dict,
        'FID': fid_score_dict,
        'Diversity': div_score_dict,
        'MultiModality': mm_score_dict
    }
    results_queue.put(results)
        
        # except Exception as e:
        #     print(f"Error in replication {replication_id} on GPU {device_id}: {e}")
        #     # 发送错误信息
        #     results_queue.put({'replication_id': replication_id, 'error': str(e)})
        #     exit(0)

def parallel_evaluation(replication_times, num_gpus=8, processes_per_gpu=3):
    mm_num_samples = 100
    mm_num_repeats = 30
    mm_num_times = 10

    diversity_times = 300
    batch_size = 32
    opt_path = sys.argv[1]
    dataset_opt_path = opt_path
    
    """并行评估主函数"""
    # 设置multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # 创建进程池和结果队列
    processes = []
    results_queue = Queue()
    all_results = []
    
    total_processes = num_gpus * processes_per_gpu
    replications_per_process = (replication_times + total_processes - 1) // total_processes
    
    print(f"Starting parallel evaluation with {total_processes} processes")
    print(f"Replications per process: {replications_per_process}")
    
    # 启动进程
    for gpu_id in range(num_gpus):
        for process_id in range(processes_per_gpu):
            start_rep = (gpu_id * processes_per_gpu + process_id) * replications_per_process
            end_rep = min(start_rep + replications_per_process, replication_times)
            
            if start_rep >= replication_times:
                break
                
            for rep_id in range(start_rep, end_rep):
                p = Process(
                    target=evaluate_replication,
                    args=(
                        rep_id, gpu_id, opt_path, dataset_opt_path,
                        batch_size, mm_num_samples, mm_num_repeats,
                        results_queue, diversity_times, mm_num_times
                    )
                )
                processes.append(p)
                p.start()
    
    # 收集结果
    completed = 0
    while completed < len(processes):
        try:
            result = results_queue.get(timeout=7200)  # 2小时超时
            if 'error' in result:
                print(f"Replication {result['replication_id']} failed: {result['error']}")
            else:
                all_results.append(result)
            completed += 1
        except Exception as e:
            print(f"Error collecting results: {e}")
            break
    
    # 等待所有进程结束
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    
    # 合并结果
    return merge_results(all_results, replication_times)

def merge_results(all_results, replication_times):
    """合并所有进程的结果"""
    all_metrics = OrderedDict({
        'Matching Score': OrderedDict({}),
        'R_precision': OrderedDict({}),
        'FID': OrderedDict({}),
        'Diversity': OrderedDict({}),
        'MultiModality': OrderedDict({})
    })
    
    # 按replication_id排序
    all_results.sort(key=lambda x: x['replication_id'])
    
    for result in all_results:
        if 'error' in result:
            continue
            
        replication_id = result['replication_id']
        
        for metric_name in all_metrics.keys():
            metric_dict = result[metric_name]
            for key, item in metric_dict.items():
                if key not in all_metrics[metric_name]:
                    all_metrics[metric_name][key] = [item]
                else:
                    all_metrics[metric_name][key] += [item]
    
    return all_metrics

def evaluation(log_file):
    replication_times = 20
    
    # 主进程使用GPU 0
    device_id = 0
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)
    
    """修改后的主评估函数"""
    # 使用并行评估
    all_metrics = parallel_evaluation(replication_times, num_gpus=8, processes_per_gpu=3)
    
    # 写入最终结果
    with open(log_file, 'w+') as f:
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                if isinstance(mean, (np.float64, np.float32)):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_num = random.randint(1000, 9999)
    log_file = f'./t2m_evaluation_dist_{timestamp}_{random_num}.log'
    evaluation(log_file)