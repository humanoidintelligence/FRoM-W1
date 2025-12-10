import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.append('./demo_humanoids')

# import math
import imageio
import gradio as gr
import random
import torch
import time
# import cv2
import imageio
import numpy as np
import torch
import pytorch_lightning as pl
# import moviepy as mp
# from moviepy import ImageSequenceClip
from pathlib import Path
from scipy.spatial.transform import Rotation as RRR
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
# import pickle
# import subprocess

from hGPT.data.build_data import build_data
from hGPT.models.build_model import build_model
from hGPT.config import parse_args
import hGPT.render.matplot.plot_3d_global as plot_3d
from hGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from hGPT.render.pyrender.smpl_render import SMPLRender
from hGPT.data.motionx.visualization.plot_3d_global import draw_to_batch_smplh

# from demo_humanoids.motiongpt import motion_token_to_string
# from demo_humanoids.gr1t1_retarget import retarget_gr1t1

# from demo_humanoids.gr1t1_sim import gr1t1_sim
# from demo_humanoids.h1_sim import h1_sim
from demo_humanoids.sim_all import SimParallel
from demo_humanoids.retarget_by_grad.grad_fit_gr1t1 import retarget_by_grad_gr1t1
from demo_humanoids.retarget_by_grad.grad_fit_h1 import retarget_by_grad_h1

# from demo_humanoids.gr1t1_robot import gr1_robot_action

from demo_humanoids.gr1t1_robot_remote import gr1_robot_action, gr1_robot_switch
from demo_humanoids.h1_robot_remote import h1_robot_action, h1_robot_switch

from demo_humanoids.h1_motion_stream import h1_add_motion

# import threading
from concurrent.futures import ThreadPoolExecutor

from accelerate import dispatch_model

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [
            0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [
    20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]

t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [
    21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]

t2m_body_hand_kinematic_chain = t2m_kinematic_chain + \
    t2m_left_hand_chain + t2m_right_hand_chain

sim = SimParallel()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
# set_random_seed(1234)
set_random_seed(42)

# work_dir = "/home/fudan/Workspace/yfgao/HumanoidGPT"



def create_device_map():
    device_map = {
        'metrics': torch.device('cpu'),
        '_losses': torch.device('cpu'),
        'vae': 0,
        'lm.language_model.base_model.model.model.embed_tokens': 0
    }
    # layer_ids = list(range(lm_layers))
    # layer_ids = [layer_ids[i: i+layer_per_gpu] for i in range(0, lm_layers, layer_per_gpu)]
    layer_ids = [
        list(range(13)),
        list(range(13, 32))
    ]
    for group_id, id_group in enumerate(layer_ids):
        for i in id_group:
            device_map[f'lm.language_model.base_model.model.model.layers.{i}'] = group_id
    device_map['lm.language_model.base_model.model.model.norm'] = 1
    device_map['lm.language_model.base_model.model.model.rotary_emb'] = 1
    device_map['lm.language_model.base_model.model.lm_head'] = 0
    return device_map

# os.environ['DISPLAY'] = ':0.0'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Load model
# cfg = parse_args(phase="webui")  # parse config file
cfg = parse_args(phase="demo")
cfg.FOLDER = 'demo_cases_2'
output_dir = Path(cfg.FOLDER)
output_dir.mkdir(parents=True, exist_ok=True)
pl.seed_everything(cfg.SEED_VALUE)

# output_dir_abs = Path(cfg.FOLDER).absolute()


cfg.ACCELERATOR = "gpu"
if cfg.ACCELERATOR == "gpu":
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")


datamodule = build_data(cfg, phase="test")
model = build_model(cfg, datamodule)
state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu", weights_only=False)
if 'state_dict' in state_dict.keys():
    state_dict = state_dict['state_dict']
elif 'module' in state_dict.keys():
    state_dict = state_dict['module']

model.load_state_dict(state_dict, strict=True)
# model.to(device)
device_map = create_device_map()
model = dispatch_model(model, device_map=device_map)

# Whisper comment out

audio_processor = WhisperProcessor.from_pretrained(cfg.model.whisper_path)
audio_model = WhisperForConditionalGeneration.from_pretrained(cfg.model.whisper_path).to(device)
# forced_decoder_ids = audio_processor.get_decoder_prompt_ids(language="zh", task="translate")
forced_decoder_ids_zh = audio_processor.get_decoder_prompt_ids(language="zh", task="translate")
forced_decoder_ids_en = audio_processor.get_decoder_prompt_ids(language="en", task="translate")




# HTML Style
Video_Components = """
<div class="side-video" style="position: relative;">
    <video width="340" autoplay loop>
        <source src="file/{video_path}" type="video/mp4">
    </video>
    <a class="videodl-button" href="file/{video_path}" download="{video_fname}" title="Download Video">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>
    </a>
    <a class="npydl-button" href="file/{motion_path}" download="{motion_fname}" title="Download Motion">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-file-box"><path d="M14.5 22H18a2 2 0 0 0 2-2V7.5L14.5 2H6a2 2 0 0 0-2 2v4"/><polyline points="14 2 14 8 20 8"/><path d="M2.97 13.12c-.6.36-.97 1.02-.97 1.74v3.28c0 .72.37 1.38.97 1.74l3 1.83c.63.39 1.43.39 2.06 0l3-1.83c.6-.36.97-1.02.97-1.74v-3.28c0-.72-.37-1.38-.97-1.74l-3-1.83a1.97 1.97 0 0 0-2.06 0l-3 1.83Z"/><path d="m7 17-4.74-2.85"/><path d="m7 17 4.74-2.85"/><path d="M7 17v5"/></svg>
    </a>
</div>
"""

Video_Components_example = """
<div class="side-video" style="position: relative;">
    <video width="340" autoplay loop controls>
        <source src="file/{video_path}" type="video/mp4">
    </video>
    <a class="npydl-button" href="file/{video_path}" download="{video_fname}" title="Download Video">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>
    </a>
</div>
"""

Text_Components = """
<h3 class="side-content" >{msg}</h3>
"""

gif_components = """
<div class="side-gif" style="position: relative;">
    <img src="{gif_path}" alt="{gif_fname}" width="340" class="looping-gif"/>
    <a class="videodl-button" href="{gif_path}" download="{gif_fname}" title="Download Motion">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>
    </a>
</div>
"""


def motion_token_to_string(motion_token, lengths, codebook_size=512):
    motion_string = []
    for i in range(motion_token.shape[0]):
        motion_i = motion_token[i].cpu(
        ) if motion_token.device.type == 'cuda' else motion_token[i]
        motion_list = motion_i.tolist()[:lengths[i]]
        motion_string.append(
            (f'<motion_id_{codebook_size}>' +
             ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
             f'<motion_id_{codebook_size + 1}>'))
    return motion_string


def render_motion(data, feats, method='fast'):
    fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    # np.save(output_npy_path, feats)
    np.save(output_npy_path, data)

    # print (" WARANING !!!!!!!!!!!!!!!!!!!! ")
    # path = '/home/fudan/Workspace/MotionGPT/demo_humanoids/test_mdm.npy'
    # data = np.load(path)

    # print (path)
    # print (" WARANING !!!!!!!!!!!!!!!!!!!! ")
    # print (data)
    # print (data.shape)
    # exit(0)


    if method == 'slow':
        if len(data.shape) == 4:
            data = data[0]
        data = data - data[0, 0]

        pose_generator = HybrIKJointsToRotmat()
        pose = pose_generator(data)
        pose = np.concatenate([
            pose,
            np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
        ], 1)
        shape = [768, 768]
        # cfg.RENDER.SMPL_MODEL_PATH = "/home/fudan/Workspace/yfgao/HumanoidGPT/deps/smpl_models"
        # render = SMPLRender(cfg.RENDER.SMPL_MODEL_PATH)
        render = SMPLRender("/home/fudan/Workspace/yfgao/HumanoidGPT/deps/smpl_models/smpl")

        r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
        pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
        vid = []
        aroot = data[[0], 0]
        aroot[:, 1] = -aroot[:, 1]
        params = dict(pred_shape=np.zeros([1, 10]),
                      pred_root=aroot,
                      pred_pose=pose)
        render.init_renderer([shape[0], shape[1], 3], params)
        for i in range(data.shape[0]):
            renderImg = render.render(i)
            vid.append(renderImg)

        out = np.stack(vid, axis=0) # shape: (192, 768, 768, 4)
        print ("motion shape: ", out.shape)
        imageio.mimwrite(output_mp4_path, out, fps=30)
        print ("output mp4 path: ", output_mp4_path)

        # output_gif_path = output_mp4_path[:-4] + '.gif'
        # imageio.mimwrite(output_gif_path, out, duration=50)
        # print ("output gif path: ", output_gif_path)
        # out_video = mp.VideoFileClip(output_gif_path)
        # out_video.write_videofile(output_mp4_path, fps=30)
        del out, render

        pose = RRR.from_matrix(pose.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 72)

        # Motion Retargeting
        ## Pipeline Version
        # print ("retargeting gr1t1 ...")
        # gr1t1_motion = retarget_by_grad_gr1t1(pose) # SMPL Pose (T, 24, 3, 3)
        # print ("retargeting h1 ...")
        # h1_motion = retarget_by_grad_h1(pose) # SMPL Pose (T, 24, 3, 3)
        # print ("Good Job!")
        ## Motion Retargeting

        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(retarget_by_grad_gr1t1, pose)
            future2 = executor.submit(retarget_by_grad_h1, pose)
            gr1t1_motion = future1.result()
            h1_motion = future2.result()

        feats_gr1t1_fname = fname + '_gr1t1.npy'
        output_npy_gr1t1_path = os.path.join(output_dir, feats_gr1t1_fname)
        np.save(output_npy_gr1t1_path, gr1t1_motion) # (T, 32)
        # print ("save to demo_gr1t1/test_gr1t1_motion.npy")

        feats_h1_fname = fname + '_h1.npy'
        output_npy_h1_path = os.path.join(output_dir, feats_h1_fname)
        np.save(output_npy_h1_path, h1_motion) # (T, 19)
        # print ("save to demo_gr1t1/test_gr1t1_motion.npy")

        # gr1t1_sim(gr1t1_motion)
        # h1_sim(h1_motion)

        #########################################################
        #                      SII DEMO                         #
        #########################################################

        FPS_PLAY = 20
        DELAY_S = 1
        DELAY_T = FPS_PLAY * DELAY_S
        len_motion = gr1t1_motion.shape[0]

        # h1_add_motion(data)

        h1_robot_switch(True)
        gr1_robot_switch(True)
        for t in range(len_motion + DELAY_T):
            if t >= len_motion:
                sim.update(gr1t1_motion[len_motion - 1], 0)
                sim.update(h1_motion[len_motion - 1], 1)
            else:
                sim.update(gr1t1_motion[t], 0)
                sim.update(h1_motion[t], 1)

            if t < DELAY_T:
                gr1_robot_action(gr1t1_motion[0], FPS_PLAY)
                h1_robot_action(h1_motion[0], FPS_PLAY)
            else:
                gr1_robot_action(gr1t1_motion[t - DELAY_T], FPS_PLAY)
                h1_robot_action(h1_motion[t - DELAY_T], FPS_PLAY)

            time.sleep(1 / FPS_PLAY)

        time.sleep(1)
        
        h1_robot_switch(False)
        gr1_robot_switch(False)


        #########################################################
        #                      SII DEMO                         #
        #########################################################

        # Debug Motion Generation
        # # 创建两个线程
        # thread1 = threading.Thread(target=gr1_robot_action, args=(gr1t1_motion,))
        # thread2 = threading.Thread(target=h1_robot_action, args=(h1_motion,))

        # # 启动线程
        # thread1.start()
        # thread2.start()

        # # 等待线程执行完成
        # thread1.join()
        # thread2.join()

            # # # dict_keys(['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'smpl_joints', 'fps'])
            # pose_aa = RRR.from_matrix(pose.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 24, 3)
            # # # 当前全局方向的旋转向量
            # global_orient = pose_aa[:, 0]  # shape: (N, 3)
            # # # 定义绕X轴旋转180度的旋转矩阵
            # rotation_180_x = RRR.from_euler('x', 180, degrees=True)
            # # # 将当前全局方向的旋转应用该变换
            # rotated_global_orient = RRR.from_rotvec(global_orient) * rotation_180_x
            # # # 提取更新后的旋转向量
            # pose_aa[:, 0] = rotated_global_orient.as_rotvec()


            # tmp_output = {}
            # tmp_output['humanoidgpt'] = {
            #     'root_trans_offset': torch.Tensor([0,0,1.05] * pose.shape[0]).reshape(-1,3).to(torch.device("cuda", index=0)), # ,#  0,0,0.97860012
            #     'pose_aa': pose_aa[:,:22,:], # pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
            #     'dof': h1_motion,
            #     'root_rot': gt_root_rot, # torch.Tensor([0,0,0,1] * pose.shape[0]).reshape(-1,4),
            #     'smpl_joints': joints,
            #     'fps': 30
            # }
            # with open('/home/fudan/Workspace/sxli/human2humanoid/legged_gym/resources/motions/h1/humanoidgpt.pkl', 'wb') as fout:
            #     pickle.dump(tmp_output, fout)


            # # 切换到目标文件夹
            # os.chdir('/home/fudan/Workspace/sxli/human2humanoid/legged_gym')
            # # 设置新的 Conda 环境并调用命令
            # conda_env_name = "omnih2o"  # 替换为你的 Conda 环境名称
            # command_to_run = "python legged_gym/scripts/play_hydra.py --config-name=config_teleop task=h1:teleop env.num_observations=1665 env.num_privileged_obs=1742 motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel motion.teleop_selected_keypoints_names=[] motion.extend_head=True num_envs=1 asset.zero_out_far=False asset.termination_scales.max_ref_motion_distance=10.0 sim_device=cuda:0 load_run=student_amass_phc_filtered_distill checkpoint=16000 env.add_short_history=True env.short_history_length=25 headless=False motion.motion_file=resources/motions/h1/humanoidgpt.pkl"  # 替换为你的目标命令
            # try:
            #     # 激活 Conda 环境并运行命令
            #     result = subprocess.run(
            #         f"conda init && conda activate {conda_env_name} && {command_to_run}",
            #         shell=True,
            #         check=True,
            #         text=True,
            #         executable="/bin/bash"  # 确保在 Bash 环境下运行
            #     )
            #     print("命令输出：", result.stdout)
            # except subprocess.CalledProcessError as e:
            #     print("运行命令时出错：", e)

            # h1_robot_action(h1_motion)

        # else:
        #     raise NotImplementedError

    elif method == 'fast':
        # output_gif_path = output_mp4_path[:-4] + '.gif'
        if len(data.shape) == 3:
            data = data[None]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # out = plot_3d.draw_to_batch(data, [''], [output_gif_path])
        out = plot_3d.draw_to_batch(data, [''], None) # torch.Size([1, 200, 960, 960, 4])
        out = np.array(out[0])
        # print (out)
        # print (out.shape)
        # # shape: (192, 768, 768, 4)
        imageio.mimwrite(output_mp4_path, out, fps=30)
        print ("output mp4 path: ", output_mp4_path)

        # out_video = mp.VideoFileClip(output_gif_path)
        # out_video.write_videofile(output_mp4_path)
        # del pose_vis

    return output_mp4_path, video_fname, output_npy_path, feats_fname

def render_motion_plt(feats, joints):

    fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        time.time())) + str(np.random.randint(10000, 99999))
    gif_fname = fname + '.gif'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_gif_path = os.path.join(output_dir, gif_fname)
    np.save(output_npy_path, feats)

    if len(joints.shape) == 3:
        joints = joints[None]
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()

    _ = draw_to_batch_smplh(joints, t2m_body_hand_kinematic_chain, title_batch=None, outname=[output_gif_path])
    return output_gif_path, gif_fname, output_npy_path, feats_fname


def load_motion(motion_uploaded, method):
    file = motion_uploaded['file']

    feats = torch.tensor(np.load(file), device=model.device)
    if len(feats.shape) == 2:
        feats = feats[None]
    # feats = model.datamodule.normalize(feats)

    # Motion tokens
    motion_lengths = feats.shape[0]
    motion_token, _ = model.vae.encode(feats)

    motion_token_string = model.lm.motion_token_to_string(
        motion_token, [motion_token.shape[1]])[0]
    motion_token_length = motion_token.shape[1]

    # Motion rendered
    joints = model.datamodule.feats2joints(feats.cpu()).cpu().numpy()
    output_mp4_path, video_fname, output_npy_path, joints_fname = render_motion(
        joints,
        feats.to('cpu').numpy(), method)
    
    # abs_video_path = os.path.join(work_dir, output_mp4_path)

    motion_uploaded.update({
        "feats": feats,
        "joints": joints,
        "motion_video": output_mp4_path,
        "motion_video_fname": video_fname,
        "motion_joints": output_npy_path,
        "motion_joints_fname": joints_fname,
        "motion_lengths": motion_lengths,
        "motion_token": motion_token,
        "motion_token_string": motion_token_string,
        "motion_token_length": motion_token_length,
    })

    return motion_uploaded


def add_text(history, text, motion_uploaded, data_stored, method="slow"):
    data_stored = data_stored + [{'user_input': text}]
    # print(motion_uploaded)
    text = f"""<h3>{text}</h3>"""
    history = history + [(text, None)]
    if 'file' in motion_uploaded.keys():
        motion_uploaded = load_motion(motion_uploaded, method)
        output_mp4_path = motion_uploaded['motion_video']
        video_fname = motion_uploaded['motion_video_fname']
        output_npy_path = motion_uploaded['motion_joints']
        joints_fname = motion_uploaded['motion_joints_fname']
        history = history + [(Video_Components.format(
            video_path=output_mp4_path,
            video_fname=video_fname,
            motion_path=output_npy_path,
            motion_fname=joints_fname), None)]

    return history, gr.update(value="",
                              interactive=False), motion_uploaded, data_stored

# Whisper comment out

def add_audio(history, audio_path, data_stored, language='en'):
    audio, sampling_rate = librosa.load(audio_path, sr=16000)
    input_features = audio_processor(
        audio, sampling_rate, return_tensors="pt"
    ).input_features  # whisper training sampling rate, do not modify
    input_features = torch.Tensor(input_features).to(device)

    if language == 'en':
        forced_decoder_ids = forced_decoder_ids_en
    else:
        forced_decoder_ids = forced_decoder_ids_zh
    predicted_ids = audio_model.generate(input_features,
                                         forced_decoder_ids=forced_decoder_ids)
    text_input = audio_processor.batch_decode(predicted_ids,
                                              skip_special_tokens=True)
    text_input = str(text_input).strip('[]"')
    data_stored = data_stored + [{'user_input': text_input}]
    gr.update(value=data_stored, interactive=False)
    history = history + [(text_input, None)]

    return history, data_stored


def add_file(history, file, txt, motion_uploaded):
    motion_uploaded['file'] = file.name
    txt = txt.replace(" <Motion_Placeholder>", "") + " <Motion_Placeholder>"
    return history, gr.update(value=txt, interactive=True), motion_uploaded


def bot(history, motion_uploaded, data_stored):

    # motion_length, motion_token_string = motion_uploaded[
    #     "motion_lengths"], motion_uploaded["motion_token_string"]

    input = data_stored[-1]['user_input']
    # prompt = model.lm.placeholder_fulfill(input, motion_length,
    #                                       motion_token_string, text="", cot="")
    data_stored[-1]['model_input'] = input
    # batch = {
    #     "length": [motion_length],
    #     "text": [prompt],
    # }
    text_batch = [input]

    batch = {
                # "name": text_batch,
                # "motion": torch.zeros(1, 200, 623),
                # "m_length": [200],
                # return_dict["motion_lengths"][b * batch_size:(b + 1) *
                #                               batch_size],
                "text": text_batch,
                # "cot": text_batch,
            }
            # print (batch["m_length"])
    # print("Prompt:", prompt)
    # print("Input", input)
    
    print ("* " * 20)
    print (batch)
    print ("* " * 20)

    # outputs = model(batch, task="t2m")
    # outputs = model.val_t2m_forward(batch)
    outputs = model.inference(batch)
    # out_feats = outputs["feats"][0]
    # out_lengths = outputs["length"][0]
    # out_joints = outputs["joints"][:out_lengths].detach().cpu().numpy()
    # out_texts = outputs["texts"][0]
    # print(out_texts)

    out_joints = outputs["joints_rst"] # [batch_size, length, joint_num, 3])
    out_lengths = outputs["length"]
    out_texts = outputs["cot_rst"]
    out_feats = outputs["m_rst"]

    # print(out_texts)
    # output_mp4_path, video_fname, output_npy_path, joints_fname = render_motion(
    #     out_joints,
    #     out_feats.to('cpu').numpy(), method)

    output_gif_path, gif_fname, output_npy_path, joints_fname = render_motion_plt(
        out_feats.to('cpu').numpy(),
        out_joints.to('cpu').numpy()
    )


    motion_uploaded = {
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
    }

    data_stored[-1]['model_output'] = {
        "feats": out_feats,
        "joints": out_joints,
        "length": out_lengths,
        "texts": out_texts,
        "motion_gif": output_gif_path,
        "motion_gif_fname": gif_fname,
        "motion_joints": output_npy_path,
        "motion_joints_fname": joints_fname,
        # "motion_video": output_mp4_path,
        # "motion_video_fname": video_fname,
    }

    if '<Motion_Placeholder>' == out_texts[0]:
        # response = [
        #     Video_Components.format(video_path=output_mp4_path,
        #                             video_fname=video_fname,
        #                             motion_path=output_npy_path,
        #                             motion_fname=joints_fname)
        # ]
        response = [
            gif_components.format(gif_path=output_gif_path,
                                 gif_fname=gif_fname)
        ]
    elif '<Motion_Placeholder>' in out_texts[0]:
        # print("Here")
        # response = [
        #     # Text_Components.format(
        #     #     msg=out_texts.split("<Motion_Placeholder>")[0]),
        #     Video_Components.format(video_path=output_mp4_path,
        #                             video_fname=video_fname,
        #                             motion_path=output_npy_path,
        #                             motion_fname=joints_fname),
        #     # Text_Components.format(
        #     #     msg=out_texts.split("<Motion_Placeholder>")[1]),
        # ]
        # response = [
        #     Text_Components.format(
        #         msg=out_texts[0].split("<soc>")[1].split('<eoc>')[0]),
        #     gif_components.format(gif_path=output_gif_path,
        #                          gif_fname=gif_fname)
        # ]
        text_part = out_texts[0].split("<soc>")[1].split('<eoc>')[0]
        # response = [
        #     text_part,
        #     gr.Image(value=output_gif_path)
        # ]
        # response = f"{text_part}\n\n![Generated Motion](file/{output_gif_path})"
        response = [f"{text_part}\n\n", f"<img src='file/{output_gif_path}' width='340' class=\"looping-gif\"/>"]
    else:
        response = f"""<h3>{out_texts[0]}</h3>"""

    history[-1][1] = ""
    # print(response)

    # for r in response:
    #     history += [(r, None)]
    #     yield history, motion_uploaded, data_stored

        # print(history)
    for character in response[0]:
        history[-1][1] += character
        time.sleep(0.02)
        yield history, motion_uploaded, data_stored
    history[-1][1] += response[1]
    yield history, motion_uploaded, data_stored

def bot_example(history, responses):
    history = history + responses
    return history


with open("assets/css/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS) as demo:

    # # Examples
    # chat_instruct = gr.State([
    #     (None,
    #      "👋 Hi, I'm MotionGPT! I can generate realistic human motion from text, or generate text from motion."
    #      ),
    #     (None,
    #      "💡 You can chat with me in pure text like generating human motion following your descriptions."
    #      ),
    #     (None,
    #      "💡 After generation, you can click the button in the top right of generation human motion result to download the human motion video or feature stored in .npy format."
    #      ),
    #     (None,
    #      "💡 With the human motion feature file downloaded or got from dataset, you are able to ask me to translate it!"
    #      ),
    #     (None,
    #      "💡 Of courser, you can also purely chat with me and let me give you human motion in text, here are some examples!"
    #      ),
    #     (None,
    #      "💡 We provide two motion visulization methods. The default fast method is skeleton line ploting which is like the examples below:"
    #      ),
    #     (None,
    #      Video_Components_example.format(
    #          video_path="assets/videos/example0_fast.mp4",
    #          video_fname="example0_fast.mp4")),
    #     (None,
    #      "💡 And the slow method is SMPL model rendering which is more realistic but slower."
    #      ),
    #     (None,
    #      Video_Components_example.format(
    #          video_path="assets/videos/example0.mp4",
    #          video_fname="example0.mp4")),
    #     (None,
    #      "💡 If you want to get the video in our paper and website like below, you can refer to the scirpt in our [github repo](https://github.com/OpenMotionLab/MotionGPT#-visualization)."
    #      ),
    #     (None,
    #      Video_Components_example.format(
    #          video_path="assets/videos/example0_blender.mp4",
    #          video_fname="example0_blender.mp4")),
    #     (None, "👉 Follow the examples and try yourself!"),
    # ])
    # chat_instruct_sum = gr.State([(None, '''
    #      👋 Hi, I'm MotionGPT! I can generate realistic human motion from text, or generate text from motion.
         
    #      1. You can chat with me in pure text like generating human motion following your descriptions.
    #      2. After generation, you can click the button in the top right of generation human motion result to download the human motion video or feature stored in .npy format.
    #      3. With the human motion feature file downloaded or got from dataset, you are able to ask me to translate it!
    #      4. Of course, you can also purely chat with me and let me give you human motion in text, here are some examples!
    #      ''')] + chat_instruct.value[-7:])

    # t2m_examples = gr.State([
    #     (None,
    #      "💡 You can chat with me in pure text, following are some examples of text-to-motion generation!"
    #      ),
    #     ("A person is walking forwards, but stumbles and steps back, then carries on forward.",
    #      Video_Components_example.format(
    #          video_path="assets/videos/example0.mp4",
    #          video_fname="example0.mp4")),
    #     ("Generate a man aggressively kicks an object to the left using his right foot.",
    #      Video_Components_example.format(
    #          video_path="assets/videos/example1.mp4",
    #          video_fname="example1.mp4")),
    #     ("Generate a person lowers their arms, gets onto all fours, and crawls.",
    #      Video_Components_example.format(
    #          video_path="assets/videos/example2.mp4",
    #          video_fname="example2.mp4")),
    #     ("Show me the video of a person bends over and picks things up with both hands individually, then walks forward.",
    #      Video_Components_example.format(
    #          video_path="assets/videos/example3.mp4",
    #          video_fname="example3.mp4")),
    #     ("Imagine a person is practing balancing on one leg.",
    #      Video_Components_example.format(
    #          video_path="assets/videos/example5.mp4",
    #          video_fname="example5.mp4")),
    #     ("Show me a person walks forward, stops, turns directly to their right, then walks forward again.",
    #      Video_Components_example.format(
    #          video_path="assets/videos/example6.mp4",
    #          video_fname="example6.mp4")),
    #     ("I saw a person sits on the ledge of something then gets off and walks away.",
    #      Video_Components_example.format(
    #          video_path="assets/videos/example7.mp4",
    #          video_fname="example7.mp4")),
    #     ("Show me a person is crouched down and walking around sneakily.",
    #      Video_Components_example.format(
    #          video_path="assets/videos/example8.mp4",
    #          video_fname="example8.mp4")),
    # ])

    # m2t_examples = gr.State([
    #     (None,
    #      "💡 With the human motion feature file downloaded or got from dataset, you are able to ask me to translate it, here are some examples!"
    #      ),
    #     ("Please explain the movement shown in <Motion_Placeholder> using natural language.",
    #      None),
    #     (Video_Components_example.format(
    #         video_path="assets/videos/example0.mp4",
    #         video_fname="example0.mp4"),
    #      "The person was pushed but didn't fall down"),
    #     ("What kind of action is being represented in <Motion_Placeholder>? Explain it in text.",
    #      None),
    #     (Video_Components_example.format(
    #         video_path="assets/videos/example4.mp4",
    #         video_fname="example4.mp4"),
    #      "The figure has its hands curled at jaw level, steps onto its left foot and raises right leg with bent knee to kick forward and return to starting stance."
    #      ),
    #     ("Provide a summary of the motion demonstrated in <Motion_Placeholder> using words.",
    #      None),
    #     (Video_Components_example.format(
    #         video_path="assets/videos/example2.mp4",
    #         video_fname="example2.mp4"),
    #      "A person who is standing with his arms up and away from his sides bends over, gets down on his hands and then his knees and crawls forward."
    #      ),
    #     ("Generate text for <Motion_Placeholder>:", None),
    #     (Video_Components_example.format(
    #         video_path="assets/videos/example5.mp4",
    #         video_fname="example5.mp4"),
    #      "The man tries to stand in a yoga tree pose and looses his balance."),
    #     ("Provide a summary of the motion depicted in <Motion_Placeholder> using language.",
    #      None),
    #     (Video_Components_example.format(
    #         video_path="assets/videos/example6.mp4",
    #         video_fname="example6.mp4"),
    #      "Person walks up some steps then leeps to the other side and goes up a few more steps and jumps dow"
    #      ),
    #     ("Describe the motion represented by <Motion_Placeholder> in plain English.",
    #      None),
    #     (Video_Components_example.format(
    #         video_path="assets/videos/example7.mp4",
    #         video_fname="example7.mp4"),
    #      "Person sits down, then stands up and walks forward. then the turns around 180 degrees and walks the opposite direction"
    #      ),
    #     ("Provide a description of the action in <Motion_Placeholder> using words.",
    #      None),
    #     (Video_Components_example.format(
    #         video_path="assets/videos/example8.mp4",
    #         video_fname="example8.mp4"),
    #      "This man is bent forward and walks slowly around."),
    # ])

    # t2t_examples = gr.State([
    #     (None,
    #      "💡 Of course, you can also purely chat with me and let me give you human motion in text, here are some examples!"
    #      ),
    #     ('Depict a motion as like you have seen it.',
    #      "A person slowly walked forward in rigth direction while making the circle"
    #      ),
    #     ('Random say something about describing a human motion.',
    #      "A man throws punches using his right hand."),
    #     ('Describe the motion of someone as you will.',
    #      "Person is moving left to right in a dancing stance swaying hips, moving feet left to right with arms held out"
    #      ),
    #     ('Come up with a human motion caption.',
    #      "A person is walking in a counter counterclockwise motion."),
    #     ('Write a sentence about how someone might dance.',
    #      "A person with his hands down by his sides reaches down for something with his right hand, uses the object to make a stirring motion, then places the item back down."
    #      ),
    #     ('Depict a motion as like you have seen it.',
    #      "A person is walking forward a few feet, then turns around, walks back, and continues walking."
    #      )
    # ])

    # Init_chatbot = chat_instruct.value[:
    #                                    1] + t2m_examples.value[:
    #                                                            3] + m2t_examples.value[:3] + t2t_examples.value[:2] + chat_instruct.value[
    #                                                                -7:]

    # Variables
    motion_uploaded = gr.State({
        "feats": None,
        "joints": None,
        "motion_video": None,
        "motion_lengths": 0,
        "motion_token": None,
        "motion_token_string": '',
        "motion_token_length": 0,
    })
    data_stored = gr.State([])

    gr.Markdown("# HumanoidGPT")

    chatbot = gr.Chatbot(
                        # Init_chatbot,
                         elem_id="hGPT",
                         height=600,
                         label="HumanoidGPT",
                         avatar_images=(None,
                                        # ("assets/images/avatar_bot.jpg")
                                        None),
                         bubble_full_width=False)

    with gr.Row():
        with gr.Column(scale=0.85):
            with gr.Row():
                txt = gr.Textbox(
                    label="Text",
                    show_label=False,
                    elem_id="textbox",
                    placeholder=
                    "Enter text and press ENTER or speak to input. You can also upload motion.",
                    container=False)
            with gr.Row():
                aud = gr.Audio(source="microphone",
                               label="Speak input",
                               type='filepath')
                # btn = gr.UploadButton("📁 Upload motion",
                #                       elem_id="upload",
                #                       file_types=["file"])
                # regen = gr.Button("🔄 Regenerate", elem_id="regen")
                clear = gr.ClearButton([txt, chatbot, aud], value='🗑️ Clear')

            # with gr.Row():
            #     gr.Markdown('''
            #     ### You can get more examples (pre-generated for faster response) by clicking the buttons below:
            #     ''')

            # with gr.Row():
            #     instruct_eg = gr.Button("Instructions", elem_id="instruct")
            #     t2m_eg = gr.Button("Text-to-Motion", elem_id="t2m")
            #     m2t_eg = gr.Button("Motion-to-Text", elem_id="m2t")
            #     t2t_eg = gr.Button("Random description", elem_id="t2t")

        # with gr.Column(scale=0.15, min_width=150):
        #     method = gr.Dropdown(["slow", "fast"],
        #                          label="Visulization method",
        #                          interactive=True,
        #                          elem_id="method",
        #                          value="slow")

        #     language = gr.Dropdown(["English", "中文"],
        #                            label="Speech language",
        #                            interactive=True,
        #                            elem_id="language",
        #                            value="English")

    txt_msg = txt.submit(
        add_text, [chatbot, txt, motion_uploaded, data_stored],
        [chatbot, txt, motion_uploaded, data_stored],
        queue=False).then(bot, [chatbot, motion_uploaded, data_stored],
                          [chatbot, motion_uploaded, data_stored])

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    # file_msg = btn.upload(add_file, [chatbot, btn, txt, motion_uploaded],
    #                       [chatbot, txt, motion_uploaded],
    #                       queue=False)
    
    # Whisper comment out
    aud_msg = aud.stop_recording(
        add_audio, [chatbot, aud, data_stored],
        [chatbot, data_stored],
        queue=False).then(bot, [chatbot, motion_uploaded, data_stored],
                          [chatbot, motion_uploaded, data_stored])
    # regen_msg = regen.click(bot,
    #                         [chatbot, motion_uploaded, data_stored, method],
    #                         [chatbot, motion_uploaded, data_stored],
    #                         queue=False)

    # instruct_msg = instruct_eg.click(bot_example, [chatbot, chat_instruct_sum],
    #                                  [chatbot],
    #                                  queue=False)
    # t2m_eg_msg = t2m_eg.click(bot_example, [chatbot, t2m_examples], [chatbot],
    #                           queue=False)
    # m2t_eg_msg = m2t_eg.click(bot_example, [chatbot, m2t_examples], [chatbot],
    #                           queue=False)
    # t2t_eg_msg = t2t_eg.click(bot_example, [chatbot, t2t_examples], [chatbot],
    #                           queue=False)

    chatbot.change(scroll_to_output=True)

demo.queue()

if __name__ == "__main__":
    # demo.launch(server_name="0.0.0.0", server_port=8889, debug=True)
    demo.launch(server_name="localhost", server_port=8889, debug=True)
