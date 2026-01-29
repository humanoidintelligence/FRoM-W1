<div align="center">

# FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions

  <img src="./assets/hi_logo.jpg" alt="FRoM-W1" width="7.5%">

  The [Humanoid Intelligence Team](https://github.com/humanoidintelligence) from [FudanNLP](https://nlp.fudan.edu.cn/nlpen/main.htm) and [OpenMOSS](https://openmoss.github.io/)

<p align="center">
  <a href="https://openmoss.github.io/FRoM-W1/">
    <img src="https://img.shields.io/badge/Project-Webpage-blue.svg" alt="Project Webpage"/>
  </a>
  <a href="https://arxiv.org/abs/2601.12799">
    <img src="https://img.shields.io/badge/arXiv-2601.12799-b31b1b.svg" alt="Paper on arXiv"/>
  </a>
  <a href="https://github.com/OpenMOSS/FRoM-W1">
    <img src="https://img.shields.io/badge/GitHub-Code-black.svg?logo=github" alt="GitHub Code"/>
  </a>
  <a href="https://huggingface.co/datasets/OpenMOSS-Team/FRoM-W1-Datasets">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Data-yellow.svg" alt="Hugging Face Data"/>
  </a>
  <a href="https://huggingface.co/OpenMOSS-Team/FRoM-W1">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow.svg" alt="Hugging Face Model"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
  </a>
</p>

</div>

## 🌟 Introduction
<div align="center">
  <img src="./assets/FRoM-W1-Teaser.png" alt="FRoM-W1" width="50%">
</div>

> For more information, please refer to our [project page](https://openmoss.github.io/FRoM-W1/) and [technical report](https://arxiv.org/abs/2601.12799).

Humanoid robots are capable of performing various actions such as greeting, dancing and even backflipping. However, these motions are often hard-coded or specifically trained, which limits their versatility. In this work, we present **FRoM-W1**[^1], an open-source framework designed to achieve general humanoid whole-body motion control using natural language.

To universally understand natural language and generate corresponding motions, as well as enable various humanoid robots to stably execute these motions in the physical world under gravity, **FRoM-W1** operates in two stages:

**(a) H-GPT**  
Utilizing massive human data, a large-scale language-driven human whole-body motion generation model is trained to generate diverse natural behaviors. We further leverage the Chain-of-Thought technique to improve the model's generalization in instruction understanding.

**(b) H-ACT**  
After retargeting generated human whole-body motions into robot-specific actions, a motion controller that is pretrained and further fine-tuned through reinforcement learning in physical simulation enables humanoid robots to accurately and stably perform corresponding actions. It is then deployed on real robots via a modular sim-to-real module.

We extensively evaluate **FRoM-W1** on Unitree H1 and G1 robots. Results demonstrate superior performance on the HumanML3D-X benchmark for human whole-body motion generation, and our introduced reinforcement learning fine-tuning consistently improves both motion tracking accuracy and task success rates of these humanoid robots. We open-source the entire **FRoM-W1** framework and hope it will advance the development of humanoid intelligence. 

[^1]: **F**oundational Humanoid **Ro**bot **M**odel - **W**hole-Body Control, Version **1**

## 🔥 Roadmap

- [x] 🎉 Release the initial codebase for the **[H-GPT](./H-GPT/README.md)** and **[H-ACT](./H-ACT/README.md)** modules
- [x] 🎉 Release the amazing humanoid-robot deployment framework **[RoboJuDo](https://github.com/HansZ8/RoboJuDo)**
- [x] Release the CoT datasets of the HumanML3D-X and Motion-X benchmarks, and the δHumanML3D-X benchmark
- [x] Release checkpoints for the baseline models, SMPL-X version of T2M, MotionDiffuse, MLD, T2M-GPT
- [x] 🎉 Release the **[Technical Report](https://arxiv.org/abs/2601.12799)** and **[Project Page](https://openmoss.github.io/FRoM-W1/)** of FRoM-W1!
- [ ] More powerful models are working in progress

## 💾 Datasets

Due to license restrictions, we cannot publicly share all of the data. Here are the reference download and processing links for the relevant datasets:

**H-GPT Module**

| **Dataset Name** | **Download Guide** |
|:----------------:|:------------------:|
|    HumanML3D-X   | Please refer to the process in the [Motion-X](https://github.com/IDEA-Research/Motion-X) repo to download and process the corresponding AMASS data. The CoT part can be downloaded [here](https://huggingface.co/datasets/OpenMOSS-Team/FRoM-W1-Datasets/tree/main/data).|
|   δHumanML3D-X   | After obtaining the HumanML3D-X data, replace the textual instructions in it with the perturbed versions provided [here](https://huggingface.co/datasets/OpenMOSS-Team/FRoM-W1-Datasets/tree/main/data). |
|     Motion-X     | Please refer to the original [Motion-X](https://github.com/IDEA-Research/Motion-X) repo. Note that we did not use the Motion-X++ version; specifically, we used the version from [2024.2.6].|

**H-ACT Module**

| **Dataset Name** | **Download Guide** |
|:----------------:|:------------------:|
|       AMASS      | Please refer to the download and processing procedures for the [AMASS](https://amass.is.tue.mpg.de/index.html) dataset in the [human2humanoid](https://github.com/LeCAR-Lab/human2humanoid?tab=readme-ov-file#amass-dataset-preparation) project. |
|     AMASS-H1     | The retargeted dataset for the Unitree H1 can be obtained from the [link](https://cmu.app.box.com/s/vfi619ox7lwf2hzzi710p3g2l59aeczv) provided by human2humanoid.|
|     AMASS-G1     | We provide a retargeted dataset for the Unitree G1, with the link available [here]().|

## 🧠 Models

To keep the repo organized, we provide a subset of core model checkpoints below:

**H-GPT Module**

| **Model Name** | **Download Guide** |
|:--------------:|:------------------:|
|     Eval Model   |    [HuggingFace link](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main/eval), which were trained following the [T2M](https://github.com/EricGuo5513/text-to-motion) pipeline with the SMPL-X format. |
|  Baseline Models |    [HuggingFace link](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main/baselines), including the SMPL-X version of the [T2M](https://github.com/EricGuo5513/text-to-motion), [MotionDiffuse](https://github.com/MotrixLab/MotionDiffuse), [MLD](https://github.com/ChenFengYe/motion-latent-diffusion/tree/main) and [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) models. |
|  H-GPT w.o. CoT  |  [HuggingFace link](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main/hgpt/humanml3d-x/lora/llama-3.1-nocot_maskinput_pkeep), you can refer to this [script](https://huggingface.co/OpenMOSS-Team/FRoM-W1/blob/main/lora_merge.py) to merge these LoRA parameters with the original [Llama-3.1](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) model. |
|       H-GPT      |  [HuggingFace link](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main/hgpt/humanml3d-x/lora/llama-3.1-cot_maskinput_pkeep), you can refer to this [script](https://huggingface.co/OpenMOSS-Team/FRoM-W1/blob/main/lora_merge.py) to merge these LoRA parameters with the original [Llama-3.1](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) model.  |
| H-GPT++ w.o. CoT |  [HuggingFace link](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main/hgpt/motionx/lora/llama-3.1-nocot), you can refer to this [script](https://huggingface.co/OpenMOSS-Team/FRoM-W1/blob/main/lora_merge.py) to merge these LoRA parameters with the original [Llama-3.1](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) model. |
|      H-GPT++     |  [HuggingFace link](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main/hgpt/motionx/lora/llama-3.1-cot), you can refer to this [script](https://huggingface.co/OpenMOSS-Team/FRoM-W1/blob/main/lora_merge.py) to merge these LoRA parameters with the original [Llama-3.1](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) model.    |

**H-ACT Module**

| **Model Name** | **Download Guide** |
|:--------------:|:------------------:|
|      H1-Full     |   [Teacher Policy](), [Student Policy](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main/hact/h1/25_12_10_14-16-23_OmniH2O_STUDENT)      |
|      H1-Clean    |   [Teacher Policy](), [Student Policy](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main/hact/h1/25_12_10_14-13-33_OmniH2O_STUDENT_filter)      |
|      G1-Full     |   [Teacher Policy](), [Student Policy](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main/hact/g1/25_12_11_18-16-37_OmniH2O_STUDENT)      |
|      G1-Clean    |   [Teacher Policy](), [Student Policy](https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/main/hact/g1/25_12_11_18-18-10_OmniH2O_STUDENT_FILTER)      |

If you require additional model checkpoints, please contact us.

## 🚀 Quick Start

### Setup

```bash
conda create -n fromw1 python=3.10
conda activate fromw1
pip install -r ./H-GPT/requirements_deploy.txt
pip install -r ./H-ACT/retarget/requirements.txt
```

### Inference

**H-GPT**

1. Download the H-GPT whole-body motion tokenizer and the motion generator from the HuggingFace.
2. Replace the path to the motion tokenizer and the motion generator at line 55 & 78 of `./H-GPT/hGPT/configs/config_deployment_cot.yaml`
3. Run `bash ./H-GPT/app.sh` to deploy the H-GPT model to a gradio app and generate human motions.

**H-ACT**

1. Download the [SMPL](https://smpl.is.tue.mpg.de/) and [MANO](https://mano.is.tue.mpg.de/) models and organize them according to the H-ACT README file. 
2. Run `python ./H-ACT/retarget/main.py` to retarget the generated human motions into humanoid robot-specific joint sequences.

### Deployment

After obtaining the redirected robot sequence, you can conveniently use our [RoboJudo](https://github.com/OpenMOSS/RoboJuDo) repo to track various strategies in both simulation and real-world scenarios.


## 🛠️ Model Training and Evaluation

### H-GPT

Please refer to the corresponding H-GPT [README](./H-GPT/README.md) file in the subfolder.

### H-ACT

Please refer to the corresponding H-ACT [README](./H-ACT/README.md) file in the subfolder.

## 🙏 Acknowledgements

We extend our gratitude to Biao Jiang for discussions and assistance regarding the motion generation models, to Tairan He and Ziwen Zhuang for their discussions and help in the motion tracking section.

And we thank all the relevant open-source datasets and open-source codes; it is these open-source projects that have propelled the advancement of the entire field!

## 📄 Citation
If you find our work useful, please cite it in the following way:

```bibtex
@misc{li2026fromw1generalhumanoidwholebody,
      title={FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions}, 
      author={Peng Li and Zihan Zhuang and Yangfan Gao and Yi Dong and Sixian Li and Changhao Jiang and Shihan Dou and Zhiheng Xi and Enyu Zhou and Jixuan Huang and Hui Li and Jingjing Gong and Xingjun Ma and Tao Gui and Zuxuan Wu and Qi Zhang and Xuanjing Huang and Yu-Gang Jiang and Xipeng Qiu},
      year={2026},
      eprint={2601.12799},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.12799}, 
}
```

Welcome to star ⭐ our GitHub Repo, raise issues, and submit PRs!
