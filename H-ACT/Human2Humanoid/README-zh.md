<h1 align="center">Human2Humanoid 训练</h1>

# 简介

本项目基于原始的 [Human2Humanoid](https://github.com/LeCAR-Lab/human2humanoid) 框架进行了扩展，并新增支持 **Unitree G1（21-DoF）人形机器人** 的训练流程。本文将介绍完整的环境配置、数据准备以及教师模型与学生模型的训练方式。


# 环境准备

进入项目目录`H-ACT/Human2Humanoid`

## 1. 创建虚拟环境并安装 PyTorch

```bash
conda create -n omnih2o python=3.8
conda activate omnih2o
pip3 install torch torchvision torchaudio
```

## 2. 安装 Isaac Gym

前往官方页面下载 Isaac Gym：
[https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)

下载完成后解压，并运行以下命令进行安装：

```bash
cd isaacgym/python && pip install -e .
```

## 3. 安装 `phc`、`legged_gym`、`rsl_rl` 以及其他依赖

```bash
pip install -r requirements.txt
cd legged_gym
```

# 准备训练数据

我们使用 [**AMASS 数据集**](https://amass.is.tue.mpg.de) 中的动作数据用于继续训练模型。

我们提供了为 **Unitree G1** 与 **Unitree H1** 预处理好的训练数据（**TODO: 添加下载链接**）。

下载后，请将文件重命名为`motion_data.pkl`并放置到`legged_gym/resources/motions/(g1 | h1)`

例如：
* 若训练 **H1** 教师模型，文件应放在
  `legged_gym/resources/motions/h1`
* 若训练 **G1** 教师模型，文件应放在
  `legged_gym/resources/motions/g1`


# 训练教师模型（Teacher Policy）

使用以下命令启动教师模型训练：

```bash
python legged_gym/scripts/train_hydra.py --config-path="../cfg/{robot_config_name}"
```

其中 `{robot_config_name}` 是机器人的配置文件夹名称。

例如，为 **H1** 训练教师模型：

```bash
python legged_gym/scripts/train_hydra.py --config-path="../cfg/cfg_h1"
```
运行模型：

```bash
python legged_gym/scripts/play_hydra.py --config-path="../cfg/cfg_h1" --config-name=config_play
```

# 训练学生模型（Student Policy, Sim2Real）

在开始学生模型训练前，需要根据教师模型的训练结果更新`legged_gym/legged_gym/cfg/{robot_config_name}/cfg_sim2real.yaml`

修改该文件以下字段，使其对应到教师模型的运行记录：

* `load_run_dagger`
* `checkpoint_dagger`

设置完成后，启动学生模型训练：

```bash
python legged_gym/scripts/train_hydra.py --config-path="../cfg/{robot_config_name}" --config-name=config_sim2real
```

例如，为 **H1** 训练学生模型：

```bash
python legged_gym/scripts/train_hydra.py --config-path="../cfg/cfg_h1" --config-name=config_sim2real
```

运行学生模型：

```bash
python legged_gym/scripts/play_hydra.py --config-path="../cfg/cfg_h1" --config-name=config_play_student
```
