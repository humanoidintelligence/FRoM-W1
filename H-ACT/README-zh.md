# H-ACT

**H-ACT** 是 **FRoM-W1** 框架中的动作执行模块，用于将 **H-GPT** 生成的动作表征序列真实地执行在人形机器人上。本模块包含 **FRoM-W1** 中的三个关键步骤：

- **Retarget**  
  将 **H-GPT** 生成的动作表征转换为 **SMPLX** 动作序列，并进一步重定向为不同 **人形机器人** 与 **灵巧手** 的关节动作序列。这意味着它同样支持将 AMASS 等人体动作数据集重定向到多种机器人平台。

- **Policy Training**  
  基于重定向后的动作数据（或其他来源的机器人数据）训练动作执行策略。

- **Sim2sim & Sim2real**  
  将训练好的策略部署到仿真环境或真实机器人，实现 sim2sim 或 sim2real 执行。

# 动作恢复与重定向

该部分主要负责将 **H-GPT** 输出的 623 维动作表征恢复为 **SMPLX** 动作序列，并重定向到目标人形机器人和灵巧手的关节空间中。

## 🧩 环境配置与模型准备

进入 `H-ACT/retarget` 目录，并使用以下命令配置 retarget 环境：

```bash
conda create -n retarget python=3.10
conda activate retarget
pip install -r requirements.txt
```

该模块依赖 **SMPL** 与 **MANO** 模型，因此在使用前需准备对应的模型文件：

1. **下载 MANO 模型**
   访问 [MANO 官方网站](https://mano.is.tue.mpg.de/)，下载并解压模型文件
   （`MANO_LEFT.pkl`, `MANO_RIGHT.pkl`）至 `models/mano`。
2. **下载 SMPL 模型**
   访问 [SMPL 官方网站](https://smpl.is.tue.mpg.de/)，下载并解压模型文件
   （`SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl`）至 `models/smpl`。

模型目录结构示例如下：

```nginx
retarget
├── models
│   ├── mano
│   │   ├── MANO_LEFT.pkl
│   │   └── MANO_RIGHT.pkl
│   └── smpl
│       ├── SMPL_NEUTRAL.pkl
│       ├── SMPL_MALE.pkl
│       └── SMPL_FEMALE.pkl
├── ...
```


## 📁 数据准备

在 `retarget` 目录内创建 `data` 文件夹，用于存放输入与输出数据：

* `data/623`：存放 **H-GPT** 生成的 623 维动作数据
* `data/smplx`：存放重定向后的 **SMPLX** 动作序列（中间表示）
* `data/output`：存放最终生成的机器人及灵巧手的动作序列

目录示例如下：

```kotlin
retarget
├── data
│   ├── 623
│   │   ├── data1.npy  # Output from H-GPT
│   │   └── data2.npy
│   ├── smplx          # Output SMPLX directory
│   ├── output         # Output robot motion directory
├── ...
```

## ▶️ 运行（Retarget 执行）

运行以下命令，将 **H-GPT** 生成的动作表征重定向到目标机器人：

```bash
python main.py
```

本模块当前支持以下机器人与灵巧手模型：

* **Unitree H1**
* **Unitree G1**
* **Dex3**
* **InspireHand**

您可以在 `main.py` 中修改第 **47–48 行** 的参数，选择希望使用的机器人类型。

# 策略训练

在此阶段，我们将基于 **Retarget** 步骤生成的动作数据（或其他来源的机器人数据）训练一个动作执行策略，用于最终的真实机器人部署。  
我们基于 **[Human2Humanoid](https://github.com/LeCAR-Lab/human2humanoid)** 的工作提出一个支持 **UnitreeH1** 与 **UnitreeG1** 的动作执行策略。我们在部署模块中已经提供了该策略的预训练模型（**Unitree G1** 和 **Unitree H1** 各一个）。

如希望自行训练，可参考我们的文档👉 [Train](Human2Humanoid/README-zh.md)

与Human2Humanoid官方仓库👉 [https://github.com/LeCAR-Lab/human2humanoid](https://github.com/LeCAR-Lab/human2humanoid) 

同时，如果你希望尝试其他工作来执行你的动作，我们的部署模块也支持 **[Beyondmimic](https://github.com/HybridRobotics/whole_body_tracking)** 、 **[TWIST](https://github.com/YanjieZe/TWIST)** 等作为策略模型来部署。在拿到最终的策略模型后，可进入[仿真与真机部署](#仿真与真机部署)部分，将策略部署到真实机器人上来执行指定动作。

## Beyondmimic 策略训练

Beyondmimic 要求输入为 **CSV 格式** 的动作数据，因此需要先将重定向后的机器人动作数据转换为 CSV 格式。

切换至目录 `H-ACT` 下，创建目录 `data/beyondmimic` 用于保存结果。
运行以下命令完成转换：

```bash
python scripts/pkl_2_csv.py
```

转换后的动作文件将保存在 `data/beyondmimic` 目录中。

随后可参考官方文档进行训练：
👉 [Beyondmimic](https://github.com/HybridRobotics/whole_body_tracking?tab=readme-ov-file#policy-training)

## TWIST 策略训练

参考官方文档👉 [TWIST](https://github.com/YanjieZe/TWIST)

# 仿真与真机部署

训练好需要部署的模型后，
我们提供了一个统一的仿真与真机部署框架 **[RoboJuDo](https://github.com/GDDG08/RoboJuDo)**，用于将动作策略真正部署到机器人上。

**RoboJuDo** 支持以下功能：

* 使用 Beyondmimic、Human2Humanoid、Twist 等多种策略进行 **sim2sim** 与 **sim2real** 部署
* 提供预训练策略模型，帮助你快速完成物理机器人部署
* 统一且简洁的接口，可方便接入自定义策略模型并快速实现部署

