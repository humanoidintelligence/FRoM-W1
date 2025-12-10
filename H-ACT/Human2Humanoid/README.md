<h1 align="center">Human2Humanoid Training</h1>

# Introduction

This project extends the original [Human2Humanoid](https://github.com/LeCAR-Lab/human2humanoid) framework and adds full training support for the **Unitree G1 humanoid robot (21 DoF)**.  
This document provides a complete guide for environment setup, dataset preparation, and training both teacher and student policies.

# Environment Setup

1. **Create a virtual environment and install PyTorch:**

   ```bash
   conda create -n omnih2o python=3.8
   conda activate omnih2o
   pip3 install torch torchvision torchaudio
   ```

2. **Install Isaac Gym:**
   Download from: [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)

   Extract the package and install it via pip:

   ```bash
   cd isaacgym/python && pip install -e .
   ```

3. **Install `phc`, `legged_gym`, `rsl_rl`, and other dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

# Preparing Training Data

We use data from the **AMASS dataset** ([https://amass.is.tue.mpg.de](https://amass.is.tue.mpg.de)).

We also provide preprocessed training datasets for Unitree G1 and Unitree H1
(TODO: Add download links).

After downloading the dataset, rename it to: `motion_data.pkl` and place it under:`legged_gym/resources/motions/(g1 | h1)`.
For example:

- For training the **H1** teacher model, place it in
`legged_gym/resources/motions/h1`
- For training the **G1** teacher model, place it in
`legged_gym/resources/motions/g1`

# Training the Teacher Policy

Use the following command to start training the teacher policy:

```cmd
python legged_gym/scripts/train_hydra.py --config-path="../cfg/{robot_config_name}"
```

Here, `{robot_config_name}` refers to the configuration folder for your robot.

For example, to train the teacher policy for **H1**:

```cmd
python legged_gym/scripts/train_hydra.py --config-path="../cfg/cfg_h1"
```

For playing, run:
```cmd
python legged_gym/scripts/train_hydra.py --config-path="../cfg/cfg_h1" --config-name=config_play
```

# Training the Student Policy

Before training a student policy, update the following fields in: `legged_gym/legged_gym/cfg/{robot_config_name}/cfg_sim2real.yaml`. Modify these entries to match the teacher model's training run:

- `load_run_dagger`
- `checkpoint_dagger`

After updating the configuration, start training the student model:

```cmd
python legged_gym/scripts/train_hydra.py --config-path="../cfg/{robot_config_name}" --config-name=config_sim2real
```

For playing, run:
```cmd
python legged_gym/scripts/train_hydra.py --config-path="../cfg/cfg_h1" --config-name=config_play_student
```
