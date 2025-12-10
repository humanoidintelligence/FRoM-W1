<div align="center">
<h1> H-GPT</h1>
</div>

# Overview

The motion-language model **H-GPT** serves as the central generative module of FRoM-W1, designed to produce diverse and semantically accurate whole-body motions conditioned on natural language instructions via a **Whole-Body Motion Tokenizer** and an auto-regressive **Motion Generator**.

# Open-Source Roadmap

- [] Pre-trained weights of all components
- [] Training Recipe
- [] Evaluation Recipe
- [] Deployment Tutorial


# Pre-trained Weights

We release the weights of all components at HuggingFace:

| Components | Weights |
| :--- | :--- |
| Whole-Body Motion Tokenizer | 🤗 |
| Motion Generator (NoCoT) | 🤗 |
| Motion Generator (CoT) | 🤗 |




# Training

## 🧩 Environment Setup

Run the following commands to setup the environments for training

```bash
conda create -n hgpt python=3.10
conda activate hgpt
pip install -r requirements.txt
```

## 📁 Data Preparation


## 🚀 Train H-GPT


# Evaluation

# Deployment

## 🧩 Environment Setup

Run the following commands to setup the environments for deployment

```bash
conda create -n hgpt_deploy python=3.10
conda activate hgpt_deploy
pip install -r requirements_deploy.txt
```

## ⚙️ Configuration

Follow the steps below to prepare the models and configurations

1. Download the [Whole-Body Motion Tokenizer]() and the [Motion Generator]()
```bash
# TODO replace the REMOTE_PATH with actual path
huggingface-cli download TOKENIZER_REMOTE_PATH --local-dir LOCAL_DIR
huggingface-cli download GENERATOR_REMOTE_PATH --local-dir LOCAL_DIR
```

2. Replace the path to the motion tokenizer and the motion generator at line 55 & 78 of `hGPT/configs/config_deployment_cot.yaml`


## 🚀 Deploy H-GPT

Run `sh app.sh` to deploy the model to a gradio app.
