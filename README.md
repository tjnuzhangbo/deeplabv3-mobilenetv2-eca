# Medical Image Segmentation Model

This project is implemented based on PyTorch and is designed for semantic segmentation of plaques in coronary artery Optical Coherence Tomography (OCT) images.

# Overview 

Overall structure of the improved Deeplab V3+
<img width="855" height="425" alt="image" src="https://github.com/user-attachments/assets/ed917ecf-3c8a-4d5b-a7b9-34972baa115d" />

ECA Attention Mechanism
<img width="859" height="315" alt="image" src="https://github.com/user-attachments/assets/e84b27bf-21ae-4895-9ac3-fb4ad0489e09" />

MobileNetV2 Architecture
<img width="681" height="485" alt="image" src="https://github.com/user-attachments/assets/5ecaa157-2939-44b7-9c6e-cea34ae8cedf" />

---

## 📘 Overview / 项目简介

The model is built on an encoder-decoder structure and is improved in three key aspects: Firstly, MobileNetV2 is used as the backbone network in the encoder, optimizing feature extraction efficiency through its linear bottleneck structure and inverted residual units, significantly reducing the number of model parameters. Secondly, the multi-scale feature extraction capability of the Atrous Spatial Pyramid Pooling (ASPP) module is utilized to compensate for the feature loss caused by the lightweight design of MobileNetV2, enhancing the model's robustness to plaque morphology. Thirdly, an Efficient Channel Attention (ECA) module is embedded at the encoder-decoder skip connection, dynamically calibrating the weights of multi-scale feature channels to significantly improve the boundary recognition accuracy of plaques. 

---

## 🚀 Quick Start / 快速开始

## Quick Start

### 1️⃣ Environment Setup / 环境配置

Make sure you have Python 3.10+ and PyTorch installed. You can create a virtual environment and install dependencies as follows:

Create a virtual environment

conda create -n yourname python=3.10

conda activate yourname

### 2️⃣ Dataset Description / 数据集说明

#### 1. Public Dataset Used / 使用的公开数据集
We used the publicly available dataset **“OCT Dataset for Segmentation of Atherosclerotic Plaque Morphological Features”** (DOI: 10.5281/zenodo.14478210) for part of our experiments.  
This dataset is licensed under **CC BY 4.0**, which allows reuse and modification with appropriate attribution.  
Original dataset link: https://zenodo.org/records/14478210  

We used the publicly available dataset **“OCT Dataset for Segmentation of Atherosclerotic Plaque Morphological Features”** for part of our experiments.  
- **DOI / Link:** [10.5281/zenodo.14478210](https://zenodo.org/records/14478210)  
- **Authors:** Yu et al.  
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)  

We used the publicly available dataset **“OCT Dataset for Segmentation of Atherosclerotic Plaque Morphological Features”** (DOI: 10.5281/zenodo.14478210, Authors: Yu et al., License: CC BY 4.0) for part of our experiments.  
Original dataset link: https://zenodo.org/records/14478210  

**Our processing:**  
- We **re-annotated the dataset**, selecting **only one type of plaque** for our segmentation task.  
- Images and corresponding masks were resized to 512×512 pixels.  
- Training and validation splits were prepared as described in our experiments (Training: 80%, Validation: 20%).  

> Note: This processed subset is a **derivative of the original dataset**.  
> Please cite the original dataset (Yu et al., DOI: 10.5281/zenodo.14478210) when using this data.


#### 2. Private/Internal Dataset / 自有私有数据集
We also conducted experiments using our own **in-house dataset** of coronary artery OCT images.  
Due to patient privacy and institutional restrictions, this dataset **cannot be publicly shared**.  
Experiments with this dataset were performed **independently of the public dataset**.

#### 3. Dataset Splits / 数据集划分
Experiments were conducted **separately** for the two datasets, each split into training and validation sets as follows:

- **Public dataset (processed derivative)**:  
  - Training: 80%  
  - Validation: 20%  

- **Private/internal dataset**:  
  - Training: 80%  
  - Validation: 20%  

### 3️⃣ Training / 模型训练

Run the following command to start training:

python train.py

### 4️⃣ Evaluation / 模型评估

python predict.py
























