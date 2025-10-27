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

```bash
# Create a virtual environment
conda create -n yourname python=3.10
conda activate yourname





Run the following command to start training:

```bash
python train.py











