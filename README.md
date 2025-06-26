# 3D重建模型训练与测试指南

本项目包含两个先进的3D重建模型：**Nerfacto** 和 **3D Gaussian Splatting**。

## 📁 项目结构

```
.
├── nerfacto/           # Nerfacto模型实现
├── gaussian-splatting/ # 3D Gaussian Splatting模型实现
└── README.md           # 本文件
```

## 🧰 环境要求

- **操作系统**：Ubuntu 20.04 LTS  
- **GPU驱动与CUDA版本**：NVIDIA显卡驱动、CUDA Toolkit 11.8  
- **显卡型号**：NVIDIA GeForce RTX 4090（24GB显存）  
- **处理器**：Intel Core i9高性能多核处理器  
- **内存**：64GB DDR4 RAM  
- **主要依赖的软件环境**：
  - Python 3.8（Anaconda管理环境）
  - PyTorch 2.1.2（CUDA 11.8版本）
  - COLMAP（三维重建工具）
  - Nerf-pytorch（用于NeRF原始模型实验）
  - Nerfstudio（用于Nerfacto变体模型实验）
  - 3D Gaussian Splatting官方实现
  - TensorBoard可视化分析工具

---

## 📖 Nerfacto 模型

Nerfacto 是一个基于 NeRF (Neural Radiance Fields) 的先进 3D 重建模型，适用于真实世界场景的高质量重建。

### 🔧 环境搭建

#### （1）NeRF-pytorch 环境搭建

```bash
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
conda create -n nerf-pytorch python=3.8 -y
conda activate nerf-pytorch
pip install -r requirements.txt
sudo apt install colmap imagemagick -y
```

- 主要依赖：
  - PyTorch 1.4+
  - Matplotlib
  - NumPy
  - imageio 与 imageio-ffmpeg
  - configargparse
  - COLMAP 与 ImageMagick（用于LLFF数据加载）

#### （2）Nerfstudio 环境搭建

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
conda create --name nerfstudio python=3.8 -y
conda activate nerfstudio
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -e .
```

### 🚀 训练

#### NeRF 训练

```bash
python run_nerf.py --config configs/pi.txt
```

#### Nerfacto 训练

```bash
ns-train nerfacto --data data/pi_process --max-num-iterations 90000 --vis viewer+tensorboard
```

### 📈 测试与评估

#### 模型评估

```bash
ns-eval --load-config outputs/.../config.yml --output-path results.json
```

#### 可视化

```bash
tensorboard --logdir outputs/
```

访问浏览器中的 `localhost:6006` 以查看训练曲线。

---

## 🎯 3D Gaussian Splatting 模型

3D Gaussian Splatting 是一种基于高斯椭球体的实时渲染方法，可实现高质量的 3D 重建与渲染。

### 📁 数据处理

3DGS 需要如下数据输入：

- 图像序列（训练输入）
- 每张图像的相机内参与外参（pose）
- 初始稀疏三维点云（用于高斯初始化）

使用流程（以静态玩偶视频为例）：

#### （1）视频抽帧

```bash
ffmpeg -i input.mp4 -qscale:v 2 images/frame_%04d.png
```

共抽取 303 帧，分辨率为 1920×1080。

#### （2）COLMAP 稀疏重建流程

```bash
colmap feature_extractor --database_path database.db --image_path images
colmap exhaustive_matcher --database_path database.db
mkdir sparse
colmap mapper --database_path database.db --image_path images --output_path sparse
colmap model_converter --input_path sparse/0 --output_path output --output_type TXT
```

- 输出内容：
  - `cameras.txt`, `images.txt`: 相机位姿
  - `points3D.txt`: 稀疏点云数据

### 🔧 环境搭建

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
conda create -n gaussian_splatting python=3.8 -y
conda activate gaussian_splatting
pip install -r requirements.txt
cmake . && make
```

### 🚀 训练

```bash
python train.py -s data/qmzy/pi/pi_process --eval
```

### 📊 测试与评估

#### 可视化

```bash
tensorboard --logdir outputs/
```

#### 量化评估

使用 `metrics.py` 脚本对训练完成后的模型进行图像质量评估。

---
