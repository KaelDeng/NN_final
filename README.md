# 3D重建模型训练与测试指南

本项目包含两个先进的3D重建模型：**Nerfacto** 和 **3D Gaussian Splatting**。

## 📁 项目结构

```
.
├── nerfacto/           # Nerfacto模型实现
├── gaussian-splatting/ # 3D Gaussian Splatting模型实现
└── README.md          # 本文件
```

## 🚀 快速开始

### 环境要求

- **GPU**: NVIDIA GPU with CUDA support (推荐24GB+ VRAM)
- **CUDA**: 11.8 (推荐)
- **Python**: 3.8+
- **操作系统**: Windows 10/11 或 Ubuntu Linux

## 📖 Nerfacto 模型

Nerfacto是一个基于NeRF (Neural Radiance Fields) 的先进3D重建模型，特别适用于真实世界场景的重建。

### 安装

```bash
# 进入nerfacto目录
cd nerfacto

# 创建conda环境
conda create --name nerfstudio -y python=3.12
conda activate nerfstudio

# 安装PyTorch (CUDA 11.8)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118

# 安装CUDA工具包
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# 安装tiny-cuda-nn
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 安装nerfstudio
pip install nerfstudio
```

### 训练

#### 1. 准备数据

```bash
# 使用自己的数据 (需要COLMAP处理)
ns-process-data images --data data/your_images/ --output-dir data/processed/
```

#### 2. 开始训练

```bash
# 基础训练命令
ns-train nerfacto --data data/your_data

# 自定义参数训练
ns-train nerfacto \
    --data data/your_data \
    --max-num-iterations 30000 \
    --steps-per-save 2000 \
    --vis viewer
```

#### 3. 训练参数说明

- `--data`: 数据路径
- `--max-num-iterations`: 最大训练迭代次数 (默认30000)
- `--steps-per-save`: 保存检查点的步数间隔
- `--vis`: 可视化方式 (viewer, tensorboard, wandb)

#### 4. 恢复训练

```bash
ns-train nerfacto --data data/nerfstudio/poster --load-dir outputs/.../nerfstudio_models
```

### 测试与评估

#### 1. 模型评估

```bash
# 计算PSNR等指标
ns-eval --load-config outputs/.../config.yml --output-path results.json
```

#### 2. 可视化

```bash
# 启动查看器
ns-viewer --load-config outputs/.../config.yml
```

#### 3. 渲染视频

```bash
# 渲染360度视频
ns-render --load-config outputs/.../config.yml --output-path renders/ --traj filename
```

### 高级功能
#### 自定义配置

```bash
# 查看所有可用参数
ns-train nerfacto --help

# 自定义学习率
ns-train nerfacto --data data/nerfstudio/poster \
    --optimizers.fields.optimizer.lr 0.01 \
    --optimizers.proposal_networks.optimizer.lr 0.01
```

---

## 🎯 3D Gaussian Splatting 模型

3D Gaussian Splatting是一个基于高斯椭球体的实时渲染方法，能够实现高质量的3D重建和实时渲染。

### 安装

```bash
# 进入gaussian-splatting目录
cd gaussian-splatting

# 创建conda环境
conda env create --file environment.yml
conda activate gaussian_splatting

# Windows用户需要设置环境变量
SET DISTUTILS_USE_SDK=1  # Windows only
```

### 数据准备

#### 1. 使用COLMAP处理图像

```bash
# 安装COLMAP (Ubuntu)
sudo apt-get install colmap

# 处理图像序列
python convert.py -s /path/to/images -o /path/to/output
```

#### 2. 数据格式要求

- 图像文件: `.jpg`, `.png`, `.jpeg`
- 相机参数: COLMAP格式的`cameras.bin`, `images.bin`, `points3D.bin`
- 图像目录结构:
```
scene/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── sparse/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── transforms.json (可选)
```

### 训练

#### 1. 基础训练

```bash
# 基础训练命令
python train.py -s /path/to/scene

# 指定输出目录
python train.py -s /path/to/scene -m /path/to/output
```

#### 2. 训练参数

```bash
python train.py \
    -s /path/to/scene \
    -m /path/to/output \
    --iterations 30000 \
    --resolution 4 \
    --eval \
    --test_iterations 7000 30000 \
    --save_iterations 7000 30000
```

#### 3. 主要参数说明

- `-s, --source_path`: 场景数据路径
- `-m, --model_path`: 模型输出路径
- `--iterations`: 训练迭代次数 (默认30000)
- `--resolution`: 图像分辨率 (1=原始, 2=1/2, 4=1/4, 8=1/8)
- `--eval`: 启用评估模式
- `--test_iterations`: 测试迭代点
- `--save_iterations`: 保存模型迭代点

#### 4. 高级训练选项

```bash
# 使用深度正则化
python train.py -s /path/to/scene -d /path/to/depths/

# 使用曝光补偿
python train.py -s /path/to/scene --exposure_lr_init 0.001 --train_test_exp

# 使用抗锯齿
python train.py -s /path/to/scene --antialiasing

# 快速训练 (使用稀疏Adam优化器)
python train.py -s /path/to/scene --optimizer_type sparse_adam
```

### 测试与评估

#### 1. 渲染测试图像

```bash
# 渲染指定迭代的模型
python render.py --iteration 30000 -s /path/to/scene -m /path/to/model

# 渲染测试集
python render.py --iteration 30000 -s /path/to/scene -m /path/to/model --eval --skip_train
```

#### 2. 计算评估指标

```bash
# 计算PSNR, SSIM等指标
python metrics.py -m "/path/to/model1" "/path/to/model2"
```

#### 3. 完整评估流程

```bash
# 运行完整评估 (训练+渲染+指标计算)
python full_eval.py \
    --mipnerf360 /path/to/mipnerf360 \
    --tanksandtemples /path/to/tanksandtemples \
    --deepblending /path/to/deepblending \
    --output_path ./eval_results
```

### 可视化

#### 1. 实时查看器

训练过程中会自动启动网络查看器，访问 `http://localhost:6009` 查看实时训练进度。

#### 2. SIBR查看器

```bash
# 编译SIBR查看器
cd SIBR_viewers
cmake -B build . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install -j

# 启动查看器
./build/install/bin/SIBR_gaussianViewer_app -m /path/to/model
```

### 性能优化

#### 1. 内存优化

```bash
# 使用CPU存储数据 (减少VRAM使用)
python train.py -s /path/to/scene --data_device cpu

# 降低分辨率
python train.py -s /path/to/scene --resolution 8
```

#### 2. 训练加速

```bash
# 使用稀疏Adam优化器
python train.py -s /path/to/scene --optimizer_type sparse_adam

# 减少迭代次数
python train.py -s /path/to/scene --iterations 15000
```