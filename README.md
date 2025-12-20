# SHT-CTNet

本仓库对应论文 **“ENHANCING AUDIO-BASED DRONE TRACKING AND CLASSIFICATION VIA SPHERICAL HARMONIC TRANSFORM”** 的实现代码。

## 环境准备

conda create -n <conda_name> python=3.9
conda activate <conda_name>
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r environment.txt## 数据准备

1. 从 https://genesys-lab.org/hovering-uavs 获取原始数据并处理成 H5 文件。  
2. 或直接下载处理好的数据集：https://share.weiyun.com/Mo2LiXtP。

## 训练与测试

### 模式一：仅训练
- `conf.run_for = 'train'`
- `conf.train_h5 = '../data/9ft_train.h5'`
- 运行 `python mix.py`

### 模式二：仅测试
- `conf.run_for = 'test'`
- `conf.test_h5 = '../data/9ft_test.h5'`
- 运行 `python mix.py`

### 模式三：自动训练 + 测试（默认）
- `conf.run_for = 'auto'`
- 运行 `python mix.py`，程序会依次遍历预定义数据集并完成训练与测试。
