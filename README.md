# SHT-CTNet

This repository provides the implementation for the paper **“Enhancing Audio-Based Drone Tracking and Classification via Spherical Harmonic Transform.”**

## Environment Setup

conda create -n <conda_name> python=3.9

conda activate <conda_name>

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r environment.txt

##Data Preparation

1. Download the raw dataset from https://genesys-lab.org/hovering-uavs and convert it into H5 format.  
2. Alternatively, download the preprocessed dataset directly from https://share.weiyun.com/Mo2LiXtP.

## Training and Testing

### Mode 1: Train Only
- Set `conf.run_for = 'train'`
- Set `conf.train_h5 = '../data/9ft_train.h5'`
- Run `python mix.py`

### Mode 2: Test Only
- Set `conf.run_for = 'test'`
- Set `conf.test_h5 = '../data/9ft_test.h5'`
- Run `python mix.py`

### Mode 3: Automatic Train & Test (Default)
- Set `conf.run_for = 'auto'`
- Run `python mix.py`; it iterates through predefined datasets, completing training and testing for each.
