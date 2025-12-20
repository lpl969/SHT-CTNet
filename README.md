# SHT-CTNet

This is the official repository for the paper: **"ENHANCING AUDIO-BASED DRONE TRACKING AND CLASSIFICATION VIA SPHERICAL HARMONIC TRANSFORM"**.

---

## 🛠 Environment Requirement

It is recommended to use `conda` for environment management.

```bash
# Create and activate the environment
conda create -n SHT-CTNet python=3.9 -y
conda activate SHT-CTNet

# Install PyTorch with CUDA 12.1 support
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r environment.txt
📊 Data Preparation
The dataset can be obtained in the following two ways:
Original Dataset: Download from Genesys Lab - Hovering UAVs and process it into .h5 files.
Pre-processed Data: Directly download the pre-processed H5 files from Weiyun (微云).
🚀 Train and Test
All experiments are executed via mix.py. You can switch between different modes by modifying the configuration settings.
Mode 1: Train Only
Configuration:
Set conf.run_for = 'train'
Set training data path: conf.train_h5 = '../data/9ft_train.h5'
Run:
code
Bash
python mix.py
Mode 2: Test Only
Configuration:
Set conf.run_for = 'test'
Set test data path: conf.test_h5 = '../data/9ft_test.h5'
Run:
code
Bash
python mix.py
Mode 3: Automatic Train & Test (Default)
This mode automatically iterates through a predefined set of datasets, completing the entire process of training and then testing for each one.
Configuration:
Set conf.run_for = 'auto'
Run:
code
Bash
python mix.py
📂 Project Structure
FAWGN/, SH_CTNet-S/, SH_CTNet_4/, TAWGN/: Model architecture and core processing modules.
mix.py: Main entry script for training and evaluation.
environment.txt: List of required Python dependencies.
