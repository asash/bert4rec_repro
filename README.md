# BERT4Rec-Replicability

This repository contains the code and the documentation necessary to replicate our results reported in the BERT4Rec replicability and reproducibility study. 

## Installation 
The instruction has been tested on an Ubuntu 22.04 LTS machine with an NVIDIA RTX 3090 GPU. 

Please follow this step-by-step instruction to reproduce our results


## 1. Install Anaconda environment manager: 

If you don't have anaconda installed in your system, please follow the instruction https://docs.anaconda.com/anaconda/install/linux/

## 2. Create the project working directory
```
mkdir aprec_repro
cd aprec_repro
```


## 3. Create an anaconda environment with necessary package versions:
```
conda create -y --name aprec_repro python=3.9.12 cudnn=8.2.1.32 cudatoolkit=11.6.0  pytorch-gpu=1.10.0 tensorflow-gpu=2.6.2 gh=2.1.0 expect=5.45.4
```

## 4. Add working working directory to the PYTHONPATH of the anaconda environment: 
```
conda env config vars set -n aprec_repro PYTHONPATH=`pwd`
```

## 5. Activate the environment
```
conda activate aprec_repro
```

## 6. Install python packages in the environment: 
```
pip3  install "jupyter>=1.0.0" "tqdm>=4.62.3" "requests>=2.26.0" "pandas>=1.3.3" "lightfm>=1.16" "scipy>=1.6.0" "tornado>=6.1" "numpy>=1.19.5" "scikit-learn>=1.0" "lightgbm>=3.3.0" "mmh3>=3.0.0" "matplotlib>=3.4.3" "seaborn>=0.11.2" "jupyterlab>=3.2.2" "telegram_send>=0.25" "transformers>=4.16.1" "recbole>=1.0.1" "wget>=3.2" "pytest>=7.1.2" "pytest-forked>=1.4.0"
```