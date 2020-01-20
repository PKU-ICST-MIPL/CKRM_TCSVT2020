# Introduction

This is the source code of our TCSVT 2020 paper "Multi-level Knowledge Injecting for Visual Commonsense Reasoning". Please cite the following paper if you find our code useful.

Zhang Wen and Yuxin Peng, "Multi-level Knowledge Injecting for Visual Commonsense Reasoning", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2020. 

# Preparation

This repo is based on the [VCR dataset repo] ((https://github.com/rowanz/r2c). The environment and setup process are pretty much the same. Specifically, we use Python 3.6, PyTorch 1.0, cuda 9.0 and test on Ubuntu 16.04 LTS.

1. Get the dataset. Follow the steps in `data/README.md`.

2. Install cuda 9.0 if it's not available already.

3. Install anaconda, and create a new environment. You need to install a few things, namely, pytorch 1.0, torchvision (layers branch, which has ROI pooling), and allennlp.

```
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
conda update -n base -c defaults conda
conda create --name CKRM python=3.6
source activate CKRM

conda install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn jpeg

conda install pytorch -c pytorch
pip install git+git://github.com/pytorch/vision.git@24577864e92b72f7066e1ed16e978e873e19d13d

pip install -r allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0
python -m spacy download en_core_web_sm


# this one is optional but it should help make things faster
pip uninstall pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```
4. Set up the environment, run source activate CKRM && export PYTHONPATH={path_to_this_repo}.

# Train/Evaluate models

Please refer to models/README.md

## help

For any questions, fell  free to open an issue or contact us. (wen_zhang@pku.edu.cn)
