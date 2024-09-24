#!/bin/bash

# bash install_PiVOT.sh /your_anaconda3_path/ got_pivot


if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3 ******************"
conda create -y --name $conda_env_name  python=3.9.18

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

pip install vot-toolkit==0.5.3
pip install vot-trax==3.0.3

echo ""
echo ""


echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas==1.5.3

echo ""
echo ""




echo ""
echo ""
echo "****************** Installing tensorboard ******************"


pip install tensorboard==2.15.0 tensorboardX==2.6.2.2

pip install chardet==5.2.0



echo ""
echo ""
echo "****************** Installing scikit-image ******************"
pip install scikit-image==0.22.0


echo ""
echo ""
echo "****************** Installing gdown ******************"
pip install gdown==4.7.1

echo ""
echo ""
echo "****************** Installing cython ******************"
# conda install -y cython=3.0.4
pip install cython==3.0.5

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools==2.0.7

echo ""
echo ""
echo "****************** Installing LVIS toolkit ******************"
pip install lvis==0.5.3


pip install --upgrade --force-reinstall urllib3==1.26.18

# pip install --upgrade --force-reinstall  submitit==1.5.0
pip install --upgrade --force-reinstall  iopath==0.1.10
pip install --upgrade --force-reinstall  omegaconf==2.3.0

# pip install --upgrade --force-reinstall  mmcv-full==1.5.0 # or use mim

pip install -U openmim
mim install mmcv
mim install mmcv==1.5.0

echo "****************** Installing tqdm ******************"


pip install --upgrade --force-reinstall  imageio==2.31.6 openxlab==0.0.28


echo "****************** Installing CLIP ******************"
# pip install --upgrade --force-reinstall  git+https://github.com/openai/CLIP.git
pip install --upgrade --force-reinstall git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33

pip install --upgrade --force-reinstall  regex==2023.10.3 ftfy==6.1.1 fvcore==0.1.5.post20221221

pip install --upgrade --force-reinstall  numba==0.57.1

pip install --upgrade --force-reinstall  numpy==1.24.4  

echo "****************** Installing dinov2 dep ******************"

pip install --upgrade --force-reinstall --no-cache-dir --extra-index-url https://pypi.nvidia.com cuml-cu11==23.10.0
pip install --upgrade --force-reinstall --no-cache-dir --extra-index-url https://pypi.nvidia.com cudf-cu11==23.10.0

echo "****************** Installing pytorch ******************"

pip install --upgrade --force-reinstall torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install --upgrade --force-reinstall -U xformers==0.0.18

echo "****************** Installing  ******************

pip install tqdm==4.65.2
pip install --upgrade --force-reinstall  pillow==10.0.0
pip install --upgrade --force-reinstall requests==2.28.2
pip install --upgrade --force-reinstall  setuptools==60.2.0
pip install protobuf==4.23


echo "****************** Installing opencv ******************
pip install --upgrade --force-reinstall  opencv-python==4.6.0.66

echo "****************** Installing matplotlib ******************
pip install matplotlib==3.5.2

echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py==0.1.4

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom==0.2.4

echo "****************** Installing timm ******************"
pip install timm==0.6.13

echo "****************** Installing numpy ******************
pip install --upgrade --force-reinstall  numpy==1.24.4  


echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib==0.10.1

echo ""
echo ""
echo "****************** Installation complete! ******************"