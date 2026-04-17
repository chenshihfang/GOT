#!/bin/bash

# bash install_gotedit.sh /your_anaconda3_path/ gotedit
# .

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
pip install vot-trax==3.0.3 # 4.0.1

# cd depth-anything-3
# pip install -e .

echo ""
echo ""

echo ""
echo ""


echo ""
echo ""
echo "****************** Installing pandas ******************"
# conda install -y pandas
pip install pandas==1.5.3

echo ""
echo ""


echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install --upgrade --force-reinstall  opencv-python==4.6.0.66

echo ""
echo ""
echo "****************** Installing tensorboard ******************"


# pip install tensorboard==2.15.0 tensorboardX==2.6.2.2
pip install --upgrade --force-reinstall   tensorboard==2.16.2 tensorboardX==2.6.2.2
# pip install -U tensorboard 

pip install chardet==5.2.0

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom==0.2.4

echo ""
echo ""
echo "****************** Installing scikit-image ******************"
pip install scikit-image==0.22.0

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib==0.10.1

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


echo ""
echo ""
echo "******** Installing spatial-correlation-sampler. Note: This is required only for KYS tracker **********"
# pip install spatial-correlation-sampler==0.4.0

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py==0.1.4

echo ""
echo ""
echo "****************** Installing ninja-build to compile PreROIPooling ******************"
echo "************************* Need sudo privilege ******************"
# sudo apt-get install ninja-build

echo ""
echo ""
echo "****************** Downloading networks ******************"
# mkdir pytracking/networks


echo ""
echo ""
echo "****************** Setting up environment ******************"
# python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
# python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"


# echo ""
# echo ""
# echo "****************** Installing jpeg4py ******************"
# while true; do
#     read -p "Install jpeg4py for reading images? This step required sudo privilege. Installing jpeg4py is optional, however recommended. [y,n]  " install_flag
#     case $install_flag in
#         [Yy]* ) sudo apt-get install libturbojpeg; break;;
#         [Nn]* ) echo "Skipping jpeg4py installation!"; break;;
#         * ) echo "Please answer y or n  ";;
#     esac
# done

echo ""
echo ""
echo "****************** Installation complete! ******************"

echo ""
echo ""
echo "****************** More networks can be downloaded from the google drive folder https://drive.google.com/drive/folders/1WVhJqvdu-_JG1U-V0IqfxTUa1SBPnL0O ******************"
echo "****************** Or, visit the model zoo at https://github.com/visionml/pytracking/blob/master/MODEL_ZOO.md ******************"


# conda install -y -c conda-forge timm==0.6.13
pip install timm==0.6.13



pip install --upgrade --force-reinstall  iopath==0.1.10
pip install --upgrade --force-reinstall  omegaconf==2.3.0

# pip install --upgrade --force-reinstall  mmcv-full==1.5.0 # or use mim

pip install -U --upgrade --force-reinstall  openmim
# mim install mmcv

mim install mmcv==1.5.0 
# or Run mim using the full path /XXX/pt20X/bin/mim install mmcv==1.5.0

echo "****************** Installing tqdm ******************"


pip install --upgrade --force-reinstall  imageio==2.31.6 openxlab==0.0.28



echo "****************** Installing pytorch ******************"

# pip install --upgrade --force-reinstall torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install --upgrade --force-reinstall torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
# pip install --upgrade --force-reinstall -U xformers==0.0.21
pip install --upgrade --force-reinstall -U xformers==0.0.18

pip install tqdm==4.65.2
pip install --upgrade --force-reinstall requests==2.28.2


# co-tracker
pip install imageio-ffmpeg==0.4.9
pip install flow_vis
pip install hydra-core==1.1.0 mediapy==1.2.0
pip install sphinx==7.2.6
pip install sphinxcontrib-bibtex==2.6.2
# cd co-tracker
# pip install -e .


# dinov2
pip install fvcore==0.1.5.post20221221
pip install --upgrade --force-reinstall  submitit==1.5.0

pip install black==22.6.0
pip install flake8==5.0.4
pip install pylint==2.15.0

pip install --upgrade --force-reinstall --no-cache-dir --extra-index-url https://pypi.nvidia.com cuml-cu11==23.10.0 --timeout 120


pip install fairscale==0.4.13
pip install fire==0.6.0
pip install sentencepiece==0.2.0



echo "****************** Installing matplotlib ******************"
pip install  --upgrade --force-reinstall  matplotlib==3.5.2
pip install --upgrade --force-reinstall  setuptools==60.2.0
pip install --upgrade --force-reinstall  six==1.16.0
pip install --upgrade --force-reinstall  numba==0.57.1
pip install --upgrade --force-reinstall  pytz==2023.3
pip install --upgrade --force-reinstall  rich==13.4.2
pip install --upgrade --force-reinstall protobuf==4.23
pip install --upgrade --force-reinstall urllib3==1.26.18
pip install --upgrade --force-reinstall  numpy==1.24.4  
pip install --upgrade --force-reinstall  pillow==10.0.0



pip install webcolors==1.11
pip install accelerate==1.10.1
pip install deepspeed

# pip install "transformers==4.39.3"
# pip install --upgrade --force-reinstall  numpy==1.24.4  
# pip uninstall -y peft transformers
pip install "peft==0.10.0" "transformers==4.37.2"
# pip3 install mpi4py




pip install  --upgrade --force-reinstall  matplotlib==3.5.2
pip uninstall tikzplotlib
pip install tikzplotlib==0.10.1
pip install --upgrade --force-reinstall  numpy==1.24.4  



# git clone --recurse-submodules https://github.com/facebookresearch/xformers.git
# cd xformers
# git checkout v0.0.18
# git submodule update --init --recursive
# pip install .


