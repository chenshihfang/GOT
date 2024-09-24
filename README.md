# PiVOT :unicorn:
This is a Generic Object Tracking Project.  

## Getting started

This is the official repository for "Improving Visual Object Tracking through Visual Prompting."

### :fire: One GOT paper accepted at TMM 2024! 👇
* [Improving Visual Object Tracking through Visual Prompting](https://drive.google.com/file/d/1W6ghQ-GChYSERjvf5CGYkSsx6oa1p6lx/view?usp=sharing) | **Code available!**

## Prerequisites

The codebase is built based on [PyTracking](https://github.com/visionml/pytracking).

Familiarity with the PyTracking codebase will help in understanding the structure of this project.

## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/sfchen94/got-pivot.git
```  

Ensure that CUDA 11.7 is installed.
   
#### Install dependencies
```bash
sudo apt-get install libturbojpeg
```  

Run the installation script to install all the dependencies. 
You need to provide the conda install path and the name for the created conda environment  
```bash
bash install_PiVOT.sh /your_anaconda3_path/ got_pivot
conda activate got_pivot
```  

#### Set Up the Dataset Environment

You can follow the setup instructions from [PyTracking](https://github.com/visionml/pytracking).

There are two different `local.py` files located in:

- `ltr/admin`
- `pytracking/evaluation`


### Raw Results
The raw results can be downloaded from [here](https://drive.google.com/drive/folders/1E0GUaat7rpBiqlRrfDpEgTXlD7GJEyQE?usp=sharing). 


### Evaluate the Tracking Performance Based on Datasets

```bash
python evaluate_PiVOT_results.py  
```  

### Pretrained Model
The pretrained model can be downloaded from [here](https://drive.google.com/drive/folders/1XTFDKt9uTXuODZ0RZ4feD7L98ZBrDmbW?usp=sharing).


### Evaluate the Tracker

1. First, set the parameter `self.infer` to `True` in `ltr/models/tracking/tompnet.py`.
2. Second, set up the Pretrained Model path in `pytracking/pytracking/parameter/tomp/pivotL27.py`.
3. Then execute the following command:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments_pivot pivot --debug 0 --threads 1

### Training

1. First, set the parameter `self.infer` to `False` in:
   `ltr/models/tracking/tompnet.py`

2. Then, proceed with the following stages:

   **Stage 1:**
     ```bash
     python ltr/run_training.py tomp tomp_L_27
     ```

   **Stage 2:**
     Place the `tomp_L_27` checkpoint in:
     `ltr/train_settings/tomp/pivot_L_27.py`

     Then run:
     ```bash
     python ltr/run_training.py tomp pivot_L_27
     ```

## Acknowledgement
This codebase is implemented on [PyTracking](https://github.com/visionml/pytracking) libraries.


## Citing PiVOT

If you find this repository useful, please consider giving a star :star: and citation :thumbsup:

```
@article{pivot2024tmm,
title={Improving Visual Object Tracking through Visual Prompting},
author={Chen, Shih-Fang and Chen, Jun-Cheng and Jhuo, I-Hong and Lin, Yen-Yu},
journal={IEEE Trans. Multimedia (TMM)},
year={2024},
publisher={IEEE}
}
```
