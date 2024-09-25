# PiVOT :unicorn:
This is a Generic Object Tracking Project.  

### :fire: PiVOT has been accepted at TMM 2024! 👇
* [Improving Visual Object Tracking through Visual Prompting](https://drive.google.com/file/d/1W6ghQ-GChYSERjvf5CGYkSsx6oa1p6lx/view?usp=sharing) | **Code available!**

## Getting started

This is the official repository for "Improving Visual Object Tracking through Visual Prompting."

### Raw Results
The raw results can be downloaded from [here](https://drive.google.com/drive/folders/1E0GUaat7rpBiqlRrfDpEgTXlD7GJEyQE?usp=sharing). 

| Dataset | Model         | AUC   | OP50  | OP75  | Precision |  NPr  |
|---------|---------------|:-----:|:-----:|:-----:|:---------:|:-----:|
| NFS     | ToMP-50       | 66.86 | 84.36 | 53.50 |   80.58   | 84.00 |
|         | PiVOT-L-27    | 68.22 | 86.05 | 55.45 |   84.53   | 86.66 |
| OTB     | ToMP-50       | 70.07 | 87.83 | 57.79 |   90.83   | 85.98 |
|         | PiVOT-L-27    | 71.20 | 89.35 | 55.73 |   94.58   | 88.46 |
| UAV     | ToMP-50       | 68.97 | 83.84 | 64.63 |   89.70   | 84.79 |
|         | PiVOT-L-27    | 70.66 | 85.69 | 67.06 |   91.80   | 86.74 |
| LaSOT   | ToMP-50       | 67.57 | 79.79 | 65.06 |   72.24   | 77.98 |
|         | PiVOT-L-27    | 73.37 | 85.64 | 75.18 |   82.09   | 84.68 |
| AVIST   | ToMP-50       | 51.61 | 59.47 | 38.88 |   47.74   | 66.66 |
|         | PiVOT-L-27    | 62.18 | 73.25 | 55.46 |   65.55   | 81.20 |


## Prerequisites

The codebase is built based on [PyTracking](https://github.com/visionml/pytracking).

Familiarity with the PyTracking codebase will help in understanding the structure of this project.

## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/chenshihfang/got-pivot.git
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
