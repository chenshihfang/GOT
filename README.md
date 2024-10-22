[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-visual-object-tracking-through/visual-object-tracking-on-avist)](https://paperswithcode.com/sota/visual-object-tracking-on-avist?p=improving-visual-object-tracking-through)

# PiVOT :unicorn:
This is a Generic Object Tracking Project.  

### :fire: PiVOT has been accepted at TMM 2024! 👇
* [Improving Visual Object Tracking through Visual Prompting](https://arxiv.org/abs/2409.18901) | **Code available!**

## Getting started

This is the official repository for "Improving Visual Object Tracking through Visual Prompting."  

> PiVOT proposes a prompt generation network with the pre-trained foundation model CLIP to automatically generate and refine visual prompts, 
enabling the transfer of foundation model knowledge for tracking.

### Raw Results
The raw results can be downloaded from [here](https://drive.google.com/drive/folders/1E0GUaat7rpBiqlRrfDpEgTXlD7GJEyQE?usp=sharing). 


| Dataset | Model         |  NPr  |  Suc   |   Pr     | OP50  | OP75  |
|---------|---------------|:-----:|:-----:|:---------:|:-----:|:-----:|
| NfS-30  | ToMP-50       | 84.00 | 66.86 |   80.58   | 84.36 | 53.50 |
|         | SeqTrack-L    | 84.35 | 65.46 |   81.93   | 82.37 | 48.69 |
|         | PiVOT-L       | 86.66 | 68.22 |   84.53   | 86.05 | 55.45 |
| LaSOT   | ToMP-50       | 77.98 | 67.57 |   72.24   | 79.79 | 65.06 |
|         | SeqTrack-L    | 81.53 | 72.51 |   79.25   | 82.98 | 72.68 |
|         | PiVOT-L       | 84.68 | 73.37 |   82.09   | 85.64 | 75.18 |
| AVisT   | ToMP-50       | 66.66 | 51.61 |   47.74   | 59.47 | 38.88 |
|         | PiVOT-L       | 81.20 | 62.18 |   65.55   | 73.25 | 55.46 |
| UAV123  | ToMP-50       | 84.79 | 68.97 |   89.70   | 83.84 | 64.63 |
|         | SeqTrack-L    | 85.83 | 69.67 |   91.35   | 84.98 | 63.31 |
|         | PiVOT-L       | 86.74 | 70.66 |   91.80   | 85.69 | 67.06 |
| OTB-100 | ToMP-50       | 85.98 | 70.07 |   90.83   | 87.83 | 57.79 |
|         | PiVOT-L       | 88.46 | 71.20 |   94.58   | 89.35 | 55.73 |

Suc: Success Rate  
Pr:  Precision  
NPr: Normalise Precision  

## Prerequisites

The codebase is built based on [PyTracking](https://github.com/visionml/pytracking).

Familiarity with the PyTracking codebase will help in understanding the structure of this project.

## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/chenshihfang/GOT.git
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

If you find this repository useful, please consider giving a star :star: and a citation


```
@inproceedings{got2024pivot,
title     = {Improving Visual Object Tracking through Visual Prompting},
author    = {Shih-Fang Chen and Jun-Cheng Chen and I-Hong Jhuo and Yen-Yu Lin},
booktitle = {Proc. {arXiv:2409.18901}},
year      = {2024}
}
```

We will update the DOI for TMM after TMM announces it.
```
@misc{PiVOT,
title={Improving Visual Object Tracking through Visual Prompting},
author={Chen, Shih-Fang and Chen, Jun-Cheng and Jhuo, I-Hong and Lin, Yen-Yu},
note={To appear in IEEE Transactions on Multimedia},
publisher={IEEE}
}
```
