
This is a Generic Object Tracking Project

## 🔥 Paper Accepted at ICLR 2026!

**GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing**

📄 Paper and Project Page
- https://chenshihfang.github.io/GOT-EDIT

Human perception for object tracking in a 2D video stream arises from the implicit use of prior visual geometry 🛰️ and semantic reasoning 👁️. GOT-Edit aligns with this principle by enabling trackers to infer 3D geometry from 2D streaming inputs for visual tracking.

The core of this work is **cross-modality online model editing**. This mechanism performs online constrained model updates to incorporate geometric information adaptively while preserving semantic discrimination for online adaptation under streaming 2D inputs.
This paradigm is **generalizable across diverse scenarios and environments** 🌐. We hope these advances chart a path toward reliability, safety, and social responsibility in vision systems.

### Raw Results

The raw results are available for download [here](https://drive.google.com/drive/folders/1ChfM4sl3ERkfy6POB_0PaVc4zUrlqkP8?usp=sharing)

| Dataset | Model         |  NPr  |  Suc   |   Pr     | OP50  | OP75  |
|---------|---------------|:-----:|:-----:|:---------:|:-----:|:-----:|
| NfS-30  | ToMP-50       | 84.00 | 66.86 |   80.58   | 84.36 | 53.50 |
|         | PiVOT-L       | 86.66 | 68.22 |   84.53   | 86.05 | 55.45 |
|         | GOT-Edit      | 87.47 | 71.12 |   86.64   | 89.30 | 59.83 |
| LaSOT   | ToMP-50       | 77.98 | 67.57 |   72.24   | 79.79 | 65.06 |
|         | PiVOT-L       | 84.68 | 73.37 |   82.09   | 85.64 | 75.18 |
|         | GOT-Edit      | 85.08 | 75.31 |   83.17   | 86.13 | 77.52 |
| AVisT   | ToMP-50       | 66.66 | 51.61 |   47.74   | 59.47 | 38.88 |
|         | PiVOT-L       | 81.20 | 62.18 |   65.55   | 73.25 | 55.46 |
|         | GOT-Edit      | 82.50 | 64.45 |   68.26   | 74.35 | 59.68 |
| OTB-100 | ToMP-50       | 85.98 | 70.07 |   90.83   | 87.83 | 57.79 |
|         | PiVOT-L       | 88.46 | 71.20 |   94.58   | 89.35 | 55.73 |
|         | GOT-Edit      | 91.47 | 74.96 |   97.42   | 93.02 | 63.22 |

Suc: Success Rate AUC  
Pr:  Precision AUC  
NPr: Normalise Precision AUC  

### Evaluate the Tracking Performance Based on Datasets

```bash
python evaluate_GOT_Edit_results.py  
```  

For the GOT-10K and TrackingNet results, please refer to the public leaderboards on the official evaluation websites for both challenges under the entry named “Edit” or “GOT-Edit.” The NfS results follow the evaluation protocol described [here](https://github.com/visionml/pytracking/issues/400).



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


### Training script

Change directory to `GOT/pytracking/`:

```bash
cd GOT/pytracking/
CUDA_VISIBLE_DEVICES=0,1,2,3 python ltr/run_training_dsA.py tomp GOT-Edit_DA3_378
```

- 💻 Code and pretrained model: More details will be updated soon.

## Consider citing “GOT-Edit” if this project impresses you

```
@inproceedings{gotedit2026iclr,
title     = {{GOT}-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing},
author    = {Shih-Fang Chen and Jun-Cheng Chen and I-Hong Jhuo and Yen-Yu Lin},
booktitle = {Proc. Int. Conf. Learn. Represent. (ICLR)},
year      = {2026}
}
```


# PiVOT :unicorn:

### :fire: PiVOT has been accepted at TMM 2025! 👇
* [Improving Visual Object Tracking through Visual Prompting](https://arxiv.org/abs/2409.18901) | **Code available!**

## Getting started

Please visit [here](https://github.com/chenshihfang/GOT/tree/c02e55fb2eeea087806e2f0b8dd75d67dfacd635) for PiVOT usage details.

> PiVOT proposes a prompt generation network with the pre-trained foundation model CLIP to automatically generate and refine visual prompts, 
enabling the transfer of foundation model knowledge for tracking.

### Raw Results
The raw results can be downloaded from [here](https://drive.google.com/drive/folders/1E0GUaat7rpBiqlRrfDpEgTXlD7GJEyQE?usp=sharing). 

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


## Acknowledgement
This codebase is implemented on [PyTracking](https://github.com/visionml/pytracking) libraries.


## Citing

If you find this repository useful, please consider giving a star :star: and a citation


```
@ARTICLE{TMM_PiVOT,
author={Chen, Shih-Fang and Chen, Jun-Cheng and Jhuo, I-Hong and Lin, Yen-Yu},
journal={IEEE Transactions on Multimedia}, 
title={Improving Visual Object Tracking Through Visual Prompting}, 
year={2025},
volume={27},
number={},
pages={2682-2694},
doi={10.1109/TMM.2025.3535323}}
```

```
@inproceedings{got_jepa,
title     = {{GOT}-JEPA: Generic Object Tracking with Model Adaptation and Occlusion Handling using Joint-Embedding Predictive Architecture},
author    = {Shih-Fang Chen and Jun-Cheng Chen and I-Hong Jhuo and Yen-Yu Lin},
booktitle = {Arxiv},
year      = {2026}
}
```

```
@inproceedings{gotedit2026iclr,
title     = {{GOT}-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing},
author    = {Shih-Fang Chen and Jun-Cheng Chen and I-Hong Jhuo and Yen-Yu Lin},
booktitle = {Proc. Int. Conf. Learn. Represent. (ICLR)},
year      = {2026}
}
```


## Contact:
mail: csf.cs09@nycu.edu.tw
