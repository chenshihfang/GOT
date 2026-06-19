# 🦄 GOT: Generic Object Tracking Project

This repository contains the implementation of our Generic Object Tracking (GOT) projects, including **GOT-JEPA** and **PiVOT**.
The codebase is built upon [PyTracking](https://github.com/visionml/pytracking).

---

## 🔥 News

### GOT-JEPA accepted by IEEE TCSVT 2026

**Paper:** [GOT-JEPA: Generic Object Tracking with Model Adaptation and Occlusion Handling using Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2602.14771)

> GOT-JEPA is a learning framework that enables dynamic model adaptation in adverse environments and fine-grained occlusion perception for generic object tracking.

### PiVOT accepted by IEEE TMM 2025

**Paper:** [Improving Visual Object Tracking Through Visual Prompting](https://arxiv.org/abs/2409.18901)

> PiVOT introduces a prompt generation network guided by the pre-trained foundation model CLIP to automatically generate and refine visual prompts, enabling effective transfer of foundation model knowledge for visual object tracking.

---

## Raw Results and Models

The raw results and models for GOT-JEPA are available here:

* [GOT-JEPA Raw Results and Models](https://drive.google.com/drive/folders/1fM0MKmMDTL7bmONtmaeVTVuEuf0cVryc?usp=sharing)

The raw results for PiVOT are available here:

* [PiVOT Raw Results](https://drive.google.com/drive/folders/1E0GUaat7rpBiqlRrfDpEgTXlD7GJEyQE?usp=sharing)

For PiVOT usage details, please refer to the previous project version:

* [PiVOT Usage Details](https://github.com/chenshihfang/GOT/tree/c02e55fb2eeea087806e2f0b8dd75d67dfacd635)

---

## Main Benchmark Results

| Dataset | Model    |  NPr  |  Suc  |   Pr  |  OP50 |  OP75 |
| ------- | -------- | :---: | :---: | :---: | :---: | :---: |
| NfS-30  | ToMP-50  | 84.00 | 66.86 | 80.58 | 84.36 | 53.50 |
|         | PiVOT-L  | 86.66 | 68.22 | 84.53 | 86.05 | 55.45 |
|         | GOT-JEPA | 87.49 | 70.81 | 86.02 | 89.59 | 58.61 |
| LaSOT   | ToMP-50  | 77.98 | 67.57 | 72.24 | 79.79 | 65.06 |
|         | PiVOT-L  | 84.68 | 73.37 | 82.09 | 85.64 | 75.18 |
|         | GOT-JEPA | 85.25 | 75.36 | 83.22 | 86.46 | 77.40 |
| AVisT   | ToMP-50  | 66.66 | 51.61 | 47.74 | 59.47 | 38.88 |
|         | PiVOT-L  | 81.20 | 62.18 | 65.55 | 73.25 | 55.46 |
|         | GOT-JEPA | 81.97 | 63.70 | 66.54 | 73.73 | 58.11 |
| OTB-100 | ToMP-50  | 85.98 | 70.07 | 90.83 | 87.83 | 57.79 |
|         | PiVOT-L  | 88.46 | 71.20 | 94.58 | 89.35 | 55.73 |
|         | GOT-JEPA | 88.88 | 73.24 | 94.85 | 86.46 | 60.34 |


**Metric definitions**

* **Suc:** Success AUC
* **Pr:** Precision
* **NPr:** Normalized Precision
* **OP50 / OP75:** Overlap Precision at IoU thresholds 0.50 and 0.75

---

## Prerequisites

This project is based on [PyTracking](https://github.com/visionml/pytracking).
Familiarity with the PyTracking codebase is helpful for understanding the project structure.

The experiments are conducted with:

* CUDA 11.7
* PyTorch 2.0.0

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/chenshihfang/GOT.git
cd GOT
```

### 2. Install system dependencies

```bash
sudo apt-get install libturbojpeg
```

### 3. Create the conda environment

Run the installation script with your Anaconda path and the desired environment name:

```bash
bash install_gotjepa.sh /your_anaconda3_path/ gotjepa
conda activate gotjepa
```

---

## Dataset Setup

Please follow the dataset setup instructions from [PyTracking](https://github.com/visionml/pytracking).

You need to configure the dataset paths in the following two `local.py` files:

```text
ltr/admin/local.py
pytracking/evaluation/local.py
```

---

## Checkpoint Setup

Please update the checkpoint paths in:

```text
ltr/models/backbone/resnet.py
ltr/cotracker2/co_tracker/cotracker/models/build_cotracker.py
```

This file contains the function calls for both the semantic and geometry backbones.

---

## Evaluation

To evaluate the tracking performance on different datasets, run:

```bash
python evaluate_GOT_JEPA_results.py
```

---

## Running the Tracker

To run GOT-JEPA, use:

```bash
CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments_gotjepa GOT_JEPA --debug 0 --threads 1
```

---

## Training

Change directory to `GOT/pytracking/`:

```bash
cd GOT/pytracking/
```

Run the following scripts for GOT-JEPA pretraining, fine-tuning, and the OccuSolver variant:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ltr/run_training_dsA.py tomp GOT_JEPA_378_pretrain

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ltr/run_training_dsA.py tomp GOT_JEPA_378_finetune

CUDA_VISIBLE_DEVICES=0,1,2,3 python ltr/run_training_dsA.py tomp GOT_JEPA_378_PT
```

---

## Acknowledgement

This codebase is implemented based on the [PyTracking](https://github.com/visionml/pytracking) library.
We sincerely thank the authors for their excellent open-source tracking framework.

---

## Citation

If you find this repository useful, please consider giving it a ⭐ and citing our papers.

```bibtex
@ARTICLE{TCSVT_GOT_JEPA,
  title={{GOT-JEPA}: Generic Object Tracking with Model Adaptation and Occlusion Handling using Joint-Embedding Predictive Architecture},
  author={Chen, Shih-Fang and Chen, Jun-Cheng and Jhuo, I-Hong and Lin, Yen-Yu},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  doi={10.1109/TCSVT.2026.3675005}
}
```

```bibtex
@ARTICLE{TMM_PiVOT,
  title={Improving Visual Object Tracking Through Visual Prompting},
  author={Chen, Shih-Fang and Chen, Jun-Cheng and Jhuo, I-Hong and Lin, Yen-Yu},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  volume={27},
  pages={2682--2694},
  doi={10.1109/TMM.2025.3535323}
}
```

---

## Contact

For questions, please contact:

```text
csf.cs09@nycu.edu.tw
shihfang1207@gmail.com
```
