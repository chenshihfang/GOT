from pytracking.evaluation import Tracker, get_dataset, trackerlist
import torch
import numpy as np
import random
import os

# conda activate gotedit

# CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments_gotedit GOT_Edit --debug 0 --threads 1

def GOT_Edit():

    trackers = trackerlist('tomp', 'got_edit_378_dino_da3', range(1))

    ####################################

    dataset = get_dataset('avist')

    print("GOT_Edit eva load")

    return trackers, dataset

    ####################
