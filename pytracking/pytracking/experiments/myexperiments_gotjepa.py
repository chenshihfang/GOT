from pytracking.evaluation import Tracker, get_dataset, trackerlist
import torch
import numpy as np
import random
import os

# conda activate gotjepa

# CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments_gotjepa GOT_JEPA --debug 0 --threads 1

def GOT_JEPA():

    trackers = trackerlist('tomp', 'got_jepa_378', range(1))

    ####################################

    dataset = get_dataset('avist')

    print("GOT_JEPA eva load")

    return trackers, dataset

    ####################
