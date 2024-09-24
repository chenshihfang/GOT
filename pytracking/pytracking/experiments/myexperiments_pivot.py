from pytracking.evaluation import Tracker, get_dataset, trackerlist
import torch
import numpy as np
import random
import os

# conda activate got_pivot

# CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments_pivot pivot --debug 0 --threads 2

def pivot():

    trackers = trackerlist('tomp', 'pivotL27', range(1))

    ####################################

    dataset = get_dataset('avist')
    # dataset = get_dataset('nfs')

    print("pivot eva load")

    return trackers, dataset

    ####################
