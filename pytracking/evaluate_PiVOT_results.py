

import os 
import sys

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results, \
                                                            print_results_per_attribute, plot_attributes_radar
from pytracking.evaluation import Tracker, get_dataset, trackerlist, get_dataset_attributes

trackers = []


#####################

# trackers.extend(trackerlist('tomp','tomp50_raw',range(0,1),'ToMP-50'))
# trackers.extend(trackerlist('tomp','PiVOT_50',range(0,1),'PiVOT-50'))
# trackers.extend(trackerlist('tomp','PiVOT_L_22',range(0,1),'PiVOT-L-22'))
# trackers.extend(trackerlist('tomp','PiVOT_L_27',range(0,1),'PiVOT-L-27'))
# trackers.extend(trackerlist('tomp','pivotL27',range(0,1),'pivotL27_yours'))

trackers.extend(trackerlist('GOT_Edit','GOT_Edit',range(0,1),'GOT_Edit'))

#####################

skm=True
force=True
print_=True
plot=False
attr=False


list_=['avist','otb','lasot','nfs']


if 'nfs' in list_:
    dataset = get_dataset('nfs')

    if print_:
        print_results(trackers, dataset, 'nfs', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),   
        skip_missing_seq=skm, force_evaluation=force, exclude_invalid_frames=True)

    if plot:
        plot_results(trackers, dataset, 'nfs', merge_results=True, plot_types=('success', 'prec', 'norm_prec'), 
                    skip_missing_seq=skm, force_evaluation=force, plot_bin_gap=0.05, exclude_invalid_frames=True)


# # ################### otb

if 'otb' in list_:
    dataset = get_dataset('otb')

    if print_:
        print_results(trackers, dataset, 'otb', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),   
        skip_missing_seq=skm, force_evaluation=force, exclude_invalid_frames=True)

    if plot:
        plot_results(trackers, dataset, 'otb', merge_results=True, plot_types=('success', 'prec', 'norm_prec'), 
                    skip_missing_seq=skm, force_evaluation=force, plot_bin_gap=0.05, exclude_invalid_frames=True)

    if attr:
        print_results_per_attribute(trackers, get_dataset_attributes('otb', mode='short'), 'otb',
                                    merge_results=True, force_evaluation=force,
                                    skip_missing_seq=skm,
                                    exclude_invalid_frames=True)

        plot_attributes_radar(trackers,
                            get_dataset_attributes('otb', mode='long'), 'otb',
                            merge_results=True, force_evaluation=force,
                            skip_missing_seq=skm,
                            plot_opts=None, exclude_invalid_frames=True)


# ################### uav123

if 'uav' in list_:
    dataset = get_dataset('uav')

    if print_:
        print_results(trackers, dataset, 'uav', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),   
        skip_missing_seq=skm, force_evaluation=force, exclude_invalid_frames=True)

    if plot:
        plot_results(trackers, dataset, 'uav', merge_results=True, plot_types=('success', 'prec', 'norm_prec'), 
                    skip_missing_seq=skm, force_evaluation=force, plot_bin_gap=0.05, exclude_invalid_frames=True)

    if attr:
        print_results_per_attribute(trackers, get_dataset_attributes('uav', mode='short'), 'uav',
                                    merge_results=True, force_evaluation=force,
                                    skip_missing_seq=skm,
                                    exclude_invalid_frames=True)

        plot_attributes_radar(trackers,
                            get_dataset_attributes('uav', mode='long'), 'uav',
                            merge_results=True, force_evaluation=force,
                            skip_missing_seq=skm,
                            plot_opts=None, exclude_invalid_frames=True)

################### lasot

if 'lasot' in list_:
    dataset = get_dataset('lasot')

    if print_:
        print_results(trackers, dataset, 'lasot', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),   
        skip_missing_seq=skm, force_evaluation=force, exclude_invalid_frames=False)


    if plot:
        plot_results(trackers, dataset, 'lasot', merge_results=True, plot_types=('success', 'prec', 'norm_prec'), 
                    skip_missing_seq=False, force_evaluation=force, plot_bin_gap=0.05, exclude_invalid_frames=False)

    if attr:

        plot_attributes_radar(trackers,
                            get_dataset_attributes('lasot', mode='long'), 'lasot',
                            merge_results=True, force_evaluation=force,
                            skip_missing_seq=skm,
                            plot_opts=None, exclude_invalid_frames=False)

        print_results_per_attribute(trackers, get_dataset_attributes('lasot', mode='long'), 'lasot',
                                    merge_results=True, force_evaluation=force,
                                    skip_missing_seq=skm,
                                    exclude_invalid_frames=False)

################### avist

if 'avist' in list_:

    dataset = get_dataset('avist')

    if print_:
        print_results(trackers, dataset, 'avist', merge_results=True, plot_types=('success', 'prec', 'norm_prec'),   
        skip_missing_seq=skm, force_evaluation=force, exclude_invalid_frames=False)

    if plot:
        plot_results(trackers, dataset, 'avist', merge_results=True, plot_types=('success', 'prec', 'norm_prec'), 
                    skip_missing_seq=False, force_evaluation=force, plot_bin_gap=0.05, exclude_invalid_frames=False)

    if attr:
        print_results_per_attribute(trackers, get_dataset_attributes('avist', mode='long'), 'avist',
                                    merge_results=True, force_evaluation=force,
                                    skip_missing_seq=skm,
                                    exclude_invalid_frames=False)

        plot_attributes_radar(trackers,
                            get_dataset_attributes('avist', mode='long'), 'avist',
                            merge_results=True, force_evaluation=force,
                            skip_missing_seq=skm,
                            plot_opts=None, exclude_invalid_frames=False)

