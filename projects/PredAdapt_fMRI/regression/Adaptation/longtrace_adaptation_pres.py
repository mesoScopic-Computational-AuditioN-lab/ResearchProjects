import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from longtrace_adaptation_timedomain import *

## MAIN PRESENTATION FUNCTION
def plot_convolved_activation(activs, adpt_activs, hrf_array, stim_train, iti, time_step, display_len='full'):
    """plot the hrf convolved activation of adapted and non-adapted neuron
    input, activs:       activation sequence
           adpt_activs:  adapted activation sequence
           hrf_array:    hrf array (in same time domain as activs and adpt_activs)
           stim_train:   set stimulus sequence,
    optional, display_len:   set presesentation length in same time domain"""
    
    # set display length
    if display_len == 'full':
        display_len = len(activs)

    # convolve activations by hrf
    convolved_raw = np.convolve(activs, hrf_array)[:-len(hrf_array)+1]
    convolved_adp = np.convolve(adpt_activs, hrf_array)[:-len(hrf_array)+1]

    # configure plot size
    fig, ax = plt.subplots(3, 
                           1, 
                           sharex=True, 
                           figsize=(12, 10), 
                           gridspec_kw={'height_ratios': [3, 1, 1]})

    # plot activations, convolved activation, and stimuli train
    ax[0].plot(activs[:display_len], label='raw activation', alpha=0.8)
    ax[0].plot(adpt_activs[:display_len], label='adapted activation', alpha=0.8)
    ax[1].plot(convolved_raw[:display_len])
    ax[1].plot(convolved_adp[:display_len])
    ax[1].plot(convolved_raw[:display_len]-convolved_adp[:display_len], 
               label='difference', 
               alpha=0.8,
               lw=2,
               ls='--',
               color='grey')
    ax[2].scatter(np.arange(1,display_len+1,1) ,
                  sequence_to_time(stim_train, 
                                   np.ones(len(stim_train)), 
                                   iti, 
                                   time_step, 
                                   hrf_array)[:display_len])

    # pimp activations
    ax[0].legend(fontsize=16, loc=1)
    ax[0].set_ylabel('relative activation', fontsize=16)
    ax[0].tick_params(axis='y', which='major', labelsize=16)

    # pimp convolved activations
    ax[1].set_ylabel('gamma', fontsize=16)
    ax[1].tick_params(axis='y', which='major', labelsize=16)
    ax[1].legend(fontsize=16, loc=1)
    
    # pimp stimuli train
    ax[2].set_ylim(1, np.max(stim_train)+1)
    ax[2].set_ylabel('presented\nstim', fontsize=16)
    ax[2].tick_params(axis='y', which='major', labelsize=16)

    # pimp layout
    plt.xlabel('time (ms)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.suptitle('HRF convolved activation', fontsize=22)
    plt.tight_layout()

    return(ax)


def plot_activation_correlation(activation_dict, adaptation_dict):
    """plot correlation matrix of adapted and non adapted activation"""
    
    # calculate the correlations
    correlations = calc_correlations(activation_dict, adaptation_dict)

    # set image settings
    fig, ax = plt.subplots(figsize=(18, 12))

    # plot heat map
    ax = sns.heatmap(correlations)

    # set correct x/y ticks
    ax.set_yticklabels(list(activation_dict.keys()))
    ax.set_xticklabels(list(activation_dict[list(activation_dict.keys())[0]].keys()))

    # pimp graph
    plt.ylabel('tuning prefference', fontsize=16)
    plt.yticks(fontsize=12)
    plt.xlabel('tuning sharpness', fontsize=16)
    plt.xticks(fontsize=12)
    plt.title('Correlation matrix\nAdapted vs non-adapted activation', fontsize=22)
    
    # display only some labels
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 == 0]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 2 != 0]
    return(ax)


def plot_stds(activations):
    """plot standard deviations of nested dict"""
    
    # calculate the correlations
    stds = calc_stds(activations)

    # set image settings
    fig, ax = plt.subplots(figsize=(18, 12))

    # plot heat map
    ax = sns.heatmap(stds)

    # set correct x/y ticks
    ax.set_yticklabels(list(activations.keys()))
    ax.set_xticklabels(list(activations[list(activations.keys())[0]].keys()))

    # pimp graph
    plt.ylabel('tuning prefference', fontsize=16)
    plt.yticks(fontsize=12)
    plt.xlabel('tuning sharpness', fontsize=16)
    plt.xticks(fontsize=12)
    plt.title('Standard deviation\ngiven tuning prefference and sharpness', fontsize=22)

    # display only some labels
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 == 0]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 2 != 0]
    return(ax)
