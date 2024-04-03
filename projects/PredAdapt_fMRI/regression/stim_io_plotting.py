## series of functions to load matlab data and parse them into usebable formats

import scipy.io
import pandas as pd
import numpy as np
import re

import os
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns

import Adaptation.longtrace_adaptation as longtrace_adaptation
import Adaptation.longtrace_adaptation_timedomain as longtrace_adaptation_timedomain

from stim_io import *

# load current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

## main functions
import warnings

def plot_design_mat(tr_df, all_freqs, pref1, tw1, pref2, tw2, runs=1, prefixes=None):
    """plot the desing matrix for two tuning functions specified"""
    warnings.filterwarnings('ignore')
    
    # check if runs in list, else place in list
    runs = [runs] if isinstance(runs, int) else runs

    # create subplots
    fig, ax = plt.subplots(2,
                           2,
                           figsize=(13,13), gridspec_kw={'height_ratios': [2, 3]})

    # make x the full range
    x = all_freqs

    # calculate the normal function for all frequency points
    y1 = gauss(x, pref1, tw1)
    y2 = gauss(x, pref2, tw2)

    # plot the first gaussian functions
    ax[0, 0].plot(x,y1, color='darkgreen', lw=3)

    # pimp the first gaussian function
    ax[0, 0].tick_params(axis='x', which='major', labelsize=18)               # set ticksizes x 
    ax[0, 0].tick_params(axis='y', which='major', labelsize=18)               # and y
    ax[0, 0].set_ylabel(f'Activation', fontsize=18) 
    ax[0, 0].set_xlabel(f'Freq - oct', fontsize=18) 
    ax[0, 0].spines['top'].set_visible(False)
    ax[0, 0].spines['right'].set_visible(False)
    ax[0, 0].set_title(f'Freq: {2**pref1:.1f}kHz\nTW-FWHM: {tw1 * 2.354:.2f}oct', fontsize=22)

    # plot the second gaussian fucntion
    ax[0, 1].plot(x,y2, color='darkgreen', lw=3)

    # pimp the second gaussian function
    ax[0, 1].tick_params(axis='x', which='major', labelsize=18)               # and y
    ax[0, 1].set_xlabel(f'Freq - oct', fontsize=18) 
    ax[0, 1].axes.get_yaxis().set_visible(False)
    ax[0, 1].spines['top'].set_visible(False)
    ax[0, 1].spines['right'].set_visible(False)
    ax[0, 1].spines['left'].set_visible(False)
    ax[0, 1].set_title(f'Freq: {2**pref2:.1f}kHz\nTW-FWHM: {tw2 * 2.354:.2f}oct', fontsize=22)

    # from the tuning width and tuning pref get the columns of interest
    colls = get_tw_collumns(tr_df, pref1, tw1, convolved=True)
    del colls[-2]                    # remove adapted activation
    colls += ['onoff']     # add onoff

    # prune collumns in wanted
    if prefixes:
        # create prefix pattern
        pattern = r'^(' + '|'.join(prefixes) + r')'
        # sellect collumns
        colls = [colls for colls in colls if re.match(pattern, colls)]
    
    # plot the first heatmap 
    sns.heatmap(normalize(tr_df[colls][tr_df['run'].isin(runs)]),cmap="crest", cbar=False, 
                ax=ax[1,0])

    # pimp the first heat map
    ax[1, 0].set_ylabel(f'Trial', fontsize=18) 
    ax[1, 0].tick_params(axis='x', which='major', labelsize=14)               # set ticksizes x 
    ax[1, 0].axes.yaxis.set_ticklabels([])

    # from the second tuning widt and tuning pref get the columns of interest
    colls = get_tw_collumns(tr_df, pref2, tw2, convolved=True)
    del colls[-2]                    # remove adapted activation
    colls += ['onoff']     # add onoff
    
    # prune collumns in wanted
    if prefixes:
        # create prefix pattern
        pattern = r'^(' + '|'.join(prefixes) + r')'
        # sellect collumns
        colls = [colls for colls in colls if re.match(pattern, colls)]

    # plot the second heatmap
    sns.heatmap(normalize(tr_df[colls][tr_df['run'].isin(runs)]),cmap="crest", cbar=False, 
                ax=ax[1,1])

    # pimp the second heat map
    ax[1, 1].axes.get_yaxis().set_visible(False)
    ax[1, 1].tick_params(axis='x', which='major', labelsize=14)               # set ticksizes x 
    fig.tight_layout()

    plt.plot()
    return(ax,fig)


def data_plot(mat, stimuli, blocknr=1, octvspace=False):
    """plot important aspects of raw data 
    optionally plot specific blocknr"""
    ## PREPAIRING DATA
    # where block is 1
    idxblock = np.where(mat['timingz'][1] == blocknr) 

    #get frequency presentation data for block
    frequencies = stimuli['pres_freq'][blocknr-1, :]

    # other values
    tps = np.sum(mat['timingz'][3, idxblock] == 1) # get trials per secion

    #get timings back from mat file, substract begin time
    timings = mat['timingz'][4, idxblock] - np.min(mat['timingz'][4, idxblock]) 

    matidx = np.where(mat['segmentz'][1] == blocknr)

    centa = 2**np.repeat(mat['segmentz'][7][matidx], tps)   # cent freq a
    centb = 2**np.repeat(mat['segmentz'][8][matidx], tps)  # cent freq b
    proba = np.repeat(mat['segmentz'][5][matidx], tps)  # prob a
    probb = np.repeat(mat['segmentz'][6][matidx], tps)  # prob b
    
    ## ACTUAL PLOTTING
    # senatry check the data
    fig, ax = plt.subplots(2, 
                           1, 
                           sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]}, figsize=(10,  7.5))

    # octave transform if wanted
    if octvspace: 
        frequencies = np.log2(frequencies)
        centa = np.log2(centa)
        centb = np.log2(centb)
    
    ax[0].scatter(timings, frequencies, color='darkslategrey', alpha=0.8)
    ax[0].axhline(y=centa[0], color='darkred', linestyle='-', alpha=0.5, ls='--', lw=4)
    ax[0].axhline(y=centb[0], color='darkred', linestyle='-', alpha=0.5, ls='--', lw=4)
    ax[0].tick_params(axis='x', which='major', labelsize=18)               # set ticksizes x 
    ax[0].tick_params(axis='y', which='major', labelsize=18)               # and y
    if octvspace: space = 'octaves'
    else: space = 'Hz'
    ax[0].set_ylabel(f'Frequencies - in {space}', fontsize=18) 
    
    #ax[1].plot(timings[0], proba, color='r')
    ax[1].plot(timings[0], probb, color='darkolivegreen', lw=8)
    ax[1].tick_params(axis='x', which='major', labelsize=18)               # set ticksizes x 
    ax[1].tick_params(axis='y', which='major', labelsize=18)               # and y
    ax[1].set_xlabel('Volume', fontsize=18)
    ax[1].set_ylabel('Prob - top', fontsize=18) 
    
    # pimp plot
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.suptitle(f'Stimuli over block {blocknr}', fontsize=26)
    plt.tight_layout()
    return(ax, fig)

def stim_plot(stim_df):
    ## ACTUAL PLOTTING
    # senatry check the data
    fig, ax = plt.subplots(2, 
                           1, 
                           sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]}, figsize=(10,  7.5))

    ax[0].scatter(stim_df['timing'], stim_df['frequencies_oct'], color='darkslategrey', alpha=0.8)

    ax[0].plot(stim_df['timing'], stim_df['center_freq_a_oct'], linestyle='-', alpha=0.5, lw=4)
    ax[0].plot(stim_df['timing'], stim_df['center_freq_b_oct'], linestyle='-', alpha=0.5, lw=4)

    ax[0].tick_params(axis='x', which='major', labelsize=18)               # set ticksizes x 
    ax[0].tick_params(axis='y', which='major', labelsize=18)               # and y
    ax[0].set_ylabel(f'Frequencies (oct)', fontsize=18) 

    #ax[1].plot(timings[0], proba, color='r')
    ax[1].plot(stim_df['timing'], stim_df['probability_a'], color='darkolivegreen', lw=8)
    ax[1].tick_params(axis='x', which='major', labelsize=18)               # set ticksizes x 
    ax[1].tick_params(axis='y', which='major', labelsize=18)               # and y
    ax[1].set_xlabel('Stimulus nr.', fontsize=18)
    ax[1].set_ylabel('Prob - top', fontsize=18) 

    # pimp plot
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.suptitle(f'Stimuli', fontsize=26)
    plt.tight_layout()
    
    return(ax, fig)

def freqs_plot(pref_range, sharp_range):
    """given a sharpness range and a prefference range plot all gaussians"""

    fig, ax = plt.subplots(1, 
                           1, 
                           sharex=True, figsize=(12,  3))

    # predefine size of design matrix
    y_data = np.zeros((len(pref_range), len(pref_range)*len(sharp_range)))
    idx = 0
    
    # loop over tuning widths
    for tw in sharp_range:
        # loop over prefferences
        for pref in pref_range:
            y_data[:, idx] = gauss(pref_range, pref, tw)
            idx += 1
    plt.suptitle('Used Tuning Gaussians', fontsize=26)
    plt.tight_layout()
    plt.imshow(y_data)
    return(ax, fig)