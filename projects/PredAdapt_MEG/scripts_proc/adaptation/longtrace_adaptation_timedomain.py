import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import nibabel as nb
import nilearn as nl
from nilearn.glm.first_level.hemodynamic_models import _gamma_difference_hrf
import scipy as sp
from scipy import signal

## MAIN FUNCTIONS
def sequence_to_time(stim_sequence, stim_array, iti, time_step, hrf_array):
    """put sequence into time domain
     stim_sequence:  the stimulus sequence of interest
     stim_array:     the behaviour (e.g. rampup) of a single stimuli (length relative to time_step)
     iti:            the inter trial interval (length relative to time_step)
     time_step:      number of timesteps per array element (i.e. 1000 is 1/1000 > miliseconds)
     hrf_array:  the hrf array (set in same time domain)
    output: returns the adjusted stimuli sequence in time domain"""

    # calculate the trial length
    stim_length  = len(stim_array)
    trial_length = stim_length + iti

    # create empty array
    stims_t = np.zeros((len(stim_sequence)*trial_length)) # +len(hrf_array)) # includes padding of hrf
    hrf_t   = np.zeros(len(stims_t))

    # fill in 
    for stim in range(len(stim_sequence)): 
        stims_t[(stim*trial_length)+iti: (stim*trial_length)+trial_length] = stim_sequence[stim] * stim_array
    
    # return stim sequence
    return(stims_t)


def activation_to_time(activ_sequence, stim_array, iti, time_step, hrf_array):
    """put sequence into time domain
     activ_sequence: the activation sequence of interest
     stim_array:     the behaviour (e.g. rampup) of a single stimuli (length relative to time_step)
     iti:            the inter trial interval (length relative to time_step)
     time_step:      number of timesteps per array element (i.e. 1000 is 1/1000 > miliseconds)
     hrf_array:  the hrf array (set in same time domain)
    output: returns the adjusted stimuli sequence in time domain"""

    # calculate the trial length
    stim_length  = len(stim_array)
    trial_length = stim_length + iti

    # create empty array
    stims_t = np.zeros((len(activ_sequence)*trial_length)) #+len(hrf_array)) # includes padding of hrf
    hrf_t   = np.zeros(len(stims_t))

    # fill in 
    for stim in range(len(activ_sequence)): 
        stims_t[(stim*trial_length)+iti: (stim*trial_length)+trial_length] = activ_sequence[stim] * stim_array
        
    # return activation sequence
    return(stims_t)


def activation_dict_to_time(activ_dict, stim_array, iti, time_step, hrf_array):
    """loop over the dictionary and return an adjusted dictionary
    input, activ_dict:  
         activ_dict:     the activation dicionary 
         stim_array:     the behaviour (e.g. rampup) of a single stimuli (length relative to time_step)
         iti:            the inter trial interval (length relative to time_step)
         time_step:      number of timesteps per array element (i.e. 1000 is 1/1000 > miliseconds)
         hrf_array:  the hrf array (set in same time domain)
     return: adjusted dictionary"""
    
    # predefine dictionary
    activ_dict_t = {}
    
    for pref in activ_dict:
        
        # check if nested
        if isinstance(activ_dict[1], dict): 
            # nest the new dict
            activ_dict_t[pref] = {}
            for shrp in activ_dict[pref]:
                # populate with time domain data
                activ_dict_t[pref][shrp] = activation_to_time(activ_dict[pref][shrp], stim_array, iti, time_step, hrf_array)
        else: activ_dict_t[pref] = activation_to_time(activ_dict[pref], stim_array, iti, time_step, hrf_array)
            
    # return dictionary
    return(activ_dict_t)


## HELPER FUNCTIONS
def convolve(activation, hrf_array):
    """convolve an activation array by a hrf function"""
    return(signal.convolve(activation, hrf_array[0])[:-len(hrf_array[0])+1])


def convolve_dict(input_dict, hrf_array):
    """Convolve complete dictionary and return new convolved dictionary"""

    # set up convolution dictionary
    convolved_dict = {}

    # loop over prefferences and convolve every item in nested
    for pref in input_dict:
        # check if dictionary is nested and convolve all nested if needed
        if isinstance(input_dict[pref], dict):
            convolved_dict[pref] = {shrp: convolve(input_dict[pref][shrp], hrf_array) for shrp in input_dict[pref]}
        else:
            convolved_dict[pref] = convolve(input_dict[pref], hrf_array)
    return(convolved_dict)


def set_nan(input_dict, replace_value=0, replace_with=np.nan):
    """replace all values within a dictionary with nans 
    (or set some other value to replace/ to replace with)"""
    for pref in input_dict:
        for shrp in input_dict[pref]:
            input_dict[pref][shrp][input_dict[pref][shrp] == replace_value] = replace_with
    return(input_dict)


def hrf(tr, time_step):
    """get hrf function, input tr and timestep"""
    return(_gamma_difference_hrf(tr=tr ,oversampling=time_step, onset=-tr/2)[np.newaxis,:])


## STATS FUNCTIONS
def calc_correlations(dict_a, dict_b):
    """input two similarly shaped dictionaries (nested) calculate correlations 
    of arrays within dictionaries. e.g. dict_a: activation matrix, and dict_b: adaptation matrix"""
    
    # predefine correlation matrix
    correlations = np.zeros((len(dict_a), len(dict_a[list(dict_a.keys())[0]])))

    # calculate correlations
    for pref in range(len(dict_a)):
        correlations[pref, :] = [np.corrcoef(dict_a[list(dict_a.keys())[pref]][shrp], 
                                             dict_b[list(dict_b.keys())[pref]][shrp])[0][1] for shrp in dict_a[list(dict_a.keys())[pref]]]
    return(correlations)


def calc_stds(dict_a):
    """calculate standard deviation of nested dict"""
    
    # predefine correlation matrix
    stds = np.zeros((len(dict_a), len(dict_a[list(dict_a.keys())[0]])))

    # calculate correlations
    for pref in range(len(dict_a)):
        stds[pref, :] = [np.std(dict_a[list(dict_a.keys())[pref]][shrp]) for shrp in dict_a[list(dict_a.keys())[pref]]]
    return(stds)