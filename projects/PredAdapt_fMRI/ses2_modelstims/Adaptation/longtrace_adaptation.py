import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import pickle
import os

## MULTI DIMENSIONAL FUNCTIONS FOR FAST ADAPTATION PROCESSING
def md_get_tuning(lin_idx, pref_range, sharp_range):
    """from linear index get tuning prefference and tuning width"""
    
    # get repmatrix for tuning pref and sharpness
    md_tunprefs = np.tile(pref_range, len(sharp_range))    # e.g. [123123123]
    md_tunsharps = np.repeat(sharp_range, len(pref_range)) # e.g. [111222333]

    # calculate the index
    return(md_tunprefs[lin_idx], md_tunsharps[lin_idx])

def md_get_linidx(tun_pref, tun_sharp, pref_range, sharp_range, precision=None):
    """from tuning prefference and tuning sharpness (and full range), 
    if precicion is specifies, round to precision, else assume
    input precision of tuning prefference and tuning sharpness.
    return linear index for tuning"""
    # calculate or take input precision
    if precision: 
        pref_prec = precision
        sharp_prec = precision
    else: 
        pref_prec = str(tun_pref)[::-1].find('.')
        sharp_prec = str(tun_sharp)[::-1].find('.')
    
    # calculate the index
    tun_pref = np.where(np.round(pref_range,pref_prec) == np.round(tun_pref, pref_prec))[0][0]
    tun_sharp = np.where(np.round(sharp_range,sharp_prec) == np.round(tun_sharp, sharp_prec))[0][0]
    print(f'tuning pref idx: {tun_pref}, tuning sharpness idx: {tun_sharp}')
    
    # get linear index from both positions
    return( md_get_linidx_pos(tun_pref, tun_sharp, pref_range) )

def md_get_linidx_pos(tunpref_idx, tunsharp_idx, pref_range):
    """input tuning prefference index, sharpness index and the tuning prefference range
    returns: linear index for multidimensional ([pref*width, stims]) array"""
    return(tunpref_idx + len(pref_range)*tunsharp_idx)

def md_stim_adaptation(stim_train, y_decay, 
                    tun_prefs, tun_sharps, tun_peak=1):
    """Input a stimulus train of some length and n-back decay array,
    then return per stimulus adaptation (multiplicative and full array)
    input: stim_train: the n long stim train
           y_decay: n-back array from decay function
           tun_prefs: a list of tuning prefferences
           tun_sharps: a list of tuning widths
    returns: total_adapt: multiplicative adaptation per stim [length stim train]
             adapt_matrix: complete adaptation matrix [length stim train  *  number of N-backs]"""
    
    # set up matrixes
    total_adapt = np.zeros([len(tun_prefs)*len(tun_sharps), 
                            len(stim_train)])                    # make empty array of stim train length
    n_back_array = np.zeros([len(tun_prefs)*len(tun_sharps), 
                               len(stim_train), len(y_decay)+1]) # zet activation array
    
    # caclulate raw (no adaptation, activations) / exitability
    activations = md_gaussian_activations(tun_prefs, tun_sharps, stim_train, tun_peak=tun_peak)
    
    # loop over number of n_back steps and fill in n_back array 
    for nback in range(len(y_decay)+1):
        n_back_array[:,:,nback] = np.roll(np.append(np.zeros((activations.shape[0],
                                                             len(y_decay))), 
                                                    activations, axis=1), nback, axis=1)[:,len(y_decay):]

    # calculate the activation and multiply for overal adaptation
    n_back_array = 1-(n_back_array[:,:,1:] * (1-y_decay))   # full length by N-backs array (same as new n_back)
    total_adapt = np.prod(n_back_array, axis=-1)            # multiplicative adaptation

    return(total_adapt, n_back_array)


def md_gaussian_activations(tun_prefs, tun_sharps, x_array, tun_peak=1):
    """Get y / hight on gaussian for a certain x array
    input: tun_pref=tuning prefference
           tun_sharp=tuning sharpness
           x=x array to calculate
    return: y value for that x"""
    
    # make x_array match tuning pref dimensions, and repeat for matrix calc
    if x_array.ndim < 2: x_array = np.repeat(x_array[:, np.newaxis], 
                                             len(tun_prefs)*len(tun_sharps), 
                                             axis=1)
        
    # get repmatrix for tuning pref and sharpness for single calculatie
    md_tunprefs = np.tile(tun_prefs, len(tun_sharps))    # e.g. [123123123]
    md_tunsharps = np.repeat(tun_sharps, len(tun_prefs)) # e.g. [111222333]
        
    # calculate in one go
    return((tun_peak * (np.exp(-((x_array-md_tunprefs)**2)/(2*md_tunsharps**2)))).transpose())


def md_plot_activationadaptation(activations, adaptations, adapted_activations):
    """input activation, adaptation and adaptated activation matrix, plot heatmap"""
    fig, ax = plt.subplots(3, 1, figsize=(12, 8) , sharex=True)

    ax[0].imshow(activations.transpose())
    ax[0].tick_params(axis='y', which='major', labelsize=18)               # set ticksizes x 
    ax[0].set_ylabel('Stimuli', fontsize=18) 

    ax[1].imshow(adaptations.transpose())
    ax[1].tick_params(axis='y', which='major', labelsize=18)               # set ticksizes x 
    ax[1].set_ylabel('Stimuli', fontsize=18) 

    ax[2].imshow(adapted_activations.transpose())
    ax[2].tick_params(axis='y', which='major', labelsize=18)               # set ticksizes x 
    ax[2].set_xlabel('Lin-idx ([prefs*tw])', fontsize=18)
    ax[2].set_ylabel('Stimuli', fontsize=18) 

    # pimp plot
    plt.xticks(fontsize=18)
    plt.suptitle(f'Activation and adaptation behaviour', fontsize=26)
    plt.tight_layout()
    return(ax, fig)

## MAIN ADAPTATION FUNCTIONS

def stim_adaptation(stim_train, y_decay, 
                    tun_pref, tun_sharp, tun_peak=1):
    """Input a stimulus train of some length and n-back decay array,
    then return per stimulus adaptation (multiplicative and full array)
    input: stim_train: the n long stim train
           y_decay: n-back array from decay function
    returns: total_adapt: multiplicative adaptation per stim [length stim train]
             adapt_matrix: complete adaptation matrix [length stim train  *  number of N-backs]"""
    
    # set up matrixes
    total_adapt = np.zeros(len(stim_train))                    # make empty array of stim train length
    n_back_array  =  np.zeros([len(stim_train), len(y_decay)+1])  # zet activation array
    
    # caclulate raw (no adaptation, activations) / exitability
    activations = gaussian_activations(tun_pref, tun_sharp, stim_train, tun_peak=tun_peak)
    
    # loop over number of n_back steps and fill in n_back array 
    for nback in range(len(y_decay)+1):
        n_back_array[:,nback] = np.roll(np.append(np.zeros(len(y_decay)), activations), nback)[len(y_decay):]
        
    # calculate the activation and multiply for overal adaptation
    n_back_array = 1-(n_back_array[:,1:] * (1-y_decay))   # full length by N-backs array (same as new n_back)
    total_adapt = np.prod(n_back_array, axis=1)           # multiplicative adaptation

    return(total_adapt, n_back_array)


def create_adaptation_dict(stim_train, y_decay, pref_range, sharp_range):
    """Input a stimulus train of some length, a n_back decay array, 
    the prefference (neuron) array range, the sharpness array range
    input:  stim_train: the n long stim train
            y_decay: n-back array from decay function
            pref_range: an array with the prefferences of specific neurons 
                        (to get adaptation in that situation)
            sharp_range: an array with the sharpnesses of the pref_range neurons
    returns: the adaptation dictonary with adapt_dict[prefferences][sharpnesses]"""
    
    # predefine a dictionary to store all prefferences and sharpnesses
    adapt_dict = {}
    
    # loop over prefferences
    for pref in pref_range:
        adapt_dict[pref] = {}    # nest dictonary to store sharpness
        # loop over sharpnesses
        for sharp in sharp_range:
            # run main stim_adaptation function
            adapt_dict[pref][sharp], _ = stim_adaptation(stim_train, y_decay, pref, sharp)
    return(adapt_dict)


def create_activation_dict(stim_train, pref_range, sharp_range, tun_peak=1):
    """Input stimulus train of some length, the prefference (neuron) array range, 
    the sharpness array range, (optional the tuning peak). returns raw activation dictionary"""
    
    # predefine a dictionary to store all raw activation
    act_dict = {}
    
    for pref in pref_range:
        act_dict[pref] = {}    # nest dictonary to store sharpness
        # loop over sharpnesses
        for sharp in sharp_range:
            # run main stim_adaptation function
            act_dict[pref][sharp] = gaussian_activations(pref, sharp, stim_train, tun_peak=tun_peak)
    return(act_dict)
    
                           
def create_adapted_activation_dict(act_dict, adapt_dict):
    """Input activation and adaptation dictionary, and return weighted activation dict
    both dictionaries should contain same keys"""

    # predefine dictionary for new adapted activation
    adp_act_dict = {}
    
    # loop over prefferences and then do list comprehension multiplication for all keys in dict
    for pref in act_dict:
        adp_act_dict[pref] = {shrp: act_dict[pref][shrp]*adapt_dict[pref][shrp] for shrp in act_dict[pref]}
    return(adp_act_dict)
    

def relative_adaptation(adapt_dict):
    """Calculate the relative adaptation away from 0,
    return adjusted dictionary"""
    
    # set up relative adaptation dict
    rel_adapt_dict = {}
    
    # loop over prefferences and calculate relative adaptation from 0
    for pref in adapt_dict:
        rel_adapt_dict[pref] = {shrp: (1-adapt_dict[pref][shrp]) for shrp in adapt_dict[pref]}
    return(rel_adapt_dict)


## FUNCTIONS FOR GAUSSIAN 

def gaussian_func(tun_pref, tun_sharp, xlims, stepsize, tun_peak=1):
    """input parameters of gaussian function
    input:  tun_pref=tuning prefference
            tun_sharp=tuning sharpness
            xlim=bounds of tuning
            stepsize=step size of x
    optional in: tun_peak=the peak of the tuning
    output: return np.array for x and y
    """
    # define x and y arrays
    return_x = np.arange(xlims[0], xlims[1]+stepsize, stepsize)
    return_y = np.zeros(return_x.shape[0])
    
    # get y value for each step x
    for x in range(return_x.shape[0]):
        return_y[x] = gaussian_point(tun_pref, tun_sharp, return_x[x], tun_peak=tun_peak)

    # return x and y array
    return(return_x, return_y) 


def gaussian_point(tun_pref, tun_sharp, x, tun_peak=1):
    """Get y / hight on gaussian for a certain x
    input: tun_pref=tuning prefference
           tun_sharp=tuning sharpness
           x=x to calculate
    return: y value for that x"""
    return(tun_peak * np.exp(-((x-tun_pref)**2)/(2*tun_sharp**2)))


def gaussian_activations(tun_pref, tun_sharp, x_array, tun_peak=1):
    """Get y / hight on gaussian for a certain x array
    input: tun_pref=tuning prefference
           tun_sharp=tuning sharpness
           x=x array to calculate
    return: y value for that x"""
    return(tun_peak * np.exp(-((x_array-tun_pref)**2)/(2*tun_sharp**2)))


## DOUBLE EXPONENTIAL DECAY FUNCTION

def double_exp_decay_func(afast, tfast, aslow, tslow, xlims, stepsize):
    """input parameters of double exponential decay and returns and x and y array
    input:  afast: magnitude of fast adaptation
            tfast: recovery of fast adaptation
            aslow: magnitde of slow adaptation
            tslow: recovery of slow adaptation
    output: return np.array for x and y
    """
    # define x and y arrays
    return_x = np.arange(xlims[0], xlims[1]+stepsize, stepsize)
    return_y = np.zeros(return_x.shape[0])
    
    # get y value for each step x
    for x in range(return_x.shape[0]):
        return_y[x] = 1 - (afast * np.exp(-(return_x[x]-1)/tfast)) - (aslow * np.exp(-(return_x[x]-1)/tslow))
    
    return(return_x, return_y) # return x and y array


## SUPPORTER FUNCTIONS

def log_trans(x):
    return( np.log10(x) )

def exp_trans(x):
    return( 10**x)

def calc_adapted_activation(adaptation, activation):
    """calculate the adapted activation"""
    return(adaptation * activation)


## PLOTTING FUNCTIONS

def plot_decay(x_decay, y_decay):
    """plot/visualize the decay function"""

    # set figure size and plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = plt.plot(x_decay, y_decay, lw=5)

    # pimp plot
    plt.title("Double decay adaptation by neurons", fontsize=20)
    plt.ylabel("Adaptation", fontsize=16)
    plt.xlabel("N-back", fontsize=16)
    plt.axhline(1, color='grey', lw=2.5, ls='--')
    plt.ylim([0.80, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return(ax)


def plot_adaptation(tun_pref, tun_sharp, stim_train, y_decay, tun_peak=1):
    """plot/visualize the decay function"""

    # calculate activation, adaptation and adaptation weighted activation
    raw_act     = gaussian_activations(tun_pref, tun_sharp, stim_train, tun_peak=tun_peak)
    adapt, _    = stim_adaptation(stim_train, y_decay, tun_pref, tun_sharp)
    adapted_act = calc_adapted_activation(adapt, raw_act)
    
    # set figure size and plot
    fig, ax = plt.subplots(2, 
                           1, 
                           sharex=True, 
                           figsize=(12, 8), 
                           gridspec_kw={'height_ratios': [3, 1]})
    
    # plot activations
    ax[0].plot(raw_act,     lw=2.5, alpha=0.8, label='non-adapted activation')
    ax[0].plot(adapted_act, lw=2.5, alpha=0.8, label='adapted activation')
    
    # plot adaptation
    ax[1].plot(adapt, lw=2.5)

    # pimp activation
    ax[0].legend(fontsize=16, loc=1)
    ax[0].set_ylabel('Activation', fontsize=16)
    ax[0].tick_params(axis='y', which='major', labelsize=16)
    ax[0].set_title('Adapted and non-adapted activation over stimuli', fontsize=18)
    
    # pimp adaptation
    ax[1].set_ylabel('Adaptation', fontsize=16)
    ax[1].tick_params(axis='y', which='major', labelsize=16)
    ax[1].set_title('Neuron adaptation', fontsize=18)
    ax[1].axhline(1, color='grey', lw=2, ls='--', alpha=0.5)
    
    # pimp layout
    plt.xlabel('Trial nr.', fontsize=16)
    plt.xticks(fontsize=16)
    plt.suptitle('Stimuli adaptation', fontsize=22)
    plt.tight_layout()
    
    return(ax)


def plot_adaptation_corr(tun_pref, tun_sharp, stim_train, y_decay, tun_peak=1):
    """plot/visualize the correlation matrix of adapted vs non-adapted activation"""

    # calculate activation, adaptation and adaptation weighted activation
    raw_act     = gaussian_activations(tun_pref, tun_sharp, stim_train, tun_peak=tun_peak)
    adapt, _    = stim_adaptation(stim_train, y_decay, tun_pref, tun_sharp)
    adapted_act = calc_adapted_activation(adapt, raw_act)

    # put adapted and non adapted activation into a dictionary
    data = {'Raw activation': raw_act,
            'Adapted activation': adapted_act}

    # get dataframe and get correlation matrix
    df = pd.DataFrame(data, columns=['Raw activation', 'Adapted activation'])
    corrMatrix = df.corr()

    # set figure size 
    fig, ax = plt.subplots(figsize=(12, 8))

    # make heatmap
    ax = sns.heatmap(corrMatrix, annot=True, fmt='.4%', center=.95, annot_kws={"size": 16})

    # pimp layout
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 18, rotation=0)
    plt.suptitle('Correlation matrix,\nadapted vs non-adapted activation', fontsize=22)
    plt.tight_layout()

    return(ax)


# SAVE FUNCTIONS
def save_dict(dict_name, filename):
    """use pickle to save a dictionary into a folder"""
    # make directory if not exist
    os.makedirs(os.path.dirname('pickle/'), exist_ok=True)
    # and then save
    return(pickle.dump( dict_name, 
                        open("pickle/{}.p".format(filename), "wb")))

def load_dict(filename):
    """use pickle to load a dictionary from a folder"""
    return(pickle.load( open("pickle/{}.p".format(filename), "rb")))


## LEGACY FUNCTIONS

def stim_adaptation_binary(stim_train, y_decay):
    """Input a stimulus train of some length and n-back decay array,
    then return per stimulus adaptation (multiplicative and full array)
    input: stim_train: the n long stim train
           y_decay: n-back array from decay function
    returns: total_adapt: multiplicative adaptation per stim [length stim train]
             adapt_matrix: complete adaptation matrix [length stim train  *  number of N-backs]
    note** this is a legacy function, relying on a boolean activation arrray (1/0)"""

    adapt_matrix = np.zeros([len(stim_train), len(y_decay)])   # make empty matrix of train length by number of N-backs 
    total_adapt = np.zeros(len(stim_train))                    # make empty array of stim train length

    # loop over stimuli starting at N+1
    for stim in range(0, len(stim_train)):

        # create a temp nback array to indicate adoptation
        n_back_array = np.zeros(len(y_decay))

        # for the first few stimulus (to avoid wrapping)
        if stim <= len(y_decay): 
            # populate Nback array with boolean, starting from right most (n-1)
            n_back_array[:stim] = (stim_train[:stim] == stim_train[stim]).astype(int)[::-1]
            
        else: 
            # populate Nback array with boolean, starting from right most (n-1)
            n_back_array[:] = (stim_train[stim-len(y_decay):stim] == stim_train[stim]).astype(int)[::-1]

        # populate our return arrays
        n_back_array *= y_decay                                      # multiply boolean array by decay array (1 encoding) 
        adapt_matrix[stim,:] = n_back_array                          # full length by N-backs array
        total_adapt[stim] = np.prod(n_back_array[n_back_array > 0])  # multiplicative adaptation

    return(total_adapt, adapt_matrix)