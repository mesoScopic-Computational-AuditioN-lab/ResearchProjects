#!/usr/bin/env python
# coding: utf-8

# # Long Trace PRF - Forward Simulation
# Here I have created a short demonstration of neuron behaviour under influence of a long trace stimulus history effect (i.e. adaptation).  The idea of this long trace model was adopted from:
# 
# + Fritsche, M., Solomon, S. G., & de Lange, F. P. (2021). Brief stimuli cast a long-term trace in visual cortex. bioRxiv.
# 
# Models are implemented for auditory perception, but since this is a toy simulation can easily be adopted to nearly all neuron prefferences.

import numpy as np
import pandas as pd
import scipy.stats as stats

# Main functions

def stim_adaptation(stim_train, y_decay):
    """Input a stimulus train of some length and n-back decay array,
    then return per stimulus adaptation (multiplicative and full array)
    input: stim_train: the n long stim train
           y_decay: n-back array from decay function
    returns: total_adapt: multiplicative adaptation per stim [length stim train]
             adapt_matrix: complete adaptation matrix [length stim train  *  number of N-backs]"""

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


def decay_step(prev_activation, a, t, res_adaptation, exp_explosion=0.99):
    """Take two components, previous residual decay and previous activity
    input: prev_activation: n-1 activation
           a: array of [afast, aslow]
           t: array of [tfast, tslow]
           res_adaptation: residual adaptation leftover,
    optional in: exp_eplosion: default 0.99 - to make sure that for 
                 long stim trains adaptation does not goes to infitity
    output: return current trial adaptation"""
    
    # Small check to reset res decay if it goes above 98,5% 
    # This is not necesarry but, for long chains (in the limit) any kind of prev_activation would be impossible
    if exp_explosion != False:
        if (1-res_adaptation.sum()) > exp_explosion: res_adaptation = np.zeros(2)
    
    # set empty array for N-1 adaptation
    n_1_adaptation = np.zeros(len(a))
    adaptation = np.zeros(len(a))
    
    # loop over the kinds of adaptation (in this case just fast and slow but can be expended)
    # ([0] = fast, [1] = slow)
    for i in range(len(a)):

        # set new adaptation and decay residual adaptation
        n_1_adaptation[i] = prev_activation * a[i]                   # calculate n-1 adaptation
        res_adaptation[i] = res_adaptation[i] * np.exp(-(1/t[i]))   # decay prev residual

        # add new to decayed residual adaptation
        adaptation[i] = n_1_adaptation[i] + res_adaptation[i]       # add current n-1 decay
        
    return(adaptation)


# Helper functions

def log_trans(x):
    return( np.log10(x) )


def exp_trans(x):
    return( 10**x)


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

    return(return_x, return_y) # return x and y array


def gaussian_point(tun_pref, tun_sharp, x, tun_peak=1):
    """Get y / hight on gaussian for a certain x
    input: tun_pref=tuning prefference
           tun_sharp=tuning sharpness
           x=x to calculate
    return: y value for that x"""
    return(tun_peak * np.exp(-((x-tun_pref)**2)/(2*tun_sharp**2)))


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


def calc_activation(activation, adaptation):
    return((1-adaptation) * activation)



