""" Fuctions for model fitting and model simulation of the Hierarchical Gaussian Filter
code takes in stimulus states and simulates perception and prediction of an agent

Model implemented as discribed in: Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston, K. J., & Stephan, K. E. (2014). Uncertainty in perception and the Hierarchical Gaussian Filter. Frontiers in human neuroscience, 8, 825.

Code adapted by Jorie van Haren (2021) """

# load nessecary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load extra (non exclusive) helper function
def _sgm(x, a):
    return(np.divide(a,1+np.exp(-x)))
## Construct dataframe function

def calc_correlations(dict_a, dict_b):
    """Input two similarly shaped dictionaries (nested) calculate correlations 
    of arrays within dictionaries. e.g. dict_a: activation matrix, and dict_b: adaptation matrix"""
    
    # predefine correlation matrix
    correlations = np.zeros((len(dict_a), len(dict_a[list(dict_a.keys())[0]])))

    # calculate correlations
    for pref in range(len(dict_a)):
        correlations[pref, :] = [np.corrcoef(dict_a[list(dict_a.keys())[pref]][shrp], 
                                             dict_b[list(dict_b.keys())[pref]][shrp])[0][1] for shrp in dict_a[list(dict_a.keys())[pref]]]
    return(correlations)


def create_surprisal_dict(pref_range, r, normalize=False, normalize_axis=None):
    """Input prefference range of some neuron, and dictionary r from HGF
    returns ordered dictionary"""

    # predefine a dictionary to store surprisals
    surprise_dict = {}

    # normalize
    if normalize: r['lls']['logll_matrix'] = normalize_mat(r['lls']['logll_matrix'], axis=normalize_axis)
    
    # loop over prefferences and store
    for pref in range(len(pref_range)):
        surprise_dict[pref_range[pref]] = r['lls']['logll_matrix'][pref,:]
    return(surprise_dict)


def create_probability_dict(surprisal_dict):
    """Input surprisal dict and log-transform it to get propability or
    likelyhood of neuron activation"""
    
    # predefine a dictionary to store probabilities
    probability_dict = {}
    
    # loop over prefferences and log transform
    for pref in surprisal_dict:
        probability_dict[pref] = np.exp(-surprisal_dict[pref])
    return(probability_dict)


def create_surprisal_deeper_dict(pref_range, r, lvl, normalize=True, normalize_axis=None):
    """Input prefference range of some neuron, and dictionary r from HGF
    returns ordered dictionary"""

    # predefine a dictionary to store surprisals
    surprise_dict = {}

    # normalize
    if normalize: r['lls']['logll_matrix_lvl{}'.format(lvl)] = normalize_mat(r['lls']['logll_matrix_lvl{}'.format(lvl)],
                                                                             axis=normalize_axis)
    
    # loop over prefferences and store
    for pref in range(len(pref_range)):
        surprise_dict[pref_range[pref]] = r['lls']['logll_matrix_lvl{}'.format(lvl)][pref,:]
    return(surprise_dict)


def normalize_mat(arr, axis=None):
    """Input an array to normalize,
    optional: define a axis over what to normalize, standard over everything
    return normalized array"""
    
    # calculate normalized array
    return((arr - np.min(arr, axis=axis))  /  (np.max(arr, axis=axis) - np.min(arr, axis=axis)))


def create_suprisal_activation_dict(act_dict, surprisal_dict):
    """Input activation and surprisal dictionary, and return weighted activation dict
    activation dictionary will have sharpness parameters where surprisal does not"""

    # predefine dictionary for new adapted activation
    spr_act_dict = {}
    
    # loop over prefferences and then do list comprehension multiplication for all keys in dict
    for pref in act_dict:
        spr_act_dict[pref] = {shrp: act_dict[pref][shrp]*surprisal_dict[pref] for shrp in act_dict[pref]}
    return(spr_act_dict)


def calc_correlations_oneway(dict_a, dict_b):
    """Input two dictionaries (nested) calculate correlations of arrays within dictionaries.
    where dictionary_a has depth 2 (dict[a][b]) and dictionary_b has depth 1 (dict[a])"""
    
    # predefine correlation matrix
    correlations = np.zeros((len(dict_a), len(dict_a[list(dict_a.keys())[0]])))

    # calculate correlations
    for pref in range(len(dict_a)):
        cur_pref = list(dict_b.keys())[pref]
        # get rid of non finite numbers
        dict_b[cur_pref] = np.nan_to_num(dict_b[cur_pref], 
                                         nan=0, 
                                         posinf=np.nanmax(dict_b[cur_pref][np.isfinite(dict_b[cur_pref])]))
        # calculate correlation
        correlations[pref, :] = [np.corrcoef(dict_a[list(dict_a.keys())[pref]][shrp], 
                                             dict_b[list(dict_b.keys())[pref]])[0][1] for shrp in dict_a[list(dict_a.keys())[pref]]]
    return(correlations)


def relative_adaptation(adapt_dict):
    """Calculate the relative adaptation away from 0,
    return adjusted dictionary"""
    
    # set up relative adaptation dict
    rel_adapt_dict = {}
    
    # loop over prefferences and calculate relative adaptation from 0
    for pref in adapt_dict:
        rel_adapt_dict[pref] = {shrp: (1-adapt_dict[pref][shrp]) for shrp in adapt_dict[pref]}
    return(rel_adapt_dict)


def plot_surprise_adaptation_correlation_activ(adaptation_dict, surprisal_dict):
    """plot correlation matrix of adapted and non adapted activation"""
    
    # calculate the correlations
    correlations = calc_correlations(adaptation_dict, surprisal_dict)
    correlations = np.nan_to_num(correlations, nan=1)
    
    # set image settings
    fig, ax = plt.subplots(figsize=(18, 12))

    # plot heat map
    ax = sns.heatmap(correlations)

    # set correct x/y ticks
    ax.set_yticklabels(np.linspace(np.min(np.array(list(adaptation_dict.keys()))), 
                                   np.max(np.array(list(adaptation_dict.keys()))), 
                                   len(ax.get_yticklabels()))) 
    ax.set_xticklabels(list(surprisal_dict[list(surprisal_dict.keys())[0]].keys()))

    # pimp graph
    plt.ylabel('tuning prefference', fontsize=16)
    plt.yticks(fontsize=12)
    plt.xlabel('tuning sharpness', fontsize=16)
    plt.xticks(fontsize=12)
    plt.title('Correlation matrix\nAdapted-activations vs Predicted activations', fontsize=22)
    
    # display only some labels
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 == 0]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 2 != 0]
    return(ax)


def plot_sequence_comp(act_dict, adapt_dict, probability_dict, stim_train, display_neuron, display_shrp):
    """"Plot components of a sequence of stimulutation following adaptation and prediction"""

    # configure plot size
    fig, ax = plt.subplots(4, 
                           1, 
                           sharex=True, 
                           figsize=(16, 14), 
                           gridspec_kw={'height_ratios': [2, 1, 1, 1]})


    ax[0].scatter(np.arange(1, len(stim_train)+1), stim_train)
    ax[0].plot(np.ones(len(stim_train))*display_neuron, ls='--', color='grey', alpha=0.8, lw=4)
    ax[1].plot(act_dict[display_neuron][display_shrp], lw=2, color='tab:orange', label='Tuning activation')
    ax[2].plot(-adapt_dict[display_neuron][display_shrp] + 1, lw=2, color='tab:green', label='Neuron adaptation')
    ax[3].plot(probability_dict[display_neuron], lw=2, color='tab:purple', label='HGF likelihood')

    ax[0].set_ylabel('Frequency', fontsize=18)
    ax[0].set_yticklabels(np.round(np.exp(ax[0].get_yticks()/1.4)+10))
    ax[0].tick_params(axis='y', which='major', labelsize=18)

    ax[1].set_ylabel('Activation', fontsize=18)
    ax[1].legend(fontsize=18, loc=1)
    ax[1].tick_params(axis='y', which='major', labelsize=18)

    ax[2].set_ylabel('Adaptation', fontsize=18)
    ax[2].legend(fontsize=18, loc=1)
    ax[2].tick_params(axis='y', which='major', labelsize=18)

    ax[3].set_ylabel('likelihood.', fontsize=18)
    ax[3].legend(fontsize=18, loc=1)
    ax[3].tick_params(axis='y', which='major', labelsize=18)

    plt.xlabel('Stimulus', fontsize=18)
    plt.xticks(fontsize=18)
    plt.suptitle('Sequence components (for neuron tuned to {}Hz)'.format(np.round(np.exp(display_neuron/1.4))+10), fontsize=24)
    plt.tight_layout()
    return(ax)


def plot_probability_adaptation_correlation(adaptation_dict, probability_dict):
    """plot correlation matrix of adapted and non adapted activation"""
    
    # calculate the correlations
    correlations = calc_correlations_oneway(adaptation_dict, probability_dict)
    correlations = np.nan_to_num(correlations, nan=1)
    
    # set image settings
    fig, ax = plt.subplots(figsize=(18, 12))

    # plot heat map
    ax = sns.heatmap(correlations)

    # set correct x/y ticks
    ax.set_yticklabels(np.linspace(np.min(np.array(list(adaptation_dict.keys()))), 
                                   np.max(np.array(list(adaptation_dict.keys()))), 
                                   len(ax.get_yticklabels()))) 
    ax.set_xticklabels(list(adaptation_dict[list(adaptation_dict.keys())[0]].keys()))

    # pimp graph
    plt.ylabel('tuning prefference', fontsize=16)
    plt.yticks(fontsize=12)
    plt.xlabel('tuning sharpness', fontsize=16)
    plt.xticks(fontsize=12)
    plt.title('Correlation matrix\nAdaptation vs Predictions', fontsize=22)

    # display only some labels
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 == 0]
    [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 2 != 0]
    return(ax)


def plot_convolved_components(activs, adapts, probs, stims, pref, shrp, display_len='full'):
    """plot the hrf convolved activations, adaptations & probabilities
    input,  activs:      activation of sequence
            adapts:      adaptation of sequence
            probs:       probabilitie by prediction of sequence
            pref:        what neuron prefference to display
            shrp:        what neuron sharpness to display
    optional, display_len:   set presentation length in same time domain"""

    # set display length
    if display_len == 'full':
        display_len = len(stims)

    # configure plot size
    fig, ax = plt.subplots(2, 
                           1, 
                           sharex=True, 
                           figsize=(12, 8), 
                           gridspec_kw={'height_ratios': [2, 1]})

    # prepair stim train and plot
    stims[stims == 0] = np.nan
    ax[0].scatter(np.arange(1,display_len+1), stims[:display_len])
    ax[0].plot(np.ones(display_len) * pref, ls='--', color='grey', lw=4)

    # plot activations, convolved activation, and stimuli train
    ax[1].plot(activs[pref][shrp][:display_len], label='Activations', alpha=0.6, lw=3)
    ax[1].plot(adapts[pref][shrp][:display_len], label='Adaptations', alpha=0.6, lw=3)
    ax[1].plot(probs[pref][:display_len], label='Probabilities', alpha=0.6, lw=3)

    # pimp activations
    ax[0].set_ylabel('presented frequency', fontsize=16)
    ax[0].set_yticklabels(np.round(np.exp(ax[0].get_yticks()/1.4)+10))
    ax[0].tick_params(axis='y', which='major', labelsize=16)

    # pimp convolved activations
    ax[1].set_ylabel('gamma', fontsize=16)
    ax[1].tick_params(axis='y', which='major', labelsize=16)
    ax[1].legend(fontsize=16, loc=1)

    # pimp layout
    plt.xlabel('time (ms)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.suptitle('HRF convolved components', fontsize=22)
    plt.tight_layout()
    return(ax)