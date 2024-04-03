## series of functions to load matlab data and parse them into usebable formats

import scipy.io
import pandas as pd
import numpy as np
import re

import os
from os.path import join
import shutil

import matlab.engine

import matplotlib.pyplot as plt

import Adaptation.longtrace_adaptation as longtrace_adaptation
import Adaptation.longtrace_adaptation_timedomain as longtrace_adaptation_timedomain

import vtc

# load current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# vtc / y data loading 

def load_vtc_chunk_runs(vtc_fns, msk_indeces):
    """load decired chunk over a list of runs
    input: vtc_fn (list of strings): list of full path vtc filenames
           msk_indeces (3x np.array): np.where style indeces of what voxels"""
    
    # load only header information
    head, _ = vtc.read_vtc_msk(vtc_fns[0], tuple((np.array([0]),
                                                  np.array([0]),
                                                  np.array([0]))))

    # get expected vtc dim
    vtcdim = vtc.get_vtc_dims(head)
    
    # predefine full image over runs
    # y = np.zeros((msk[0].shape[0], vtcdim[-1], nr_runs))
    y = np.zeros((msk_indeces[0].shape[0], vtcdim[-1], len(vtc_fns)))
    run_nr = np.zeros((vtcdim[-1], len(vtc_fns)))
    
    # loop over all filenames
    for run in range(len(vtc_fns)):

        # set vtc path
        fullpath = vtc_fns[run]
        
        # mask the vtc
        _, y[:,:,run] = vtc.read_vtc_msk(fullpath, msk_indeces)
        run_nr[:, run] = run + 1
        
    # reshape into single dim
    run_nr = run_nr.reshape((-1),order='F')
    y = y.reshape((y.shape[0], -1), order='F')
    return(y, run_nr)
    
    
def reconstruct_vtc(img, msk_indeces, vtc_for_header=None):
    """reconstruct image in zeros array, takingen [voxels x time] and indeces 
    as input.
    input: img (array): input array of [voxels, timepoints]
           msk_indeces (3x np.array): np.where style indeces of what voxels
                        should have same length as img
           vtc_for_header (string) : vtc filename to use for header - needed for 
                        dimensions. when missing take max of msk_indeces instead.
                        (this will cause one sided padding).
    return: reconstructed image in format of vtc"""
    
    # check if we can load a vtc from header
    #  else, take from mask (will be padded)
    if vtc_for_header:
        # load header info only (no img data)
        head, _ = vtc.read_vtc_msk(vtc_for_header, tuple((np.array([0]),
                                                    np.array([0]),
                                                    np.array([0]))))
        # get vtc dims
        vtcdim = vtc.get_vtc_dims(head)
    else:
        # set dimensions if no vtc header was given
        vtcdim = [np.max(c)+1 for c in msk_indeces] + [img.shape[-1]]
        
    # preprecreat a empty image with the dimension
    #  of the full vtc - for plotting purpuses
    rec_img = np.zeros(vtcdim)

    # fill in chuck in reconstructed image
    rec_img[msk_indeces] = img
    return(rec_img)

## main functions

## post processing
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def find_padding(tr, ntrials, tlen):
    padding = 0
    notdiv = int(ntrials*tlen*10) % int(tr*10)
    while notdiv != 0:
        padding += 1
        newlen = ntrials + padding
        # check if modulated by 10 is whole number 
        if (newlen*tlen*10).is_integer():
            notdiv = int(newlen*tlen*10) % int(tr*10)
        if padding > 100000: 
            raise Exception("Sorry, padding not possible") 
    return(padding)

def get_reg_collumns(df, convolved=False,
                     pred_reg = 'pred_prob_[_0-9.]+',
                     acti_reg = 'raw_acti_[_0-9.]+',
                     adpt_reg = 'raw_adapt_[_0-9.]+',
                     acadpt_reg = 'adapt_activ_[_0-9.]+',
                     surp_reg = 'surprisal',
                     onoff_reg = 'onoff'):
    """get all regressors collumn names in one list
    optionally sellect convolved variants"""
    # get one list with all
    columns_reg = []
    # add convolved suffix
    if convolved: suffix='convolved$'
    else: suffix='$'
    # loop over all, adding suffix
    for reg in ['{}{}'.format(i, suffix) for i in [pred_reg, acti_reg, adpt_reg, acadpt_reg, surp_reg, onoff_reg]]:
        columns_reg += df.filter(regex=reg).columns.tolist()
    return(columns_reg)

def get_tw_collumns(df, tp, tw, convolved=False,
                     pred_reg = 'pred_prob_',
                     acti_reg = 'raw_acti_',
                     adpt_reg = 'raw_adapt_',
                     acadpt_reg = 'adapt_activ_',
                     surp_reg = 'surprisal',
                     exta_reg = 'onoff'):
    """get all by tuning prefference (tp) and tuning width (tw)"""
    # get one list with all
    columns_reg = []
    # add convolved suffix
    if convolved: suffix='_convolved$'
    else: suffix='$'
      
    # adjust column names
    pred_reg = '{}{:.3f}{}'.format(pred_reg, tp, suffix)
    acti_reg = '{}{:.3f}_{:.3f}{}'.format(acti_reg, tp, tw, suffix)
    adpt_reg = '{}{:.3f}_{:.3f}{}'.format(adpt_reg, tp, tw, suffix)
    acadpt_reg = '{}{:.3f}_{:.3f}{}'.format(acadpt_reg, tp, tw, suffix)
    surp_reg = '{}{}'.format(surp_reg, suffix)
    exta_reg = '{}{}'.format(exta_reg, suffix)
    # loop over all
    for reg in [pred_reg, acti_reg, adpt_reg, acadpt_reg, surp_reg, exta_reg]:
        columns_reg += df.filter(regex=reg).columns.tolist()
    return(columns_reg)

def stims_add_temporal_pad(df, volumes_df, trialleng=None, trleng=None):
    """take dataframe, loop over blocks, and add inbetween timing values,
    while this is not 100% precise (and it doesnt have to be), this step is crucial
    in order for the hrf convolvement to work as intended
    input: df (stimulus dataframe), trailleng (length of single trial - if None, get from first diff),
    and append_end (how manny seconds to add after last block)"""

    # if not specified calculate trialleng
    if not trialleng: trialleng = df['timing'].diff()[1]
    if not trleng: trleng = volumes_df['timing'].diff()[1]
        
    # get column regressor names
    columns_reg = get_reg_collumns(df)
    
    temp_dfs = {}
    # loop over runs
    for blk in df['block'].unique():
        
        # get last and first idx value of this block
        blk_endidx = df[df['block'] == blk].index[-1]
        blk_stridx = df[df['block'] == blk].index[0]
        currun = df['run'].iloc[blk_endidx]
        
        # if not last volume of block, simply count to next stimuli
        if currun == df['run'].shift(-1).iloc[blk_endidx]:
            nd_timing = df['timing'].iloc[blk_endidx+1]-0.1,   # ensure non duplicated
        # if last block of a run, count to last volume in run
        else:
            nd_timing = volumes_df[volumes_df['run'] == currun]['timing'].to_numpy()[-1]

        # create intermediate timing values
        blk_append = np.arange(df['timing'].iloc[blk_endidx]+trialleng, 
                               nd_timing,
                               trialleng)
            
        # if first block of run also pad from start of block  # NOW JUST FIRST BLOCK, MAKE FIRST BLOCK RUN
        if currun != df['run'].shift(1).iloc[blk_stridx]:
            run_starttime = volumes_df[volumes_df['run'] == currun]['timing'].to_numpy()[0] - trleng
            blk_append = np.append(np.arange(run_starttime,  # ensure first volume
                                             df['timing'].iloc[blk_stridx]-0.1,   # ensure non duplicated
                                             trialleng),
                                   blk_append)

        # allocate everything with zeros (important for convolution)
        temp_dfs[blk] = pd.DataFrame(0, 
                                     index=np.arange(len(blk_append)),
                                     columns=df.columns)
    
        # set non regressors to nan or to corresponding
        temp_dfs[blk][list(set(df.columns) - set(columns_reg))] = np.nan
        temp_dfs[blk]['timing'] = blk_append
        temp_dfs[blk]['run'] = currun
        temp_dfs[blk]['block'] = blk
    
    # combine dataframes
    df_list = [0] + list(df['block'].unique())
    temp_dfs[0] = df
    df = pd.concat([temp_dfs[k] for k in df_list], ignore_index=True)
    df = df.sort_values(by=['timing'], ignore_index=True)
    
    # finally recalculate what volume this timing fall in (both floor and closest)
    timings = df['timing'].to_numpy()
    run = df['run'].to_numpy()
    vol_absz, vol_relz = closest_vol(volumes_df, timings, run)
    vol_abs_flrz, vol_rel_flrz = closest_vol_floor(volumes_df, timings, run)
    
    # apply volume timing to dataframe
    df['closest_volume_rel'] = vol_relz
    df['closest_volume_abs'] = vol_absz
    df['volume_rel'] = vol_rel_flrz
    df['volume_abs'] = vol_abs_flrz
    
    # add on off column
    df['onoff'] = df['frequencies'].notna().astype(float)
    
    return(df)

def stims_convolve_hrf(df, hrf):
    """input pandas dataframe, and convolve with an hrf array (must have same x-dim)
    return adjusted dataframe with suffix _convolved"""
    
    # sellect columns to convolve
    columns_reg = get_reg_collumns(df)
    input_array = df[columns_reg].to_numpy()

    # do actual convolvement, many with one
    convolved = np.apply_along_axis(lambda m: np.convolve(m, hrf, mode='full'), 
                                    axis=0, 
                                    arr=df[columns_reg].to_numpy())[:input_array.shape[0],:]

    # put in dataframe and join into input dataframe
    convolved_df = pd.DataFrame(convolved, columns=columns_reg)
    df = df.join(convolved_df, rsuffix='_convolved')
    return(df)

def stims_to_tr(stim_df, volumes_df, downsample_unconv=False):
    """from dataframe in stimulus domain, create tr dataframe
    if downsample_unconv is True, also downsample (using scipy) unconvolved columns"""
    
    # to get dataframe in tr space, simply 
    tr_df = stim_df.groupby(['volume_abs']).nth(3)

    # also downsameple in a waveform manner, to preserve more data
    all_reg = get_reg_collumns(stim_df, convolved = True)
    if downsample_unconv: all_reg += get_reg_collumns(stim_df, convolved = False)
    downsampled_reg = pd.DataFrame(scipy.signal.resample(stim_df[all_reg], len(volumes_df)),
                                   columns = all_reg)
    
    # make, setting new values to original, and from groupby to '_sum'
    tr_df = tr_df.join(downsampled_reg, lsuffix='_mid')
    return(tr_df)


def con_hrf_stimdomain(hrf, stim_df, plotres=False):
    """given a mat function and triallength obtained from stim_df, 
    get hrf in correct domain for convolution"""

    # input hrf
    con_hrf = [hrf['xdata'][0,:], hrf['ydata'][0,:]]

    # hrf and stim lengts
    trialleng = stim_df['timing'].diff()[1]
    con_hrf_vollen = np.diff(con_hrf[0])[0]

    # interpolate
    xnew = np.arange(np.min(con_hrf[0]), np.max(con_hrf[0]), trialleng)
    f = scipy.interpolate.interp1d(con_hrf[0], con_hrf[1], kind='cubic')
    newhrf = normalize(f(xnew)) # normalized (top scaled to 1)
    
    # plot results if wanted
    if plotres:
        plt.plot(xnew, newhrf)
        plt.plot(con_hrf[0], normalize(con_hrf[1]))
        print(f'new step size {np.diff(xnew)[0]}')
    return(newhrf)

#drex
def stims_export_mat(pp, input_dir, stim_df, pref_range):
    """export dataframe into mat file"""
    stim_mat = {}

    # get stimuli data
    stim_mat['stims'] = stim_df.to_dict('list')

    # aditionally get range data
    stim_mat['oct_range'] = list(pref_range)
    stim_mat['freq_range'] = list(2 ** pref_range)

    scipy.io.savemat(join(input_dir, '{}/{}_stimdf.mat'.format(pp, pp)), stim_mat)
    return


def run_drex(pp, input_dir):
    """run drex model in matlab, save output as matfile"""
    eng = matlab.engine.start_matlab()

    # add prediction, drex path
    s = eng.genpath(join(dir_path, 'Prediction', 'DREX'))
    eng.addpath(s, nargout=0)

    # run drex wrapper
    eng.rundrex_stims(pp, input_dir, nargout=0)
    return


def stims_add_drex(pp, input_dir, stim_df):
    """load drex output mat, and append to dataframe"""
    # load drex mat
    mat = scipy.io.loadmat(join(input_dir,'{}/{}_drexdf.mat'.format(pp, pp)))

    # loop over frequencies
    collumn_names = ['pred_prob_{:.3f}'.format(frq) for frq in mat['s_range'][0]]
    temp_df = pd.DataFrame(columns=collumn_names)
    for frq in range(len(mat['s_range'][0])):
        cur_frq = mat['s_range'][0, frq]
        temp_df['pred_prob_{:.3f}'.format(cur_frq)] = mat['prob_array'][frq]

    # append surprisal and predictive probabilities
    stim_df['surprisal'] = mat['surp_array'][0]
    stim_df = pd.concat([stim_df, temp_df], axis=1)
    return(stim_df)

# adaptation
def run_adaptation(stim_df, pref_range, sharp_range, y_decay):
    """wrapper functions to run adaptation model and return long matrixes of [pref*tw, stimuli]"""
    
    # calculate raw activation
    stims = stim_df['frequencies_oct'].to_numpy()
    activations = longtrace_adaptation.md_gaussian_activations(pref_range, sharp_range, stims)
    adaptations = np.zeros([len(pref_range)*len(sharp_range), len(stims)])
    n_back_adaptations = np.zeros([len(pref_range)*len(sharp_range), len(stims), len(y_decay)])

    for blk in stim_df['block'].unique():
        # get all stimuli within this block & get start and end idx of block
        stims = stim_df['frequencies_oct'][stim_df['block'] == blk].to_numpy()
        st_idx = stim_df.index[stim_df['block'] == blk][0]
        nd_idx = stim_df.index[stim_df['block'] == blk][-1] + 1

        # calculate adaptation for current block
        adaptations[:, st_idx:nd_idx], n_back_adaptations[:, st_idx:nd_idx, :] = longtrace_adaptation.md_stim_adaptation(stims, 
                                                                                                    y_decay, 
                                                                                                    pref_range, 
                                                                                                    sharp_range)

    # calculate adaptation weighted activations
    adapted_activations = np.multiply(adaptations, activations)

    return(activations, adaptations, adapted_activations, n_back_adaptations)


def stims_add_adaptation(stim_df, pref_range, sharp_range, activations, adaptations, adapted_activations):
    """given a adaptation, activation and adaptated activation matrix, update the dataframe"""
    # create a list of all indexes
    all_idxs = np.arange(len(pref_range) * len(sharp_range))

    # get list of 
    tunprefs, tunsharps = longtrace_adaptation.md_get_tuning(all_idxs, pref_range, sharp_range)

    # get dictionaries by naming
    acti_names = {'raw_acti_{:.3f}_{:.3f}'.format(tunprefs[idx], tunsharps[idx]): 
                  activations[idx, :] for idx in all_idxs}
    adapt_names = {'raw_adapt_{:.3f}_{:.3f}'.format(tunprefs[idx], tunsharps[idx]): 
                   adaptations[idx, :] for idx in all_idxs}
    adapt_acti_names = {'adapt_activ_{:.3f}_{:.3f}'.format(tunprefs[idx], tunsharps[idx]): 
                        adapted_activations[idx, :] for idx in all_idxs}

    # combine dictionaries
    acti_names.update(adapt_names)
    acti_names.update(adapt_acti_names)
    
    # append adapation and activation to pd dataframe
    stim_df = pd.concat([stim_df, pd.DataFrame(acti_names)], axis=1)
    return(stim_df)

# main loading
def data_load(pp,input_dir):
    """load mainpred mat file and stimuli matfile"""
    mat = scipy.io.loadmat(join(input_dir,
                                f'{pp}-mainpred.mat'))
    stimuli = scipy.io.loadmat(join(input_dir, 
                                    f'{pp}_main_stims.mat'))
    return(mat, stimuli)


def stims_load(puls_df, volumes_df, mat, stimuli):
    """using information from stimuli and pulse timing create dataframe 
    with frequency information, pulse location etc.
    note: 'volume_rel' & 'vol_abs' are the volume where this stimuli was measured
    'closest_volume_rel' & 'closest_volume_abs' are the volume which is the closest in time
    (half tr shift) - since a tr should capture information within that tr"""

    # set arrays
    freqz   = np.array([])
    timingz  = np.array([])
    runz     = np.array([])
    blockz   = np.array([])
    segmenz  = np.array([])
    centaz   = np.array([])
    centbz   = np.array([])
    probaz   = np.array([])
    probbz   = np.array([])

    for blk in np.arange(1, puls_df['block'].max()+1):
        # get blockidx
        idxblock = np.where(mat['timingz'][1] == blk) # where block is 1

        #get frequency presentation data for block
        frequencies = stimuli['pres_freq'][int(blk)-1, :]

        # other values
        tps = np.sum(mat['timingz'][3, idxblock] == 1) # get trials per secion

        #get timings back from mat file, substract begin time
        timings = mat['timingz'][4, idxblock]
        matidx = np.where(mat['segmentz'][1] == blk)

        # append to arrays
        freqz = np.append(freqz, frequencies)
        timingz = np.append(timingz, timings)
        runz = np.append(runz, np.repeat(mat['segmentz'][0][matidx], tps))
        blockz = np.append(blockz, np.repeat(mat['segmentz'][1][matidx], tps))
        segmenz = np.append(segmenz, np.repeat(mat['segmentz'][2][matidx], tps))
        centaz = np.append(centaz, 2**np.repeat(mat['segmentz'][7][matidx], tps))   # cent freq a
        centbz = np.append(centbz, 2**np.repeat(mat['segmentz'][8][matidx], tps))  # cent freq b
        probaz = np.append(probaz, np.repeat(mat['segmentz'][5][matidx], tps))
        probbz = np.append(probbz, np.repeat(mat['segmentz'][6][matidx], tps))

    # oct variant 
    freqz_oct = np.log2(freqz)
    centaz_oct = np.log2(centaz)
    centbz_oct = np.log2(centbz)

    # get closest pulse
    vol_absz, vol_relz = closest_vol(volumes_df, timingz, runz)
    vol_abs_flrz, vol_rel_flrz = closest_vol_floor(volumes_df, timingz, runz)

    # put data into a dictionary and subsequentially in a dataframe
    stim_df_dict = {'frequencies': freqz,
                    'frequencies_oct': freqz_oct,
                    'timing': timingz,
                    'closest_volume_rel' : vol_relz,
                    'closest_volume_abs' : vol_absz,
                    'volume_rel' : vol_rel_flrz,
                    'volume_abs' : vol_abs_flrz,
                    'run': runz,
                    'block': blockz,
                    'segment': segmenz,
                    'center_freq_a': centaz,
                    'center_freq_b': centbz,
                    'center_freq_a_oct': centaz_oct,
                    'center_freq_b_oct': centbz_oct,
                    'probability_a': probaz,
                    'probability_b': probbz
                   }

    stim_df = pd.DataFrame(stim_df_dict)
    return(stim_df)


def pulses_load(pp, input_dir, nr_runs):
    """load pulses of each run into a pandas dataframe"""
    
    # set empty array to concatenate pulse fetched data
    allpulsez = 0

    # loop over runs and parse data
    for currun in np.arange(1,nr_runs+1):
        pulsez = scipy.io.loadmat(join(input_dir,
                                       f'{pp}/_{pp}-r{currun}-pulses.mat'))['pulsez']
        pulsez = np.vstack([np.repeat(currun, pulsez.shape[1]), pulsez])
        try:
            allpulsez = np.concatenate((allpulsez,pulsez),axis=1)
        except:
            allpulsez = pulsez

    # put in dataframe
    pulsez_df = pd.DataFrame(np.transpose(allpulsez), columns=['run', 'block', 'timing'])
    return(pulsez_df)

def volumes_load(pulsez_df, tr, volumes, nr_runs=False):
    """given pulses dataframe, tr and volumes
    create dataframe in volume space (add no pulses gaps)"""
    
    # nr of runs calculations
    if not nr_runs: nr_runs = int(pulsez_df['run'].max())

    # precreate dicts
    tm = {}
    onoff = {}
    block = {}

    # loop over all the runs
    for currun in np.arange(1, nr_runs+1):
        tm[currun], onoff[currun] = find_onoff(pulsez_df, currun, tr=tr, volumes=volumes)
        block[currun] = find_block(pulsez_df, 1, tr=tr, volumes=volumes)
        
    # append to one full array
    tmz = np.array([])
    onoffz = np.array([])
    runz = np.array([])
    blockz = np.array([])

    # loop over runs and fill
    for currun in tm.keys():
        tmz = np.append(tmz, tm[currun])
        onoffz = np.append(onoffz, onoff[currun]) 
        runz = np.append(runz, np.array([currun] * len(tm[currun])))
        blockz = np.append(blockz, block[currun]) 

    # put everything in a pandas dataframe
    df_inf = {'timing':tmz, 'run':runz, 'block':blockz, 'on-times':onoffz}
    volumes_df = pd.DataFrame(df_inf)
    
    return(volumes_df)

def find_onoff(pulsez_df, currun, tr=1.8, volumes=245):
    """for volumes and tr, calculate the on off timings"""
    # load important info
    ab_t0 = pulsez_df['timing'][pulsez_df['run']==currun].iloc[0]  # this is for run 1
    ab_end = pulsez_df['timing'][pulsez_df['run']==currun].iloc[-1]  # this is for run 1
    
    # set range
    pulserangerun = np.linspace(ab_t0, ab_end, num=volumes, retstep=False)
    ontimes = pulsez_df['timing'][pulsez_df['run']==currun].to_numpy()
    
    # get off times
    offtimes = np.abs(pulserangerun[:,None]-ontimes).argmin(0) # get on times
    onoff = np.ones(volumes)
    onoff[offtimes] = 0 # set off times to 0
    return(pulserangerun, onoff)

def find_block(pulsez_df, currun, tr=1.8, volumes=245):
    # load important info
    ab_t0 = pulsez_df['timing'][pulsez_df['run']==currun].iloc[0]  # this is for run 1
    ab_end = pulsez_df['timing'][pulsez_df['run']==currun].iloc[-1]  # this is for run 1
    blocks = pulsez_df['block'][pulsez_df['run']==currun].to_numpy()
    
    # set range
    pulserangerun = np.linspace(ab_t0, ab_end, num=volumes, retstep=False)
    ontimes = pulsez_df['timing'][pulsez_df['run']==currun].to_numpy()

    # get off times
    offtimes = np.abs(pulserangerun[:,None]-ontimes).argmin(0) # get on times
    allblocks = np.empty(volumes)
    allblocks[:] = np.nan
    
    allblocks[offtimes] = blocks # set off times to 0
    allblocks = ffill(allblocks) # forward fill the array
    return(allblocks)

def closest_vol(volumes_df, timingz, runz):
    a = volumes_df['timing'].to_numpy()
    b = timingz
    
    # get volume
    seq_vol = np.abs(a[:,None]-b).argmin(0)
    rel_vol = seq_vol - (runz-1)*float(len(volumes_df[volumes_df['run'] == 1]))
    return(seq_vol, rel_vol.astype(int))

def closest_vol_floor(volumes_df, timingz, runz):
    a = volumes_df['timing'].to_numpy()
    b = timingz

    # get volume
    dist = (a[:,None]-b) # get the distance
    dist[dist < 0.05] = 999 # get rid of pos distancane (- some window)
    seq_vol = np.abs(dist).argmin(0)
    rel_vol = seq_vol - (runz-1)*float(len(volumes_df[volumes_df['run'] == 1]))
    return(seq_vol, rel_vol.astype(int))

def closest_vol_timing(volumes_df, timingz, runz):
    a = volumes_df['timing'].to_numpy()
    b = timingz

    # get volume
    seq_vol = np.abs(a[:,None]-b).argmin(0)
    rel_vol = seq_vol - (runz-1)*float(len(volumes_df[volumes_df['run'] == 1]))
    return(seq_vol, rel_vol)

def ffill(arr, axis=0):
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [np.arange(k)[tuple([slice(None) if dim==i else np.newaxis
        for dim in range(len(arr.shape))])]
        for i, k in enumerate(arr.shape)]
    slc[axis] = idx
    return arr[tuple(slc)]

def flatten(d):
    return(pd.json_normalize(d, sep='_').to_dict(orient='records')[0])

def gauss(x, x0, sigma):
    return np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def copy_files(origins, destinations):
    """function for copying a series of files from on destiantion to another,
    usefull for temporary ram-drive or ssd saving for efficiently searching of spaced binary data"""
    if len(origins) != len(destinations):
        print("Error: The lists of origins and destinations are not of the same length.")
        return

    for origin, destination in zip(origins, destinations):
        try:
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)

            # Copy the file
            shutil.copy2(origin, destination)
            print(f"Copied {origin} to {destination}")

        except FileNotFoundError:
            print(f"File not found: {origin}")
        except PermissionError:
            print(f"Permission denied when copying {origin} to {destination}")
        except Exception as e:
            print(f"Error occurred when copying {origin} to {destination}: {e}")