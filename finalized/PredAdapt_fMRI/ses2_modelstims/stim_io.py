## series of functions to load matlab data and parse them into usebable formats

import scipy.io
import pandas as pd
import numpy as np
import re
import sys
import pickle
import itertools

import os
from os.path import join
import shutil

import matlab.engine

import matplotlib.pyplot as plt

# load current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

## import local

import Adaptation.longtrace_adaptation as longtrace_adaptation
import Adaptation.longtrace_adaptation_timedomain as longtrace_adaptation_timedomain
import IdealObserver.idealobserver as idealobserver

import vtc
import bvbabel

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

def downsample_with_boundary_inclusion(arr, factor=2, boundary_option=1, mr=2):
    """
    Downsamples a 3D array by the given factor, with options for handling boundaries.
    
    Input:
    arr : (3d numpy array) - array to downsample (e.g. voi array)
    factor : (int) - default 2 - blocksize of vtc compared to vmrs
    boundary_option : (int) - option for handling boundaries:
        1 - if any voxel in the block is active, downsampled voxel will be active.
        2 - only if all voxels in the block are active, downsampled voxel will be active.
        3 - majority rule, if more than half of the voxels in the block are active (or percentage decided by mr), downsampled voxel will be active.
    mr : int
        Option for majority rule, 2 requires majority to be in, 3 only 1/3th etc
    returns: (3d numpy array) - downsampled 3D array.
    """
    # Determine new shape after downsampling
    new_shape = np.array(arr.shape) // factor
    downsampled = np.zeros(new_shape, dtype=arr.dtype)

    # Iterate over the new downsampled shape
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                # Extract the relevant block from the original array
                block = arr[i*factor:(i+1)*factor, 
                            j*factor:(j+1)*factor, 
                            k*factor:(k+1)*factor]
                
                # Handle boundary according to the chosen option
                if boundary_option == 1:
                    # Option 1: If any voxel is active
                    if np.any(block):
                        downsampled[i, j, k] = 1

                elif boundary_option == 2:
                    # Option 2: Only if all voxels are active
                    if np.all(block):
                        downsampled[i, j, k] = 1

                elif boundary_option == 3:
                    # Option 3: Majority rule
                    if np.sum(block) > (block.size // mr):
                        downsampled[i, j, k] = 1
    return downsampled


def voi_msk(voi_head, voi_img, msk_head, boundary_option=1, mr=2, returnidx=True):
    """msk vtc dimension style files using a voi file, returns dictionary of msks 
    (times number of VOI in .voi)
    input voi_head (dict): voi header information obtained from bvbabel.voi
          voi_img (list of arrays): list of numpy arrays with each element representing a new VOI
          msk_head (dit): dummy msk header information, needed to capture vtc box location
          boundary_option : (int) - option for handling boundaries/voxels withing block [1: any_voxel, 2: all_voxel, 3: majority rule]
          mr (int): option for majority rule, 2 requires majority to be in, 3 only 1/3th etc
          returnidx (bool) - return either linear indexes of msk position (True) or full mask (False)
    return (dict) - dictionary of either linear indexes (by nr of vols) or mask files
    """

    # Initialize a dictionary to store voxel indices for each VOI
    vox_idx = {}

    # loop over vois
    for v in range(len(voi_img)):  # Iterate through each VOI entry (assuming 3 VOIs as per your example)

        # Predefine new image array
        new_img = np.zeros((voi_head['OriginalVMRFramingCubeDim'],
                            voi_head['OriginalVMRFramingCubeDim'],
                            voi_head['OriginalVMRFramingCubeDim']))

        # Get the VOI coordinates
        xind = voi_img[v]['Coordinates'][:,0]
        yind = voi_img[v]['Coordinates'][:,1]
        zind = voi_img[v]['Coordinates'][:,2]

        # Populate the new image array with the VOI mask
        new_img[xind, yind, zind] = 1

        # Convert dimensions (transpose and flip)
        new_img = np.transpose(new_img, (2,0,1))
        new_img = new_img[::-1, ::-1, ::-1]

        # Extract the relevant VMR region using the mask's coordinates
        extracted_vmr = new_img[-msk_head['ZEnd'] +1 : -msk_head['ZStart']+1,
                                -msk_head['XEnd'] +1 : -msk_head['XStart']+1, 
                                -msk_head['YEnd'] +1 : -msk_head['YStart']+1]

        # Perform downsampling with boundary inclusion
        factor = msk_head['VTC resolution relative to VMR (1, 2, or 3)']
        downsampled_vmr = downsample_with_boundary_inclusion(extracted_vmr, 
                                                             factor=factor, 
                                                             boundary_option=boundary_option, 
                                                             mr=mr)

        # Store voxel indices of the downsampled array where the value is greater than 0
        if returnidx == True:
            vox_idx[voi_img[v]['NameOfVOI']] = np.where(downsampled_vmr > 0)
        # Or return full mask file
        else:
            vox_idx[voi_img[v]['NameOfVOI']] = downsampled_vmr
    return vox_idx

import numpy as np

def count_voxels_in_blocks(arr, factor=2):
    """
    Counts the number of active voxels within each downsampling block.
    
    Input:
    arr : (3D numpy array) - array to analyze (binary mask, 0 or 1)
    factor : (int) - downsampling factor (e.g., 2 for 2x2x2 blocks)
    
    Returns:
    (3D numpy array) - same shape as the downsampled grid, containing voxel counts.
    """
    # Determine new shape after downsampling
    new_shape = np.array(arr.shape) // factor
    voxel_counts = np.zeros(new_shape, dtype=int)

    # Iterate over the new downsampled shape
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                # Extract the relevant block from the original array
                block = arr[i*factor:(i+1)*factor, 
                            j*factor:(j+1)*factor, 
                            k*factor:(k+1)*factor]

                # Count the number of active voxels in the block
                voxel_counts[i, j, k] = np.sum(block)

    return voxel_counts


def voi_voxel_count(voi_head, voi_img, msk_head, returnfull=False):
    """
    Computes the count of active voxels within each downsampled block for each VOI.
    
    Input:
    voi_head (dict) - VOI header information from bvbabel.voi
    voi_img (list of dicts) - List of numpy arrays representing each VOI
    msk_head (dict) - Mask header information (needed for defining VTC extraction)
    returnidx (bool) - Whether to return indices of nonzero counts (True) or full voxel count masks (False)
    
    Returns:
    dict - Dictionary of voxel count data, either as indices or full masks.
    """

    # Initialize a dictionary to store voxel counts for each VOI
    vox_counts = {}

    # Loop over VOIs
    for v in range(len(voi_img)):  

        # Initialize new image array
        new_img = np.zeros((voi_head['OriginalVMRFramingCubeDim'],
                            voi_head['OriginalVMRFramingCubeDim'],
                            voi_head['OriginalVMRFramingCubeDim']))

        # Get VOI coordinates
        xind = voi_img[v]['Coordinates'][:,0]
        yind = voi_img[v]['Coordinates'][:,1]
        zind = voi_img[v]['Coordinates'][:,2]

        # Populate the new image array with the VOI mask
        new_img[xind, yind, zind] = 1

        # Convert dimensions (transpose and flip)
        new_img = np.transpose(new_img, (2,0,1))
        new_img = new_img[::-1, ::-1, ::-1]

        # Extract the relevant VMR region using the mask's coordinates
        extracted_vmr = new_img[-msk_head['ZEnd'] +1 : -msk_head['ZStart']+1,
                                -msk_head['XEnd'] +1 : -msk_head['XStart']+1, 
                                -msk_head['YEnd'] +1 : -msk_head['YStart']+1]

        # Perform voxel counting within downsampling blocks
        factor = msk_head['VTC resolution relative to VMR (1, 2, or 3)']
        voxel_counts = count_voxels_in_blocks(extracted_vmr, factor=factor)

        # Store voxel indices of nonzero values or return the full mask
        if returnfull:
            vox_counts[voi_img[v]['NameOfVOI']] = voxel_counts
        else:
            vox_counts[voi_img[v]['NameOfVOI']] = voxel_counts[np.where(voxel_counts > 0)]

    return vox_counts

def voi_bin_layers(voi_dict, voi_counts, priority_order = [2, 3, 1, 0], outval=0.1):
    """Use voi 'active voxel' counts to estimate most likely layer according to its majority proportion.
     
    Input:
    voi_dict (dict) - Dictionary of xyz indexes for a number of vois
    voi_counts (dict) - Dictionary of voxel count data, as dictionary[keys] being layers, 
                      and interger arrays denoting activation
    priority_order (list) - Optional - Priority order of counts used, which is used in case of a tie (first instance)
                            [2, 3, 1, 0]: give prefference to middle layers before supperficial and deep
    outval (flout) - value to note as outside value (where there is no voi), will force counts of 0 (no activity)
                     to fall in 'no voi'
                     
    "note: if an order of [0, 2, 3, 1] is used, please set outval accordingly to enforce more stickt
           boundary sampling"
    
    Returns:
    dict - Altered voi_dict, now including most active labeling 
    """
    
    # loop over hemispheres
    for hs in voi_counts.keys():

        # Dynamically determine the max shape
        max_x = max(voi_dict[hs][layer][0].max() for layer in voi_dict[hs].keys()) + 1
        max_y = max(voi_dict[hs][layer][1].max() for layer in voi_dict[hs].keys()) + 1
        max_z = max(voi_dict[hs][layer][2].max() for layer in voi_dict[hs].keys()) + 1
        shape = (max_x, max_y, max_z)

        # predefine img dict
        imdict = {}

    ## PUT COUNTS IN ARRAY ##

        # loop over layers
        for layer in voi_dict[hs].keys():

            # get indexes and counts
            indxs = voi_dict[hs][layer]
            counts = voi_counts[hs][layer]

            # Initialize new image array
            new_img = np.zeros(shape)
            # populate
            new_img[indxs] = counts
            # save
            imdict[layer] = new_img

    ## STACK 4D ARRAY ##

        # Convert dictionary values (layer images) into a list and stack along a new axis
        md_array = np.stack([imdict[layer] for layer in sorted(imdict.keys())], axis=0)
        # Create a new layer filled with -1
        negative_layer = outval * np.ones_like(md_array[0])
        # Stack the new layer in front of the existing md_array
        md_array = np.concatenate(([negative_layer], md_array), axis=0)

    ## REORDER BASED ON PRIORITY (IN CASE OF TIE, FIRST MIDDLE ETC.) ##

        # reorder
        md_array_reorder = np.stack([md_array[indx,:,:,:] for indx in priority_order])

        # Compute argmax with NumPy (it will now follow our priority order naturally)
        max_indices = np.argmax(md_array_reorder, axis=0)

        # Convert priority order into a mapping array
        remap_array = np.array(priority_order)
        # Apply the remapping directly
        remapped_indices = remap_array[max_indices]

    ## LOOP OVER LAYERS AND SAVE RESULTS

        # predefine nesting
        voi_dict[f'winning_{hs}'] = {}

        # loop over layers
        for l_idx, layer in enumerate(voi_dict[hs]):
            # get lin index
            voi_dict[f'winning_{hs}'][layer] = np.where(remapped_indices == l_idx+1)

    return voi_dict

def voi_idx_in_masked(msk_idx, submsk_idx):
    """given voi 3d indexes (must be in vtc space using voi_msk) return linear indexes in masked image
    input msk_idx: np.where for 3 dimensions, submsk_idx np.where for 3 dimensions - subset of msk_idx"""
    
    # get maximum length of each dimension - to generate a maximum bounding box
    max_msk = [np.max(xyz) for xyz in msk_idx]
    max_submsk = [np.max(xyz) for xyz in submsk_idx]
    pseudoshape = tuple(np.maximum(max_msk, max_submsk) + 1)
    
    # convert the 3D indices to linear indices
    msk_linear = np.ravel_multi_index(msk_idx, pseudoshape)
    submsk_linear = np.ravel_multi_index(submsk_idx, pseudoshape)

    # find indeces of submask within mask
    result_indices = np.nonzero(np.isin(msk_linear, submsk_linear))[0]
    return result_indices

def intersect_msks(msk_a_idx, msk_b_idx):
    """given voi 3d indexes (must be in vtc space using voi_msk) return linear indexes in masked image
    input msk_idx: np.where for 3 dimensions, submsk_idx np.where for 3 dimensions - subset of msk_idx"""
    
    # get maximum length of each dimension - to generate a maximum bounding box
    max_mska = [np.max(xyz) for xyz in msk_a_idx]
    max_mskb = [np.max(xyz) for xyz in msk_b_idx]
    pseudoshape = tuple(np.maximum(max_mska, max_mskb) + 1)
    
    # convert the 3D indices to linear indices
    msk_a_linear = np.ravel_multi_index(msk_a_idx, pseudoshape)
    msk_b_linear = np.ravel_multi_index(msk_b_idx, pseudoshape)

    # find indeces of submask within mask
    intersected_linear = np.intersect1d(msk_a_linear, msk_b_linear)
    # find corresponding xyz indices for og format
    intersected_xyz = np.unravel_index(intersected_linear, pseudoshape)
    
    return intersected_xyz


def union_msks(msk_a_idx, msk_b_idx):
    """Given voi 3d indexes (must be in VTC space using voi_msk), return linear indexes in masked image
    representing the union of the two masks.
    
    input msk_a_idx: np.where for 3 dimensions
    input msk_b_idx: np.where for 3 dimensions - can be a subset or overlap of msk_a_idx
    """
    
    # Get maximum length of each dimension to generate a maximum bounding box
    max_mska = [np.max(xyz) for xyz in msk_a_idx]
    max_mskb = [np.max(xyz) for xyz in msk_b_idx]
    pseudoshape = tuple(np.maximum(max_mska, max_mskb) + 1)
    
    # Convert the 3D indices to linear indices
    msk_a_linear = np.ravel_multi_index(msk_a_idx, pseudoshape)
    msk_b_linear = np.ravel_multi_index(msk_b_idx, pseudoshape)

    # Find the union of the linear indices from both masks
    union_linear = np.union1d(msk_a_linear, msk_b_linear)
    
    # Convert the linear indices back to 3D indices
    union_xyz = np.unravel_index(union_linear, pseudoshape)
    
    return union_xyz


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
                     error_reg = 'error_[_0-9.]+',
                     acti_reg = 'raw_acti_[_0-9.]+',
                     adpt_reg = 'raw_adapt_[_0-9.]+',
                     acadpt_reg = 'adapt_activ_[_0-9.]+',
                     globerr_reg = '^glob_err',
                     surp_reg = '^surprisal',
                     prec_reg = '^precision',
                     precsurp_reg = '^prec_w_surprisal',
                     onoff_reg = 'onoff'):
    """get all regressors collumn names in one list
    optionally sellect convolved variants"""
    # get one list with all
    columns_reg = []
    # add convolved suffix
    if convolved: suffix='_convolved$'
    else: suffix='$'
    # loop over all, adding suffix
    for reg in ['{}{}'.format(i, suffix) for i in [pred_reg, error_reg, acti_reg, adpt_reg, acadpt_reg, globerr_reg, surp_reg, prec_reg, precsurp_reg, onoff_reg]]:
        columns_reg += df.filter(regex=reg).columns.tolist()
    return(columns_reg)

def get_tw_collumns(df, tp, tw, convolved=False, resampled=False,
                     pred_reg = 'pred_prob_',
                     error_reg = 'error_',
                     acti_reg = 'raw_acti_',
                     adpt_reg = 'raw_adapt_',
                     acadpt_reg = 'adapt_activ_',
                     globerr_reg = '^glob_err',
                     surp_reg = '^surprisal',
                     prec_reg = '^precision',
                     precsurp_reg = '^prec_w_surprisal',
                     exta_reg = 'onoff'):
    """get all by tuning prefference (tp) and tuning width (tw)"""
    # get one list with all
    columns_reg = []
    # add convolved suffix
    if convolved: cs='_convolved'
    else: cs=''
    # add resampled suffix
    if resampled: rs='_resampled'
    else: rs=''
    suffix = '{}{}$'.format(cs,rs)
      
    # adjust column names
    pred_reg = '{}{:.3f}{}'.format(pred_reg, tp, suffix)
    error_reg = '{}{:.3f}_{:.3f}{}'.format(error_reg, tp, tw, suffix)
    acti_reg = '{}{:.3f}_{:.3f}{}'.format(acti_reg, tp, tw, suffix)
    adpt_reg = '{}{:.3f}_{:.3f}{}'.format(adpt_reg, tp, tw, suffix)
    acadpt_reg = '{}{:.3f}_{:.3f}{}'.format(acadpt_reg, tp, tw, suffix)
    globerr_reg = '{}{}'.format(globerr_reg, suffix)
    surp_reg = '{}{}'.format(surp_reg, suffix)
    prec_reg = '{}{}'.format(prec_reg, suffix)
    precsurp_reg = '{}{}'.format(precsurp_reg, suffix)
    exta_reg = '{}{}'.format(exta_reg, suffix)
    
    # loop over all
    for reg in [pred_reg, error_reg, acti_reg, adpt_reg, acadpt_reg, globerr_reg, surp_reg, prec_reg, precsurp_reg, exta_reg]:
        columns_reg += df.filter(regex=reg).columns.tolist()
    return(columns_reg)

def get_collumn_order(pred_reg = 'pred_prob',
                    error_reg = 'error',
                    acti_reg = 'raw_acti',
                    adpt_reg = 'raw_adapt',
                    acadpt_reg = 'adapt_activ',
                    globerr_reg = 'glob_err',
                    surp_reg = 'surprisal',
                    prec_reg = 'precision',
                    precsurp_reg = 'prec_w_surprisal',
                    exta_reg = 'onoff'):
    """get full model column order of coeefs"""
    return([pred_reg, error_reg, acti_reg, adpt_reg, acadpt_reg, globerr_reg, surp_reg, prec_reg, precsurp_reg, exta_reg])

def get_onoff_column(df, convolved=False, resampled=False, onoffreg='onoff'):
    """get onoff column from dataframe, given options for convolvement and resampling / both"""
    
    columns_reg = []
    # add convolved suffix
    if convolved: cs='_convolved'
    else: cs=''
    # add resampled suffix
    if resampled: rs='_resampled'
    else: rs=''
    suffix = '{}{}$'.format(cs,rs)

    # select reg
    name_reg = '{}{}'.format(onoffreg, suffix)
    columns_reg += df.filter(regex=name_reg).columns.tolist()

    return columns_reg

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
    
    # sellect what method to use within the grouping approach, iloc[3] will sellect middle value
    agg_dict = {
        'frequencies': lambda x: x.iloc[3],
        'frequencies_oct': lambda x: x.iloc[3],
        'timing': lambda x: x.iloc[0],
        'closest_volume_rel': lambda x: x.iloc[3],
        'closest_volume_abs': lambda x: x.iloc[3],
        'volume_rel': lambda x: x.iloc[3],
        'volume_abs': lambda x: x.iloc[3],
        'run': lambda x: x.iloc[0],
        'block': lambda x: x.iloc[0],
        'segment': lambda x: x.iloc[0],
        'center_freq_a': lambda x: x.iloc[0],
        'center_freq_b': lambda x: x.iloc[0],
        'center_freq_a_oct': lambda x: x.iloc[0],
        'center_freq_b_oct': lambda x: x.iloc[0],
        'probability_a': lambda x: x.iloc[0],
        'probability_b': lambda x: x.iloc[0],
        'onoff': 'mean',
        'onoff_convolved': 'mean',
    }
    # for the rest of the columns [surprisal, grid of expectations etc.], use 'mean'
    agg_dict.update({col: 'mean' for col in stim_df.columns if col not in agg_dict})

    # also downsameple in a waveform manner, to preserve more data
    all_reg = get_reg_collumns(stim_df, convolved = True)
    if downsample_unconv: all_reg += get_reg_collumns(stim_df, convolved = False)
    downsampled_reg = pd.DataFrame(scipy.signal.resample(stim_df[all_reg], len(volumes_df)),
                                   columns = all_reg)

    # group by 'volume_abs' and aggregate using the specified dictionary
    tr_df = stim_df.groupby('volume_abs').agg(agg_dict).reset_index(drop=True)

    # join resampled data back, assigning as _resampled
    tr_df = tr_df.join(downsampled_reg, rsuffix='_resampled')

    ## --CLEANUP RESAMPLING / WAVEFORM ARTIFACTS RELATED TO ZEROS-- ##
    # take the indexes where the non-resampled values are zero, to prevent boundary effects
    # only needed for high frequency step functions, not for 'convolved' waveforms
    all_cols = get_reg_collumns(stim_df, convolved=False)
    for colname in all_cols:
        indices = tr_df.index[(tr_df[colname] < 1e-10) | (tr_df[f'{colname}_resampled'] < 0)]
        tr_df.loc[indices, f'{colname}_resampled'] = 0
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
    s = eng.genpath(join(dir_path, 'DREX'))
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
    stim_df['surprisal'] = mat['surp_array'][0][:len(stim_df)]
    stim_df['precision'] = np.max(mat['prob_array'][:,:len(stim_df)], axis=0)
    # calculate predicted freqs
    pred_freqs = mat['s_range'][0][np.argmax(mat['prob_array'][:,:len(stim_df)], axis=0)]
    stim_df['glob_err'] = np.abs(pred_freqs - stim_df['frequencies_oct'])
    # Concatenate
    stim_df = pd.concat([stim_df, temp_df[:len(stim_df)]], axis=1)
    return(stim_df)

#IdealObserver
def stims_add_ideal(pp, input_dir, stim_df, pref_range):
    """load ideal observer pre-generated dataframe, calculate per frequency probabilities
    append to dataframe"""

    # load pregenerated idealobserver dataframe with predictive landscape
    with open(join(input_dir, f'{pp}/{pp}_idealobserver.pickle'), 'rb') as handle:
        df_ideal = pickle.load(handle)

    # predefine temporary dataframe
    collumn_names = ['pred_prob_{:.3f}'.format(frq) for frq in pref_range]
    temp_df = pd.DataFrame(columns=collumn_names)

    # extract the necessary parameters from df_ideal
    mus = df_ideal[['mu_A', 'mu_B']].to_numpy().T
    sigmas = df_ideal[['sig_A', 'sig_B']].to_numpy().T
    weights = df_ideal[['wg_A', 'wg_B']].to_numpy().T

    # loop over frequencies (todo: vectorize)
    for frq in pref_range:

        # value we want to estimate probability for
        value_oi = np.repeat(frq, len(stim_df))

        # Estimate the CDF for the new value across all positions
        probabilities, probability, surprise = idealobserver._est_cdf(value_oi, mus, sigmas, weights)
        # handle nans - asuming flat prio
        probability, surprise = idealobserver._interpolate_nans(probability, 
                                                                surprise, 
                                                                len(stim_df['frequencies_oct'].unique()), 
                                                                stim_df['block'])
        
        # append to dataframe
        temp_df['pred_prob_{:.3f}'.format(frq)] = probability

    # append ideal observer surprisal and probabilities back
    _, surprise = idealobserver._interpolate_nans(df_ideal['prob'].to_numpy(), 
                                                  df_ideal['surp'].to_numpy(), 
                                                  len(stim_df['frequencies_oct'].unique()),
                                                  stim_df['block'])
    stim_df['surprisal'] = surprise
    stim_df['precision'] = np.max(temp_df.to_numpy(), axis=1)
    # compute global error (distance between predicted frequency and actual)
    pred_freqs = pref_range[np.argmax(temp_df.to_numpy(), axis=1)]
    stim_df['glob_err'] = np.abs(pred_freqs - stim_df['frequencies_oct'])
    # concatenate back
    stim_df = pd.concat([stim_df, temp_df], axis=1)
    return stim_df

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

# error calculation
def stims_add_error(stim_df, pref_range, sharp_range):
    """
    Add absolute error columns to stim_df by computing the difference between
    raw activations and scaled predicted probabilities.

    Parameters:
    - stim_df (pd.DataFrame): The dataframe containing predicted probabilities and raw activation estimates.
    - pref_range (iterable): List or array of frequency preferences.
    - sharp_range (iterable): List or array of tuning sharpness values.

    Returns:
    - stim_df (pd.DataFrame): Original dataframe with additional error columns.
    """
    
    # Gather predicted probability column names
    pred_names = ['pred_prob_{:.3f}'.format(frq) for frq in pref_range]

    # Scale predicted probabilities globally to a 0–1 range
    max_val = stim_df[pred_names].values.max()
    scaled_cols = stim_df[pred_names] / max_val

    # Get tuning preferences and sharpnesses
    all_idxs = np.arange(len(pref_range) * len(sharp_range))
    tunprefs, tunsharps = longtrace_adaptation.md_get_tuning(all_idxs, pref_range, sharp_range)

    # Build absolute error dataframes
    error_dfs = []
    for pref in pref_range:
        
        # Format the prefix for the activation column names / find all columns starting with
        prefix = 'raw_acti_{:.3f}_'.format(pref)
        matching_cols = [col for col in stim_df.columns if col.startswith(prefix)]
        raw_acts = stim_df[matching_cols]

        # Get the predicted probablility column for this freq
        pred_col = 'pred_prob_{:.3f}'.format(pref)
        pred_vals = scaled_cols[[pred_col]].values

        # calculate actual errors and rename column
        errors = raw_acts - pred_vals
        errors.columns = ['error' + col[8:] for col in matching_cols]

        # Append to dataframe
        error_dfs.append(errors)

    # combine all and take absolute
    all_errors_df = pd.concat(error_dfs, axis=1)
    all_errors_df = np.abs(all_errors_df)
    
    # store back into stim_df
    stim_df = pd.concat([stim_df, all_errors_df], axis=1)

    return stim_df

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


def pulses_check(puls_df, fix=True):
    """Check whether length of puls dataframe is consistent (dependent on how long that run ran for)
    Gives a warning message what runs are non consistent,
    optionally automatically truncate longer running runs"""
    
    # get totalcounts of pulses and mean
    timing_counts = puls_df.groupby('run')['timing'].count()
    median_count = int(timing_counts.median())
    
    # compare with median and give warning if deviation
    if not all(timing_counts == median_count):
        print(f"Warning: Not all timing counts match the value of {median_count}.\n"
             f"Deviation in runs: {', '.join(map(str, timing_counts[timing_counts != median_count].index.tolist()))}\n"
             f"Which had {', '.join(map(str, timing_counts[timing_counts != median_count].tolist()))} pulses instead\n"
             f"Automatic truncation is set to {fix}")
    
        if fix == True:
            # identify runs to runcate
            runs_to_truncate = timing_counts[timing_counts > median_count].index
            # apply the truncation to each group
            puls_df = puls_df.groupby('run').apply(_truncate_run, 
                                                   runlen=median_count, 
                                                   runs_to_truncate=runs_to_truncate).reset_index(drop=True)
    
    return puls_df


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

def save_df(df, input_dir, pp, fn='df'):
    """Save pandas dataframe using the HDFStore/h5 file format.
    Data is saved within pp subfolder and using pp_ as suffix
    input: df: df to be saved
           input_dir: parent directory to use
           pp: subject number - should correspond to directory name
           fn: filename suffix"""

    # open HDFStore file for h5 files
    store = pd.HDFStore(join(input_dir, '{}/{}_{}.h5'.format(pp, pp, fn)))
    # store df in file
    store['df'] = df
    return

def load_df(input_dir, pp, fn='df'):
    """Load pandas dataframe using the HDFStore/h5 file format.
    Data is loaded within pp subfolder and using pp_ as suffix
    input: input_dir: parent directory to use
           pp: subject number - should correspond to directory name
           fn: filename suffix
    return: returns loaded pandas dataframe"""

    # open HDFStore file for h5 files
    store = pd.HDFStore(join(input_dir, '{}/{}_{}.h5'.format(pp, pp, fn)))
    # store df in file
    return store['df']

def vmp_add_realsigma(vmp_img, msk, mustep, oct_col=3, logtransform=np.log2):
    """since the vmp brainvoyager file only contains the octivegrid and a negative variation of the sigma
    we will have to calculate the real sigma ourselves. This function calculates the fwhm vals and 
    the realsigma vallue given this vallue
    input vmp_img (np_array): bv_vmp file, loaded using bvbabel
          msk (np_array): bool array, bv_msk file, loaded using bvbabel
          mustep: (float): mu stepsize within grid
          oct_col (int): default: 3, collumn containing octgrid
          logtransform (function): default np.log2, log or log2, must be concise with matlab
    return pandas dataframe with added vmp vallues appended
          """
    # load tonotopy vmp
    vmp_df = pd.DataFrame(vmp_img[msk]) # 1. prfMU, 2. prfMU_hz, prfS, prfO

    # get octivegrid values from vmp file
    valid_indices = vmp_df[vmp_df[oct_col] != 0].index
    octgridvals = vmp_df.loc[valid_indices,oct_col] # only for valid values

    # calculate fwhm
    fwhmvals = (octgridvals / mustep) + 1
    vmp_df.loc[valid_indices, 'fwhm'] = fwhmvals

    # recalculate realsigma vals needed
    rsigmavals = np.sqrt((mustep * fwhmvals / 2)**2 / (2 * logtransform(2)))
    vmp_df.loc[valid_indices, 'realsigma'] = rsigmavals
    
    # refill zeros
    vmp_df = vmp_df.fillna(0)
    return vmp_df

def generate_voxel_indices(hemispheres, rois, voi_dict, idxs, layers=False,
                           depthLambda = lambda hs: f'WMGM_{hs}_D1-3.voi',
                           roisLambda = lambda hs: f'{hs}_auditory.voi',
                           boundaryLambda = lambda hs: f'{hs}_boundary.voi'):
    """Given a list of hemispheres rois layers and a voi dict, return indx_dict with indexes per sublocation
    input hemispheres: list of hemispheres present in voi_dict or 'combined'
          rois: list of rois present in voi_dict
          idxs: 3x np array of indexes (obtained from scores['indexes']) used to get linear indexing
          layers: default false, list of layers to loop over to subdevide
    returns indx_dict a dictionary with voi indexes per location combination
          """
    
    # Predefine dictionary for voi indexes
    indx_dict = {}

    # Generate full grid
    grid = list(itertools.product(rois, hemispheres))

    # Loop over full grid
    for gp in grid:
        
        # Get current hemisphere, layer, and region
        r = gp[0]
        hs = gp[1]
        
        # Predefine nesting
        indx_dict[f'{gp[0]}_{gp[1]}'] = {}
        
        # If combined, get union
        if hs == 'combined':
            
            # Take union of two masks
            msk_union = union_msks(voi_dict[roisLambda('LH')][r], 
                                           voi_dict[roisLambda('RH')][r])
            # Cut-off boundaries of mask
            boundary_union = union_msks(voi_dict[boundaryLambda('LH')]['in'], 
                                                voi_dict[boundaryLambda('RH')]['in'])
            msk_minBoundary = intersect_msks(msk_union, boundary_union)
            
            # check if we need to subdivide in layers
            if layers != False:
                # Loop over layers
                for l in layers:
                    # Intersect with layer
                    dep_union = union_msks(voi_dict[depthLambda('LH')][l], 
                                                   voi_dict[depthLambda('RH')][l])
                    msk_forlayer = intersect_msks(msk_minBoundary, dep_union)
                    # Get linear indexes from voi indexes
                    voi_idx = voi_idx_in_masked(idxs, msk_forlayer)
                    indx_dict[f'{gp[0]}_{gp[1]}'][l] = voi_idx
            else:
                # get linear indexes from voi indexes
                voi_idx = voi_idx_in_masked(idxs, msk_minBoundary)
                indx_dict[f'{gp[0]}_{gp[1]}'] = voi_idx
        else:
            # Cut-off boundaries of mask
            msk_minBoundary = intersect_msks(voi_dict[roisLambda(hs)][r], 
                                                     voi_dict[boundaryLambda(hs)]['in'])            
            # check if we need to subdivide in layers
            if layers != False:
                # Loop over layers
                for l in layers:
                    # Intersect with layer            
                    msk_forlayer = intersect_msks(msk_minBoundary, voi_dict[depthLambda(hs)][l])
                    # Get linear indexes from voi indexes
                    voi_idx = voi_idx_in_masked(idxs, msk_forlayer)
                    indx_dict[f'{gp[0]}_{gp[1]}'][l] = voi_idx
            else:
                # get linear indexes from voi indexes
                voi_idx = voi_idx_in_masked(idxs, msk_minBoundary)
                indx_dict[f'{gp[0]}_{gp[1]}'] = voi_idx
    return indx_dict


def _truncate_run(group, runlen, runs_to_truncate):
    """helper function, not to be called from outside
    truncate group based on indicated length of single run"""
    if group.name in runs_to_truncate:
        return group.iloc[:runlen]
    return group