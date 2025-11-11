## series of functions to run regressions

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix,  classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
import scipy.io
import re
import itertools

import os
from os.path import join
import contextlib
from copy import deepcopy
import imp 
import time 
import sys
import shutil


# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join('..', 'ses2_modelstims')))

# load local functions
import stim_io
import stim_io_plotting
import vtc
import bvbabel
from stats import compute_permutation_p_values, _test_optimal_chunksize, compute_permutation_p_values_parallel, f_stats, r2_stats


zs=lambda x: (x-x.mean(0)) / (x.std(0))
corr_column= lambda a,b: (zs(a)*zs(b)).mean(0)

def zs_prior(x, prior_entr):
    return((prior_entr-x.mean(0)) /x.std(0))

def run_model_grid(tr_df, stim_df, vmp_df, vtc_fns,
                   msk, vmp_img,
                   pref_range, sharp_range,
                   models, model_regressors,
                   mustep, n_splits, modeltype, key_ai,
                   save_predict=False, convolved=True, resampled=True,
                   zs=True, ts=True, hp=True, dr=True):
    """High level function to run regression model - in a multi-output manner, chucked by all
    grid positions. and looped over different models as defined by the set (3set:7regressions), 
    (2set:3regressions)
    input: tr_df: pandas dataframe - dataframe with regressors in tr domain
           stim_df: pandas dataframe - dataframe with regressors in stimulus domain
           vmp_df: pandas dataframe - vmp file transformed to dataframe of length voxels, containing prf grid positions
           vtc_fns: list - filenames of vtc files - to be used as y
           msk: array - mask array used for vmp processing
           vmp_img: image - vmp image used for processing
           pref_range: array - np.array([[]]) with prf preferences
           sharp_range: array - np.array([[]]) with prf sharpnesses
           models: list - model names to be used by the full set (must be keys in model_regressors)
           model_regressors: dict - dictionary of regressors within each model dict[model] : [regressors]
           mustep: array - step array used for vmp processing
           n_splits: int - number of splits to use
           modeltype: sklearn function - which model type to use, e.g. LinearRegression (ols), Ridge(alpha=..), Lasso(alpha=..)
           key_ai: list - what keys to save
           save_predict: default True - whether to save y_pred
           convolved: default True - whether to use the convolved with HRF or unconvolved column
           resampled: default True - whether to use the scipy frequency domain resampled value, or simple mean
           zs: default True - zscore data per run
           ts: default True - temporally smooth per run - change settings in function default
           hp: default True - highpass per run - change settings in function default
           dr: default True - add drift regressors per block to account for within block drift
    return: scores: nested dict - with per model, per fold, information of model fit"""

    ## --- DEFINING TEST TRAIN SPLIT ---

    # get test train splits per run
    train_matrix, test_matrix = train_test_splits(stim_df, n_splits)


    ## --- BASED ON TUNING PREF AND TW LOAD DESIRED COLUMNS OUT OF DF ---

    # predefine dictionaries for saving regression results
    scores = {}

    # predifine indexes arrays for reconstruction
    idx1 = np.array([], dtype=int)
    idx2 = np.array([], dtype=int)
    idx3 = np.array([], dtype=int)

    # Set numpy to ignore divide and invalid warnings
#   np.seterr(divide='ignore', invalid='ignore')

    ## --- LOOP OVER FULL GRID OF PREFS * TWS ---

    # set chunck indexing and prepare timekeeping
    idx = 0
    st_f = time.time()
    
    # create grid and do full loop
    for tpref, tw in itertools.product(pref_range[0], sharp_range[0]):
        
        # keep track of timing
        st = time.time()
        idx += 1

        # get grid position indexes for tuning pref and tw
        grid_idx = vmp_df.loc[np.isclose(vmp_df[0], tpref, rtol=1e-3) & 
                              np.isclose(vmp_df['realsigma'], tw, rtol=1e-3)].index

        # check if any voxels in gridposition
        if grid_idx.empty:
            continue
        
        # use these grid positions to get a chuck of the mask
        msk_chunk = [i[grid_idx] for i in msk]

        # load vtc for chunk
        y, run_nr = stim_io.load_vtc_chunk_runs(vtc_fns, msk_chunk)
        y = y.transpose() # transpose to make k-fold splits simpler

        # sellect only vallid indices and alter grid_idx and msk chunk accordingly
        y_ind = _find_valid(y, run_nr)
        #if len(y_ind) < len(y): print(f'Found {len(y) - len(y_ind)} voxels without data') #to-check
        grid_idx = grid_idx[y_ind]
        msk_chunk = [i[y_ind] for i in msk_chunk]
        y = y[:,y_ind]

        # check if still voxels after pruning
        if y.shape[-1] == 0:
            continue

        # zscore y, then temporal smooth, and finally highpass
        if zs == True: y = zs_per_run(y, run_nr)
        if ts == True: y = temporal_smooth_per_run(y, run_nr)
        if hp == True: y = highpass_per_run(y, run_nr)

        # load xnames for this columns
        col_names = stim_io.get_tw_collumns(tr_df, tpref, tw, convolved=convolved, resampled=resampled)

        # translate grid positions to mask indexes and save full list
        idx1 = np.concatenate((idx1, msk[0][grid_idx]))
        idx2 = np.concatenate((idx2, msk[1][grid_idx]))
        idx3 = np.concatenate((idx3, msk[2][grid_idx]))

# --- DO ALPHA ESTIMATION IF USING RIDGE OR LASSO MODEL - CV METHOD ---

        # get X for full model 
        col_regressors = extract_sublist(col_names, model_regressors[models[-1]])  # models[-1] is full model
        X = tr_df[col_regressors].to_numpy()

        # for Ridge or Lasso, do per model Alpha estimation - using CV
        if isinstance(modeltype, Ridge):
            best_alpha = estimate_best_alpha_ridge(X, y, n_splits=n_splits)
            modeltype = modeltype.set_params(alpha=best_alpha)
        elif isinstance(modeltype, Lasso):
            best_alpha = estimate_best_alpha_lasso(X, y, n_splits=n_splits)
            modeltype = modeltype.set_params(alpha=best_alpha)

# --- LOOP OVER ALL MODELS FOR DOING SET THEORY --- 
        for model in models:

            # get X for this model 
            col_regressors = extract_sublist(col_names, model_regressors[model])
            X = tr_df[col_regressors].to_numpy()
#           print(f'{model}:\n\t\t{col_regressors}\t{X.shape}\n')

            # zscore X can be done using `X = zs_per_run(X, run_nr)` though non adviced as scaling will be <0
            # normalize X to scale 0-1
            X = stim_io.normalize(X) # CAN CAUSE DEV/0 CONDITION IN VERY RARE INSTANCES

            # add drift regressor if wanted, append to X
            if dr == True:
                dr_array = dr_per_run(run_nr)
                X = np.concatenate((X, dr_array), axis=1)

            # (re)predefine dictionary for storing this models all folds
            nscore = {}

### --- RUN SCRIPT TO DO KFOLDED REGRESSION, AND PARSE OUTPUT IN MEANINGFULL WAY    
            # get cross validated scores based on previously defined test/train splits
            for fold in range(len(train_matrix)):
                # save indexes for text number in array
                train_idx = np.argwhere(np.in1d(run_nr, train_matrix[fold])).flatten()
                test_idx = np.argwhere(np.in1d(run_nr, test_matrix[fold])).flatten()

                # select train and test sets for fold
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # do the regression
                nscore[fold+1] = model_fit(modeltype, 
                                           X_train,
                                           X_test,
                                           y_train,
                                           y_test,
                                           save_predict=save_predict)

            # check if we update previous chunk or initate dict using this one
            if model not in scores:
                scores[model] = nscore
            else:
                # combine all the folds into one dictionary - aditionally append chunks
                for fold in range(len(train_matrix)):
                    for key in scores[model][fold+1]:
                        if key != 'predict': 
                            scores[model][fold+1][key] = np.concatenate((scores[model][fold+1][key],
                                                                         nscore[fold+1][key]))
                        if key == 'predict': # since predict is nested, handle seperately
                            scores[model][fold+1]['predict'] = {'y_pred' : np.concatenate((scores[model][fold+1]['predict']['y_pred'],
                                                                                       nscore[fold+1]['predict']['y_pred']), axis=1),
                                                                'y' : np.concatenate((scores[model][fold+1]['predict']['y'],
                                                                                       nscore[fold+1]['predict']['y']), axis=1)}

### --- DO ANOTHER REGRESSION WITHOUT CROSS VALIDATION, TO VALIDATE RESULTS ---
            # get non-crossvalidated scores
            nscore_noncv = non_cv_fit(modeltype, 
                                        X,
                                        y,
                                        save_predict=save_predict)
            # nest 'score' inside an array to allow for concatenation
            nscore_noncv['score'] = np.array([nscore_noncv['score']]) 
            # check if exist otherwise add to existing
            if 'non-cv' not in scores[model]:
                scores[model]['non-cv'] = nscore_noncv
            else:
                for key in scores[model]['non-cv']:
                    if key != 'predict': 
                        scores[model]['non-cv'][key] = np.concatenate((scores[model]['non-cv'][key],
                                                                       nscore_noncv[key]))
                    if key == 'predict':
                        scores[model]['non-cv']['predict'] = {'y_pred' : np.concatenate((scores[model]['non-cv']['predict']['y_pred'],
                                                                                   nscore_noncv['predict']['y_pred']), axis=1),
                                                              'y' : np.concatenate((scores[model]['non-cv']['predict']['y'],
                                                                                   nscore_noncv['predict']['y']), axis=1)}

        # update user on progress
        print(f"""grid: {idx}/{len(pref_range[0]) * len(sharp_range[0])}, 
        -current chuck took: {time.time()-st:.2f} seconds
        -estimated time elapsed: {(time.time()-st_f) / 60:.2f} minutes of {(len(pref_range[0]) * len(sharp_range[0])) * (time.time() - st_f) / idx / 60:.2f} minutes""")


    ### --- STORE MEDIAN AND MEAN ACROSS FOLDS ---
    for model in models:
        for k in key_ai:
            cv_scores = {'median':  np.median(np.array([scores[model][fold+1][k] for 
                                                        fold in range(len(train_matrix))]), axis=0),
                         'mean':   np.nanmean(np.array([scores[model][fold+1][k] for 
                                                        fold in range(len(train_matrix))]), axis=0)}
            scores[model][k] = cv_scores

    # save indexes
    scores['indexes'] = (idx1, idx2, idx3)
    # save regressor labels
    for model in models:
        scores[model]['feature_labels'] = model_regressors[model]
    
    return scores

def run_model_grid_noncv(tr_df, stim_df, vmp_df, vtc_fns,
                   msk, vmp_img,
                   pref_range, sharp_range,
                   models, model_regressors,
                   mustep, modeltype, key_ai,
                   save_predict=False, convolved=True, resampled=True,
                   zs=True, ts=True, hp=True, dr=True):
    """High level function to run regression model - in a multi-output manner, chucked by all
    grid positions. and looped over different models as defined by the set (3set:7regressions), 
    (2set:3regressions)
    input: tr_df: pandas dataframe - dataframe with regressors in tr domain
           stim_df: pandas dataframe - dataframe with regressors in stimulus domain
           vmp_df: pandas dataframe - vmp file transformed to dataframe of length voxels, containing prf grid positions
           vtc_fns: list - filenames of vtc files - to be used as y
           msk: array - mask array used for vmp processing
           vmp_img: image - vmp image used for processing
           pref_range: array - np.array([[]]) with prf preferences
           sharp_range: array - np.array([[]]) with prf sharpnesses
           models: list - model names to be used by the full set (must be keys in model_regressors)
           model_regressors: dict - dictionary of regressors within each model dict[model] : [regressors]
           mustep: array - step array used for vmp processing
           modeltype: sklearn function - which model type to use, e.g. LinearRegression (ols), Ridge(alpha=..), Lasso(alpha=..)
           key_ai: list - what keys to save
           save_predict: default True - whether to save y_pred
           convolved: default True - whether to use the convolved with HRF or unconvolved column
           resampled: default True - whether to use the scipy frequency domain resampled value, or simple mean
           zs: default True - zscore data per run
           ts: default True - temporally smooth per run - change settings in function default
           hp: default True - highpass per run - change settings in function default
           dr: default True - add drift regressors per block to account for within block drift
    return: scores: nested dict - with per model information of model fit"""

    ### --- BASED ON TUNING PREF AND TW LOAD DESIRED COLUMNS OUT OF DF ---

    # predefine dictionaries for saving regression results
    scores = {}

    # predifine indexes arrays for reconstruction
    idx1 = np.array([], dtype=int)
    idx2 = np.array([], dtype=int)
    idx3 = np.array([], dtype=int)

    ## --- LOOP OVER FULL GRID OF PREFS * TWS ---

    # set chunck indexing and prepare timekeeping
    idx = 0
    st_f = time.time()
    
    # create grid and do full loop
    for tpref, tw in itertools.product(pref_range[0], sharp_range[0]):
        
        # keep track of timing
        st = time.time()
        idx += 1

        # get grid position indexes for tuning pref and tw
        grid_idx = vmp_df.loc[np.isclose(vmp_df[0], tpref, rtol=1e-3) & 
                              np.isclose(vmp_df['realsigma'], tw, rtol=1e-3)].index

        # check if any voxels in gridposition
        if grid_idx.empty == False:

            # use these grid positions to get a chuck of the mask
            msk_chunk = [i[grid_idx] for i in msk]

            # load vtc for chunk
            y, run_nr = stim_io.load_vtc_chunk_runs(vtc_fns, msk_chunk)
            y = y.transpose() # transpose to make k-fold splits simpler

            # sellect only vallid indices and alter grid_idx and msk chunk accordingly
            y_ind = _find_valid(y, run_nr)
            #if len(y_ind) < len(y): print(f'Found {len(y) - len(y_ind)} voxels without data') #to-check
            grid_idx = grid_idx[y_ind]
            msk_chunk = [i[y_ind] for i in msk_chunk]
            y = y[:,y_ind]
            
            # zscore y, then temporal smooth, and finally highpass
            if zs == True: y = zs_per_run(y, run_nr)
            if ts == True: y = temporal_smooth_per_run(y, run_nr)
            if hp == True: y = highpass_per_run(y, run_nr)

            # load xnames for this columns
            col_names = stim_io.get_tw_collumns(tr_df, tpref, tw, convolved=convolved, resampled=resampled)

            # translate grid positions to mask indexes and save full list
            idx1 = np.concatenate((idx1, msk[0][grid_idx]))
            idx2 = np.concatenate((idx2, msk[1][grid_idx]))
            idx3 = np.concatenate((idx3, msk[2][grid_idx]))

    # --- DO ALPHA ESTIMATION IF USING RIDGE OR LASSO MODEL - CV METHOD ---

            # get X for full model 
            col_regressors = extract_sublist(col_names, model_regressors[models[-1]])  # models[-1] is full model
            X = tr_df[col_regressors].to_numpy()

            # for Ridge or Lasso, do per model Alpha estimation - using CV
            if isinstance(modeltype, Ridge):
                best_alpha = estimate_best_alpha_ridge(X, y, n_splits=n_splits)
                modeltype = modeltype.set_params(alpha=best_alpha)
            elif isinstance(modeltype, Lasso):
                best_alpha = estimate_best_alpha_lasso(X, y, n_splits=n_splits)
                modeltype = modeltype.set_params(alpha=best_alpha)

    # --- LOOP OVER ALL MODELS FOR DOING SET THEORY --- 
            for model in models:
            
                # predefine nesting for saving
                if model not in scores:
                    scores[model] = {}
       
                # get X for this model 
                col_regressors = extract_sublist(col_names, model_regressors[model])
                X = tr_df[col_regressors].to_numpy()
                #print(f'{model}:\n\t\t{extract_sublist(col_names, model_regressors[model])}\n')

                # normalize X to scale 0-1
                X = stim_io.normalize(X)
                
                ## HIER GEGLEVEN - ADDING DRIFTS CHECK IF CORRECT AFTER HOLIDAY!!! 
                # add drift regressor if wanted, append to X
                if dr == True:
                    dr_array = dr_per_run(run_nr)
                    X = np.concatenate((X, dr_array), axis=1)
                    
    # --- DO REGRESSION WITHOUT CROSS VALIDATION ---
                # get non-crossvalidated scores
                nscore_noncv = non_cv_fit(modeltype, 
                                            X,
                                            y,
                                            save_predict=save_predict)
                # nest 'score' inside an array to allow for concatenation
                nscore_noncv['score'] = np.array([nscore_noncv['score']]) 
                # check if exist otherwise add to existing
                if 'non-cv' not in scores[model]:
                    scores[model]['non-cv'] = nscore_noncv
                else:
                    for key in scores[model]['non-cv']:
                        if key != 'predict': 
                            scores[model]['non-cv'][key] = np.concatenate((scores[model]['non-cv'][key],
                                                                           nscore_noncv[key]))
                        if key == 'predict':
                            scores[model]['non-cv']['predict'] = {'y_pred' : np.concatenate((scores[model]['non-cv']['predict']['y_pred'],
                                                                                       nscore_noncv['predict']['y_pred']), axis=1),
                                                                  'y' : np.concatenate((scores[model]['non-cv']['predict']['y'],
                                                                                       nscore_noncv['predict']['y']), axis=1)}

                    
        # update user on progress
        print(f"""grid: {idx}/{len(pref_range[0]) * len(sharp_range[0])}, 
        -current chuck took: {time.time()-st:.2f} seconds
        -estimated time elapsed: {(time.time()-st_f) / 60:.2f} minutes of {(len(pref_range[0]) * len(sharp_range[0])) * (time.time() - st_f) / idx / 60:.2f} minutes""")

    # save indexes
    scores['indexes'] = (idx1, idx2, idx3)
    # save regressor labels
    for model in models:
        scores[model]['feature_labels'] = model_regressors[model]
    
    return scores


def train_test_splits(df, n_splits, shuffle=True, random_state=123, group='run'):
    """ Main function for getting an array of train test splits
    
    input: df        : pandas dataframe, to obtain 'group' from
           n_splits  : number of train/test splits (k-folds)
           shuffle   : (default true) whether to shuffle KFolds
           random_state : (default 123) the random seed to use
           group     : (default 'run') the group to use for splitting
    
    returns: training indexes : [n_splits * groups in training]
             testing_indexes  : [n_splits * groups in testing]"""

    # use sklearn tool for getting kfolds
    kf = KFold(n_splits=n_splits, 
               shuffle=True,
               random_state=123)

    # get runs
    runz = df['run'].unique()

    # save empty list for train and test texts
    train_indexes = np.empty((0, int(len(runz)-len(runz)/n_splits)), int)
    test_indexes = np.empty((0, int(len(runz)/n_splits)), int)

    # loop for text in folds
    for train_index, test_index in kf.split(runz):
        train_indexes = np.append(train_indexes, 
                                  np.array([runz[train_index]]), 
                                  axis=0)
        test_indexes = np.append(test_indexes,
                                 np.array([runz[test_index]]),
                                 axis=0)
        
    # return train and test indexes
    return(train_indexes, test_indexes)
    
def model_fit(model, X_train,X_test,y_train,y_test, save_predict=False):
    """take the linear model, the train/test data and do scoring
    input:  model:    input model type
            X_train:  regressor input train data
            X_test:   regressor input test data
            y_train:  target train data
            y_test:   target test data
            save_predict:   save individual trail-by-trail predictions
    output: return scores dictionary
    """
    scores = {}
    
    # fit model, get y_pred, model proba and nullmodel proba
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # get model score, classification report, confustion matrix and coefs/intercepts
    scores['score'] = np.array([model.score(X_test, y_test)]) # place in array to make chunking easier
    scores['raw_scores'] = r2_stats(y_test, y_pred)
    scores['coefs'] = model.coef_[:]
    scores['intercepts'] = model.intercept_
    scores['correlation'] = corr_column(y_test, y_pred)
    
#     # get a (oncorrected) p-val estimation
#     scores['p_value'] = compute_permutation_p_values_parallel(model, X_train, y_train, 
#                                                               X_test, y_test, 
#                                                               observed_scores=scores['raw_scores'], 
#                                                               n_permutations=200, chunk_size=35)

    # save prediction scores if flagged
    if save_predict:
        scores['predict'] = {'y_pred': y_pred, 'y': y_test}
    
    # return cross validated scores
    return(scores)

def non_cv_fit(model, X, y, save_predict=False):
    """take the linear model, the train/test data and do scoring
    input:  model:    input model type
            X:        regressors
            y:        target var
            save_predict:   save individual trail-by-trail predictions
    output: return scores dictionary
    """
    scores = {}
    
    # fit model, get y_pred, model proba and nullmodel proba
    model.fit(X, y)
    y_pred = model.predict(X)
    #scores['modelproba'] = model.predict_proba(X)
    #scores['nullproba'] = pred_proba_null(y)
    
    # get model score, classification report, confustion matrix and coefs/intercepts
    scores['score'] = model.score(X, y)
    scores['raw_scores'] = r2_stats(y, y_pred)
    #scores['conf_mat'] = (confusion_matrix(y, y_pred).T / confusion_matrix(y, y_pred).sum(axis=1)).T
    scores['coefs'] = model.coef_[:]
    scores['intercepts'] = model.intercept_
    scores['correlation'] = corr_column(y, y_pred)
    
    # calculate f-stats
    scores['fmap'], scores['p_value'] = f_stats(y, y_pred, len(y), X.shape[1])
#     scores['tmap'], scores['p_value'] = t_stats(X, y, y_pred, model.coef_[:], len(y), X.shape[1], contrast=[0,0,1,1,-1,-1])
    
    # save prediction scores if flagged
    if save_predict:
        scores['predict'] = {'y_pred': y_pred, 'y': y}
    
    # return scores    
    return(scores)

def zs_per_run(x, runs):
    """z-score x/y in dimension 1 given a list of runs of same length
    e.g. if x = (2820, 5) then runs should be shape (2820) giving the indexes of runs
    the resulting x is in this case zscored in the '2820' direction, devided in runs"""

    # z-scoring across the zerod axis
    zs=lambda i: (i-i.mean(0)) /i.std(0)

    # z-score per run
    for run in np.unique(runs):
        x[runs == run] = zs(x[runs == run])
    # get rid of nans and return
    return(np.nan_to_num(x))

def temporal_smooth_per_run(y, runs, sigma=1):
    """do temporal smoothing per run
    Friston, K. J., Holmes, A. P., Poline, J.-B., Grasby, P. J., Williams, S. C. R., Frackowiak, R. S. J. and Turner, R. (1995) Analysis of fMRI Time-Series Revisited. Neuroimage 2,45-53. 
    https://users.fmrib.ox.ac.uk/~stuart/thesis/chapter_6/section6_2.html
    recomended smoothing 2.8 seconds, - 2.8/TR=sigma = 1.56 (atm we take slighly shorter)
    """
    
    # temporal smooth per run
    for run in np.unique(runs):
        y[runs == run] = gaussian_filter1d(y[runs == run], sigma=sigma, axis=0)
    # get rid of nans and return
    return y


def highpass_per_run(y, runs, TR=1.8, cutoff=0.01):
    """apply highpass filter per run"""

    # design a Butterworth high-pass filter
    b, a = butter(N=4, Wn=cutoff, btype='highpass', fs=TR)
    
    # temporal smooth per run
    for run in np.unique(runs):
        # apply the filter along the time axis (axis=0)
        y[runs == run] = filtfilt(b, a, y[runs == run], axis=0)
    return y

def dr_per_run(runs, space=[0,1]):
    """get a drift regressor per run using a liniar space from 0-1
    input runs, with indexes of the runs and optionally the linear space to use,
    return a 1d vector of length runs"""

    # predefine driftarray
    driftarray = np.zeros((len(runs), len(np.unique(runs))))
    
    # loop over runs
    for run in np.unique(runs):
        indices = np.where(runs == run)[0]  # Get the indices for the current run
        driftarray[indices, np.array(int(run-1))] = np.linspace(space[0], 
                                                                space[1], 
                                                                np.sum(runs == run))
    return driftarray

def extract_sublist(lst, prefixes):
    return [item for item in lst for prefix in prefixes if re.match(f"^{prefix}", item)]

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

def estimate_best_alpha_ridge(X, y, n_splits=6, alphas=None):
    """
    Estimate the best alpha for Ridge regression using cross-validation.
    
    Parameters:
    - X: np.array, feature matrix
    - y: np.array, target matrix
    - n_splits: int, number of splits for cross-validation
    - alphas: list or np.array, list of alpha values to test (default: np.logspace(-6, 6, 13))
    
    Returns:
    - best_alpha: float, the best alpha value found by cross-validation
    """
    if alphas is None:
        alphas = np.logspace(-6, 6, 13)
    
    # Create parameter grid
    param_grid = {'alpha': alphas}
    
    # Initialize Ridge model and GridSearchCV
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=n_splits, scoring='neg_mean_squared_error')
    
    # Fit GridSearchCV on the provided data
    grid_search.fit(X, y)
    
    # Get the best alpha value
    best_alpha = grid_search.best_params_['alpha']
    
    return best_alpha

def estimate_best_alpha_lasso(X, y, n_splits=6, alphas=None):
    """
    Estimate the best alpha for Lasso regression using cross-validation.
    
    Parameters:
    - X: np.array, feature matrix
    - y: np.array, target matrix
    - n_splits: int, number of splits for cross-validation
    - alphas: list or np.array, list of alpha values to test (default: np.logspace(-6, 6, 13))
    
    Returns:
    - best_alpha: float, the best alpha value found by cross-validation
    """
    if alphas is None:
        alphas = np.logspace(-6, 6, 13)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create parameter grid
    param_grid = {'alpha': alphas}
    
    # Initialize Lasso model and GridSearchCV
    lasso = Lasso(max_iter=10000)
    grid_search = GridSearchCV(lasso, param_grid, cv=n_splits, scoring='neg_mean_squared_error')
    
    # Fit GridSearchCV on the standardized data
    grid_search.fit(X_scaled, y)
    
    # Get the best alpha value
    best_alpha = grid_search.best_params_['alpha']
    
    return best_alpha

def _find_valid(arr, runs):
    """prune an array and sellect only rows with data
    do this on a per run basis, meaning one run without data will discount the full voxel
    if you use for y, make sure axis=0 is the correct location (transposed etc.)
    return indices"""

    # start with all indices as potentially valid
    valid_indices = np.arange(arr.shape[1])
    
    # loop over runs
    for run in np.unique(runs):
        
        # sellect only valid
        run_indices = np.where(np.mean(arr[runs == run], axis=0) != 0)[0]
        # intersect with currently valid indices
        valid_indices = np.intersect1d(valid_indices, run_indices)

    return valid_indices