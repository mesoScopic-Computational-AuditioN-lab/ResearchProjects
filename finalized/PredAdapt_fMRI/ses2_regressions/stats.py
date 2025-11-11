import numpy as np
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

import scipy 
import scipy.stats as st

import time

def f_stats(y, y_pred, n, k):
    """given y and y_pred (typically non cross-validated) return f-statistics and p-values
    computation is optimized for vectorized data (i.e. multi-output regressions)
    input: y: (observations * outputs) 2d numpy array
           y_pred: (observations * outputs) 2d numpy array - obtained from model.predict(X)
           n: (int) number of observations/TRs
           k: (int) number of predictors - X.shape[1]
    return: F (F-map values array of length outputs), p_value (p-values array of length outputs)"""

    # residuals
    residuals = y - y_pred
    
    # calculate RSS & TSS & ESS
    RSS = np.sum(residuals ** 2, axis=0)
    TSS = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    ESS = TSS - RSS
    
    # calculate F-statistic
    F = (ESS / k) / (RSS / (n - k - 1))
    
    # calculate p-value from F-distribution
    p_value = st.f.sf(F, k, n - k - 1)
    return F, p_value

def t_stats(X, y, y_pred, beta, n, p, contrast, speudo_solver = 'qr_decomp'):
    """given X, y and y_pred return t-statistics and p-values for a given contrast
    computation currently only works for single output regression (1d y) - but can be vectorized
    input: X: (designmatrix) 2d numpy array
           y: (observations) 1d numpy array
           y_pred: (observations) 2d numpy array - obtained from model.predict(X)
           beta: (np array) 1d numpy array - obtained from model.coef_
           n: (int) number of observations/TRs (len(y))
           p: (int) number of predictors (X.shape[1])
           contrast: (contrasts x observations) 1d or 2d must match number of regressors and be balanced,  
           speudo_solver: 'qr_decomp' or 'cholesky' for stability or speed
    return: T (T stats value), p_value (p-value)"""
    
    # compute the residuals and variance of residuals
    residuals = y - y_pred
    residual_variance = np.var(residuals, ddof=p)
    
    # degrees of freedom
    df = n - p
    
    # use qr_decomp (numerically stable)
    if speudo_solver == 'qr_decomp':
        # decomposition of X and calculate the inverse of R (upper triangular matrix)
        Q, R = qr(X, mode='economic')
        R_inv = solve_triangular(R, np.eye(R.shape[0]))
        # use r to get the equivalent of inverse X.T@X
        XTX_inv = R_inv @ R_inv.T
    # use cholesky (computationally more efficient)
    elif speudo_solver == 'cholesky':
        # compute X.T dot X
        XTX = np.dot(X.T, X)
        # decomposition of X and calculate the inverse of L (lower triangular matrix)
        L = cholesky(XTX, lower=True)
        L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
        # use l to get the equivelent of inverse X.T @ X
        XTX_inv = L_inv.T @ L_inv
    
    # ensure contrast is 2d 
    if contrast.ndim == 1:
        contrast = contrast[np.newaxis,:]
    
    # Initialize lists to store t-stats and p-values
    t_stats = np.zeros(contrast.shape[0])
    p_values = np.zeros(contrast.shape[0])
    
    # loop over contrasts
    for c in range(len(contrast)):

        # compute the standard error of the contrast
        std_error = np.sqrt(residual_variance * np.dot(np.dot(contrast[c,:].T, XTX_inv), contrast[c,:]))

        # compute the t-statistic
        t_stats[c] = np.dot(contrast[c,:], beta) / std_error

        # compute the p-value for the two-sided test
        p_values[c] = 2 * (1 - st.t.cdf(np.abs(t_stats[c]), df))
    
    return t_stats, p_values

def r2_stats(y, y_pred):
    """given y and y_pred return r2-stats 
    computation is optimized for vectorized data (i.e. multi-output regressions)
    input: y: (observations * outputs) 2d numpy array
           y_pred: (observations * outputs) 2d numpy array - obtained from model.predict(X)
    return: r2values (F-map values array of length outputs)"""

    # residuals
    residuals = y - y_pred
    
    # calculate RSS & TSS & ESS
    RSS = np.sum(residuals ** 2, axis=0)
    TSS = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    
    # calculate R2-statistic
    R2 = 1 - (RSS / TSS)
    return R2

def bootstrap_ci(x,alpha=0.05,bootnum=10000,avgfun=np.mean):
    """compute confidence intervals around the mean/median using bootstrapping
    in:
    -x: ndarray, repetitions across first dimension
    -alpha: float. alpha level (1-alpha) confidence interval
    -bootnum: int. number of bootstraps (10000 default)
    -avgfun: callable (e.g. np.mean or np.median). requires 'axis' keyword
    
    out:
    -left,right: lower and upper bounds
    """

    bootfunc=lambda x: avgfun(x,axis=0)
    x_bootmean=bootstrap(x,bootfunc=bootfunc,bootnum=bootnum)
    left=np.percentile(x_bootmean,alpha/2*100,axis=0)
    right=np.percentile(x_bootmean,100-alpha/2*100,axis=0)
    return(left,right)

def bootstrap_se(x,bootnum=10000,avgfun=np.mean):
    """compute confidence standard error of the mean/median using bootstrapping
    in:
    -x: ndarray, repetitions across first dimension
    -alpha: float. alpha level (1-alpha) confidence interval
    -bootnum: int. number of bootstraps (10000 default)
    -avgfun: callable (e.g. np.mean or np.median). requires 'axis' keyword
    
    out:
    -left,right: lower and upper bounds
    """
    bootfunc=lambda x: avgfun(x,axis=0)
    x_bootmean=bootstrap(x,bootfunc=bootfunc,bootnum=bootnum)
    se=np.std(x_bootmean,axis=0)
    return(se)

def bootstrap_BET(samps_in,pop_mean=0,tail='2s',n_boots=10e4):
    """one-sample (paired) bootstrap t-test; returns p-value only
                    
    in:
    - samps: nd.array, shape(n_samples)
        datapoints 
    - pop_mean: float, Default=0
        mean to test against
    - tail: str, default: '2s'
        options: '2s','l','r' (for two-tailed,left or right-tailed)
    - n_boots: int; default=10e3
        number of bootstraps (determines precision)
    
    out:
    -pval: float 
        fraction of instances where simulated null distribution returns
        test statistic that is at least as extreme as emprical test stat.
        
    dependencies: bootstrap from astropy
    """
    seed = np.random.randint(1000)
    # test stat 
    t_func=lambda x,dim:(x.mean(dim)-pop_mean)/(x.std(dim)/np.sqrt(x.shape[dim]))
    # make null distribution 
    null_boot_test=t_func(bootstrap(samps_in-samps_in.mean(0)+pop_mean,bootnum=int(n_boots), 
                                    seed=seed),1)
    emp_test=t_func(samps_in,0)
    
    # make alt distribution
    alt_distr=bootstrap(samps_in,bootnum=int(n_boots), 
                        seed=seed)
    alpha_distr=alt_distr.mean()-null_boot_test

    # if distribution = 0
    if (alpha_distr>0).sum() == 0:
        omega = 1
    else:
        omega= (alpha_distr<=0).sum() / (alpha_distr>0).sum()

    # get mu, sigma and sample size
    mu = alpha_distr.mean()
    sigma = alpha_distr.std()
    n = len(samps_in)

    # calculate evidence strength
    evidence_strength = np.log( ((1-omega) * mu + (omega*sigma))   /   ((1 + omega * np.log(n)) * sigma))
    return(evidence_strength)

def fmt_boot_pval(pval,n_boots=10e4,scientific=False) -> str:
    """convertr bootstrap pvalues to expression that takes into account precision 
    (e.g. p=0 will become p < x, with x being determined by number of bootstraps)"""
    if scientific:
        p_str= f'p={pval}' if pval>0 else f'p < {round(1/float(n_boots),int(np.log10(n_boots)+1))}'
    else:
        if pval>0:p_str=f'p={pval}'
        else:p_str='p < {atleast:.{decim}f}'.format(atleast=1/float(n_boots),decim=int(np.log10(n_boots)))
    return(p_str)

def bootstrap_t_onesample(samps_in,pop_mean=0,tail='2s',n_boots=10e4,seed=123):
    """one-sample (paired) bootstrap t-test; returns p-value only
                    
    in:
    - samps: nd.array, shape(n_samples)
        datapoints 
    - pop_mean: float, Default=0
        mean to test against
    - tail: str, default: '2s'
        options: '2s','l','r' (for two-tailed,left or right-tailed)
    - n_boots: int; default=10e3
        number of bootstraps (determines precision)
    
    out:
    -pval: float 
        fraction of instances where simulated null distribution returns
        test statistic that is at least as extreme as emprical test stat.
    see also:
    - fmt_boot_pval, function to format the pvalues, changes p=0 into P < (1/n_boots) statement 
    dependencies: bootstrap from astropy
    """
    t_func=lambda x:(x.mean(0)-pop_mean)/(x.std()/np.sqrt(x.shape[0]))

    # test stat 
    t_func=lambda x,dim:(x.mean(dim)-pop_mean)/(x.std(dim)/np.sqrt(x.shape[dim]))
    # make null distribution 
    null_boot_test=t_func(bootstrap(samps_in-samps_in.mean(0)+pop_mean,bootnum=int(n_boots),seed=seed),1)
    emp_test=t_func(samps_in,0)
    # return p-value as probability of obtaining a test stat at least as extreme under the null 
    if tail in ['2s','two','both']:
        left_pval=np.mean(null_boot_test<emp_test)
        right_pval=np.mean(null_boot_test>emp_test)
        return(2*min(left_pval,right_pval))
    elif tail.lower() in ['l','left']:
        return(np.mean(null_boot_test<emp_test))
    elif tail.lower() in ['r','right']:
        return(np.mean(null_boot_test>emp_test))
    else:
        raise ValueError('tail not recognised!')
        
flatten= lambda l: np.array([item for sublist in l for item in sublist])

        
#-------------------- copy bootstrap to have no dependencies 
def bootstrap(data, bootnum=1000, samples=None, bootfunc=None, seed=False):
    """Performs bootstrap resampling on numpy arrays. (FUNCTION FROM ASTROPY)

    Bootstrap resampling is used to understand confidence intervals of sample
    estimates. This function returns versions of the dataset resampled with
    replacement ("case bootstrapping"). These can all be run through a function
    or statistic to produce a distribution of values which can then be used to
    find the confidence intervals.

    Parameters
    ----------
    data : numpy.ndarray
        N-D array. The bootstrap resampling will be performed on the first
        index, so the first index should access the relevant information
        to be bootstrapped.
    bootnum : int, optional
        Number of bootstrap resamples
    samples : int, optional
        Number of samples in each resample. The default `None` sets samples to
        the number of datapoints
    bootfunc : function, optional
        Function to reduce the resampled data. Each bootstrap resample will
        be put through this function and the results returned. If `None`, the
        bootstrapped data will be returned

    Returns
    -------
    boot : numpy.ndarray

        If bootfunc is None, then each row is a bootstrap resample of the data.
        If bootfunc is specified, then the columns will correspond to the
        outputs of bootfunc.

    """
    if seed != False:
        np.random.seed(seed)
    
    if samples is None:
        samples = data.shape[0]

    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")

    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
        # test number of outputs from bootfunc, avoid single outputs which are
        # array-like
        try:
            resultdims = (bootnum, len(bootfunc(data)))
        except TypeError:
            resultdims = (bootnum,)

    # create empty boot array
    boot = np.empty(resultdims)

    for i in range(bootnum):
        bootarr = np.random.randint(low=0, high=data.shape[0], size=samples)
        
#         if seed != False: print(bootarr)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr])

    return boot

def compute_permutation_p_values(model, X_train, y_train, X_test, y_test, observed_scores=None, n_permutations=100):
    """
    Compute p-values for each output (voxel) using permutation testing with refitting the model
    
    Args:
        model: A scikit-learn model instance that supports the fit and predict methods
        X_train: Training data features (shape: [n_samples_train, n_features])       
        y_train: Training data target values (shape: [n_samples_train, n_outputs])   
        X_test: Test data features (shape: [n_samples_test, n_features])             
        y_test: Test data target values (shape: [n_samples_test, n_outputs])         
        observed_scores: Precomputed R2 scores from the original y_test and y_pred, if None: compute
        n_permutations: Number of permutations to perform for p-value computation      
    
    Returns:
        p_values: An array of p-values for each output.
    """
    st = time.time()
    # get number of samples in train and number of outputs outputes
    n_samples, n_outputs = y_train.shape
    
    # calculate r2 for pred/test
    if observed_scores is None:
        y_pred = model.predict(X_test) #note:before refitting i.e. old model
        observed_scores = r2_score(y_test, y_pred, multioutput='raw_values')  # Or any other metric
        
    # Generate a random 2D array to shuffle the indices and argsort for random permutation of indexes
    random_shuffles = np.random.default_rng().random((n_permutations, n_samples))
    perm_indices = np.argsort(random_shuffles, axis=1) #shape(datapoints : permutations)
    
    # create grid of permuted: taking y_train(datapoints : n_outputs) and permuting them with perm_indices
    #  this works since y_train[perm_indices] will result in (permutations;datapoints;n_outputs) data
    #  which we transpose/reshape to stack n_outputs with permutations (in order to only do 1 additional modelfit)
    y_perm_grid = y_train[perm_indices].transpose(2,0,1).reshape(-1,n_samples).T  #shape(datapoints : n_outputs*perms)

    # refit the model and predict using permuted outputs
    model.fit(X_train, y_perm_grid)
    y_perm_pred = model.predict(X_test)

    # repeat y_test number of permutations times, for testing in one go
    #  stacking y_test to match y_perm_grid (basically repeating design matrix)
    y_test_rep = np.repeat(y_test[np.newaxis, :, :], n_permutations, axis=0).transpose(2,0,1).reshape(-1,y_test.shape[0]).T
    perm_scores = r2_score(y_test_rep, y_perm_pred, multioutput='raw_values')

    # calculate p-stats
    p_values = np.mean(perm_scores.reshape((n_permutations, -1)) >= observed_scores, axis=0)
    print(f'permutation test took: {time.time()-st}')
    return p_values


def compute_permutation_p_values_parallel(model, X_train, y_train, X_test, y_test, observed_scores=None, 
                                 n_permutations=1000, n_jobs=-1, chunk_size=40):
    """
    Compute p-values for each output (voxel) using permutation testing with refitting the model in parallel with chunking.

    Args:
        model: A scikit-learn model instance that supports the fit and predict methods
        X_train: Training data features (shape: [n_samples_train, n_features])       
        y_train: Training data target values (shape: [n_samples_train, n_outputs])   
        X_test: Test data features (shape: [n_samples_test, n_features])             
        y_test: Test data target values (shape: [n_samples_test, n_outputs])         
        observed_scores: Precomputed R2 scores from the original y_test and y_pred, if None: compute
        n_permutations: Number of permutations to perform for p-value computation   
        n_jobs: The number of jobs to run in parallel (default: -1, uses all available CPUs)
        chunk_size: The number of permutations to process in each chunk - 40 seems optimal

    Returns:
        p_values: An array of p-values for each output.
    """

    n_samples, n_outputs = y_train.shape

    if observed_scores is None:
        y_pred = model.predict(X_test)
        observed_scores = r2_score(y_test, y_pred, multioutput='raw_values')

    # Function to process a chunk of permutations
    def process_chunk(chunk_indices):
        perm_scores_chunk = []
        for perm_idx in chunk_indices:
            y_train_perm = y_train[perm_idx]
            model.fit(X_train, y_train_perm)
            y_perm_pred = model.predict(X_test)
            perm_scores_chunk.append(r2_score(y_test, y_perm_pred, multioutput='raw_values'))
        return perm_scores_chunk

    # Generate permutation indices
    perm_indices_list = [np.random.permutation(n_samples) for _ in range(n_permutations)]

    # Create chunks of permutation indices
    chunks = [perm_indices_list[i:i + chunk_size] for i in range(0, n_permutations, chunk_size)]

    # Use joblib to parallelize the chunk processing
    perm_scores_chunked = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in chunks)

    # Flatten the list of permutation scores
    perm_scores = np.vstack(perm_scores_chunked)

    # Calculate p-values
    p_values = np.mean(perm_scores >= observed_scores, axis=0)

    return p_values


def _test_optimal_chunksize(model, X_train, y_train, X_test, y_test, observed_scores=None, n_permutations=500, n_jobs=-1):
    """
    Function to test optimal chunk size to minimize processing overhead, with chunk size
    dynamically adjusted based on the total size of y_train.
    """
    
    # Determine the total size of y_train (number of elements)
    total_size = np.prod(y_train.shape)
    
    # Define chunk sizes based on the total size, ranging from 1 to 40
    chunk_sizes = np.linspace(1, min(80, total_size // 10), 10).astype(int)

    optimal_chunk_size = None
    min_time = float('inf')

    for chunk_size in chunk_sizes:
        # Preset timings
        timings = np.zeros(5)
        
        for i in range(5):
            st = time.time()

            n_samples, n_outputs = y_train.shape

            if observed_scores is None:
                y_pred = model.predict(X_test)
                observed_scores = r2_score(y_test, y_pred, multioutput='raw_values')

            # Function to process a chunk of permutations
            def process_chunk(chunk_indices):
                perm_scores_chunk = []
                for perm_idx in chunk_indices:
                    y_train_perm = y_train[perm_idx]
                    model.fit(X_train, y_train_perm)
                    y_perm_pred = model.predict(X_test)
                    perm_scores_chunk.append(r2_score(y_test, y_perm_pred, multioutput='raw_values'))
                return perm_scores_chunk

            # Generate permutation indices
            perm_indices_list = [np.random.permutation(n_samples) for _ in range(n_permutations)]

            # Create chunks of permutation indices
            chunks = [perm_indices_list[i:i + chunk_size] for i in range(0, n_permutations, chunk_size)]

            # Use joblib to parallelize the chunk processing
            perm_scores_chunked = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in chunks)

            # Flatten the list of permutation scores
            perm_scores = np.vstack(perm_scores_chunked)

            # Calculate p-values
            p_values = np.mean(perm_scores >= observed_scores, axis=0)
            
            # Save timing
            timings[i] = time.time() - st
        
        avg_time = np.mean(timings)
        print(f'Total size: {total_size}, Chunk size: {chunk_size} took: {avg_time:.4f} seconds')

        if avg_time < min_time:
            min_time = avg_time
            optimal_chunk_size = chunk_size

    print(f'Optimal chunk size: {optimal_chunk_size} with average time: {min_time:.4f} seconds for total size: {total_size}')
    return p_values
