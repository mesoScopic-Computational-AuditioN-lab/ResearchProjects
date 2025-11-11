from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn3
import numpy as np 
import pandas as pd

import bvbabel

def two_way_varpart(A,B,AuB,correct_R2s=True,return_sep=True, handle_negatives='zero'):
    """2-way variance partitioning. Handles scalars (single R2s) and vectors (multiple R2s).
    
    By default, this function implements a correction to handle cross-validated R2s. 
    That is, estimates and adds the smallest bias vector (in an L2-sense) such that the 
    set theoretic equations yield no inconsistent results. See De Heer, Huth, et al. 
    
    For info, see:
    - de Heer, Huth, et al. (2017) Journal of Neuroscience, 37(27), 6539-6557.

    In:
    positional args: 
    - (A,B,AuB): floats or np.array, shape(n_resp)
        R2 values for GLMs with each feature space, and the union of them. 
    - return_sep: bool (Default: True)
        if True, returns separate variable for each partition (i.e. as tuple of scalars/vectors)  
        If False, returns values in vectorised form (i.e. vector of scalars / matrix of row-vectors)
    -correct_R2s: Bool (Default: True)
        Implement correction to avoid impossible values.
    - handle_negatives: str (default: 'zero')
        How to handle negative R² values (which may occur with cross-validation).
        Options:
            'zero'          -> Set individual negative values to zero
            'zero_partials' -> Zero out all R²s for entries with *any* negative input
            'remove'        -> Exclude entries (e.g. voxels) with any negative R² value
            'aub_remove'  -> Remove entries where the full model (AuB) has negative R²; 
                               zero out other negative values (AuB < 0 breaks partitioning logic)
            'aub_zero'    -> Zero out all R²s for entries where the full model (AuB) < 0
    
    Out:
    - 6 separate variables for each (adjusted) input R2 and each partition 
        if return_sep=True
    OR
    - 1 variable for all (adjusted) input R2s and partitions 
        (6-dimensional vector if input is scalar; (6 x n_resp) dim matrix if input is vector)
    order of output:
    (A,B,C, AuB, # input R2s (adjusted values if needed)
     A*,B*, AnB) # output partitions
    
    -------
    MH 2020, JVH 2021/2024
    -------
    """
    eqs=np.array([[0,-1,1],  # A_ =  AuB - B 
                  [-1,0,1],  # B_ =  AuB - A 
                  [1,1,-1]]) # AnB=  A+B -AuB 
    all_ABAuB=np.vstack((A,B,AuB))

    # Handling negative R² values
    if handle_negatives == 'zero':
        # Option 1: Set all individual negative values to zero
        all_ABAuB[all_ABAuB<0]=0
    elif handle_negatives == 'zero_partials':
        # Option 2: For any column (e.g. voxel) with *any* negative value, zero out all R²s
        any_neg = np.any(all_ABAuB < 0, axis=0)
        all_ABAuB[:, any_neg] = 0
    elif handle_negatives == 'remove':
        # Option 3: Completely remove any columns with *any* negative value
        any_neg = np.any(all_ABAuB < 0, axis=0)
        all_ABAuB = all_ABAuB[:, ~any_neg]
    elif handle_negatives == 'aub_remove':
        # Option 4: Remove only entries where AuBuC < 0, zero others
        aubuc_neg = all_ABAuB[2] < 0 # index 2 = AuB
        all_ABAuB[all_ABAuB < 0] = 0 # zero all valid negative R²s
        all_ABAuB = all_ABAuB[:, ~aubuc_neg]
    elif handle_negatives == 'aub_zero':
        # Option 5: Where AuBuC < 0, zoro out all R²s
        aubuc_neg = all_ABAuB[2] < 0
        all_ABAuB[:, aubuc_neg] = 0

#OLD     all_ABAuB[all_ABAuB<0]=0 # ignore negative values 
    
    # estimate biases (still iterative over responses...)
    all_biases=np.zeros_like(all_ABAuB)
    if correct_R2s:
        for vox_i, ABAuB in enumerate(all_ABAuB.T):
            all_biases[:,vox_i]=_est_bias_2wayVP(ABAuB[0],ABAuB[1],ABAuB[2])
        
    all_ABAuB_adjusted=all_ABAuB+all_biases
    all_A_B_AnB=eqs.dot(all_ABAuB_adjusted)

    if return_sep: # if return values as separate variables (i.e. as tuple of scalars/vectors)
        return(all_ABAuB_adjusted[0],all_ABAuB_adjusted[1],all_ABAuB_adjusted[2],
           all_A_B_AnB[0],all_A_B_AnB[1],all_A_B_AnB[2])
    else: # if return vectorised form 
        return(np.vstack((all_ABAuB_adjusted,all_A_B_AnB)))

def three_way_varpart(A,B,C,AuB,AuC,BuC,AuBuC,correct_R2s=False, handle_negatives='zero'):
    """3-way variance partitioning. Handles scalars (single R2s) and vectors (multiple R2s).
    
    By default, thi function implements a correction to handle cross-validated R2s. 
    That is, estimates and adds the smallest bias vector (in an L2-sense) such that the 
    set theoretic equations yield no inconsistent results. See De Heer, Huth, et al. 
    
    In:
    positional args: 
    - (A,B,C,AuB,AuC,BuC,AuBuC): floats or np.array, shape(n_resp)
        R2 values for GLMs with each feature space, each pair, and all featurespaces together.
    -correct_R2s: Bool (Default: True)
        Implement correction to avoid impossible values.
    - handle_negatives: str (default: 'zero')
        How to handle negative R² values (which may occur with cross-validation).
        Options:
            'zero'          -> Set individual negative values to zero
            'zero_partials' -> Zero out all R²s for entries with *any* negative input
            'remove'        -> Exclude entries (e.g. voxels) with any negative R² value
            'aubuc_remove'  -> Remove entries where the full model (AuBuC) has negative R²; 
                               zero out other negative values (AuBuC < 0 breaks partitioning logic)
            'aubuc_zero'    -> Zero out all R²s for entries where the full model (AuBuC) < 0
    
    Out:
    - 14 tuple of scalars/vectors for each (adjusted) input R2 and computed each partition 
         the input R2s that are used in the set-theoretic equations are returned to check correction
            order of output:
            -(A, B, C,AuB,AuC,BuC,AuBuC,    # input R2s (adjusted values if needed)
              A*,B*C*,AnB*,AnC*,BnC*,AnBnC) # output partitions (index 7 and beyond)    
    -------
    MH 2020, JVH 2021/2024
    -------
    For info, see:
    de Heer, Huth, et al. (2017) Journal of Neuroscience, 37(27), 6539-6557.
    """
    # express set theoretic equations in matrix form
                 # A   B   C  AuB AuC BuC AuBuC
    eqs=np.array([[0 , 0 , 0 , 0 , 0 ,-1 , 1 ],  # A*  = AuBuC - BuC  
                  [0 , 0 , 0 , 0 ,-1 , 0 , 1 ],  # B*  = AuBuC - AuC  
                  [0 , 0 , 0 ,-1 , 0 , 0 , 1 ],  # C*  = AuBuC - AuB
                  [0 , 0 , -1, 0 , 1 , 1 , -1],  # AnB*= AuC + BuC - C - AuBuC
                  [0 , -1, 0 , 1 , 0 , 1 , -1],  # AnC*= AuB + BuC - B - AuBuC
                  [-1, 0 , 0 , 1 , 1 , 0 , -1],  # BnC*= AuB + AuC - A - AuBuC
                  [1 , 1 , 1 ,-1 ,-1 ,-1 , 1 ]]) # AnBnC* =  AuBuC + A+B+C -AuB - AuC -BuC
    all_ABC_pluspairs=np.vstack((A,B,C,AuB,AuC,BuC,AuBuC))
    
    # Handling negative R² values
    if handle_negatives == 'zero':
        # Option 1: Set all individual negative values to zero
        all_ABC_pluspairs[all_ABC_pluspairs < 0] = 0
    elif handle_negatives == 'zero_partials':
        # Option 2: For any column (e.g. voxel) with *any* negative value, zero out all R²s
        any_neg = np.any(all_ABC_pluspairs < 0, axis=0)
        all_ABC_pluspairs[:, any_neg] = 0
    elif handle_negatives == 'remove':
        # Option 3: Completely remove any columns with *any* negative value
        any_neg = np.any(all_ABC_pluspairs < 0, axis=0)
        all_ABC_pluspairs = all_ABC_pluspairs[:, ~any_neg]
    elif handle_negatives == 'aubuc_remove':
        # Option 4: Remove only entries where AuBuC < 0, zero others
        aubuc_neg = all_ABC_pluspairs[6] < 0 # index 6 = AuBuC
        all_ABC_pluspairs[all_ABC_pluspairs < 0] = 0 # zero all valid negative R²s
        all_ABC_pluspairs = all_ABC_pluspairs[:, ~aubuc_neg]
    elif handle_negatives == 'aubuc_zero':
        # Option 5: Where AuBuC < 0, zoro out all R²s
        aubuc_neg = all_ABC_pluspairs[6] < 0
        all_ABC_pluspairs[:, aubuc_neg] = 0

    
    # estimate biases (iterative over responses but should be reasonably fast...)
    all_biases=np.zeros_like(all_ABC_pluspairs)
    if correct_R2s: # loop over responses (voxels, sensors, whatever)
        for vox_i, these_ABCpluspairs in enumerate(all_ABC_pluspairs.T):
            if _needs_correction(*these_ABCpluspairs):
                all_biases[:,vox_i]=_est_bias_3wayVP(*(v for v in these_ABCpluspairs),)
        
    all_ABC_pluspairs_adjusted=all_ABC_pluspairs+all_biases
    all_ABC_derived = np.round(eqs.dot(all_ABC_pluspairs_adjusted),14) # estimate partition sizes 

    # if return values as separate variables (i.e. as tuple of scalars/vectors)
    return(*(v for v in np.vstack((all_ABC_pluspairs_adjusted,all_ABC_derived))),)
    
def _est_bias_2wayVP(A,B,AuB):
    """estimate bias vector for A*,B* and AnB in 2 way variance partitioning.
    in:
    - A,B,AuB: floats
        R2 for A B and AuB feauterespaces
    returns:
    -b123: bias vector for A,B,AnB
    """
    def obj(x):
        return(np.linalg.norm(x)) #  l2 norm 

    def c1(x): # A_ >=0; in other words, AuB+x[2] - B+x[1] >=0
        return((AuB+x[2]) - (B+x[1]))

    def c2(x): # B_ >=0; in other words, AuB+x[2] - A+x[0] >=0
        return((AuB+x[2]) - (A+x[0]))

    def c3(x): # AnB >=0; in other words, (A+x[0]) + (B+x[1]) - (AuB+x[2])] >=0
        return((A+x[0]) + (B+x[1]) - (AuB+x[2]))
    cons=(
        {'type':'ineq','fun':c1},
        {'type':'ineq','fun':c2},
        {'type':'ineq','fun':c3},
    )
    res=minimize(obj,np.random.rand(3)*-.001,constraints=cons,bounds=((None,0),)*3,)
    if not res.success: 
        print('initial faillure... trying harder......')
        res = minimize(obj,np.random.rand(3)*-.001,constraints=cons,bounds=((None,0),)*3,
                      method='SLSQP',options={'maxiter':9999})
        if res.success: return(res.x)
        else: # if still not, check what's happening 
            print('A',A);print('B:',B);print('AuB',AuB)
            set_trace() # to be commented later on
            return(np.nan(3))
    else: 
        return(res.x)
    
def _est_bias_3wayVP(A,B,C,AuB,AuC,BuC,AuBuC, only_overlap=True):
    """estimate bias vector for A*,B*,C* and AnB,AnC,BnC,AnBnC in 3 way variance partitioning.
    in:
    - A,B,C,AuB,AuC,BuC,AuBuC: floats
        R2 for A B and AuB feauterespaces
    returns:
    -b123: bias vector for A,B,C,AuB,AuC,BuC,AuBuC
    """
    def obj(x):
        return(np.linalg.norm(x)) #  l2 norm 
    
    def c1(x): # A*  = AuBuC - BuC  
        return((AuBuC+x[6]) - (BuC+x[5]))
    def c2(x): # B* >=0; in other words:  (AuBuC+x[6]) - (AuC+x[4]) >=0
        return((AuBuC+x[6]) - (AuC+x[4]))
    def c3(x): # C*  >=0; in other words AuBuC - AuB
        return((AuBuC+x[6]) - (AuB+x[3]))
    def c4(x):# AnB*= AuC + BuC - C - AuBuC
        return((AuC+x[4]) + (BuC+x[5]) - (C+x[2])- (AuBuC+x[6]))
    def c5(x):# AnC* = AuB + BuC - B - AuBuC
        return((AuB+x[3]) + (BuC+x[5]) - (B+x[1])- (AuBuC+x[6]))
    def c6(x):# BnC*= AuB + AuC - A - AuBuC
        return((AuB+x[3]) + (AuC+x[4]) - (A+x[0])- (AuBuC+x[6]))
    def c7(x):# AnBnC =  AuBuC + A+B+C - AuB - AuC -BuC
        return((AuBuC+x[6]) + (A+x[0]) + (B+x[1]) + (C+x[2]) -
               (AuB+x[3]) - (AuC+x[4]) - (BuC+x[5]))
    
    # define constraintes: all funcs >=0
    np.random.seed(123)
    if only_overlap == True: 
        cons=tuple({'type':'ineq','fun':c} for c in [c4,c5,c6,c7])
        res=minimize(obj,np.random.rand(7)*-.0001,constraints=cons,bounds=((None,0),)*7,)
    else: 
        cons=tuple({'type':'ineq','fun':c} for c in [c1,c2,c3,c4,c5,c6,c7])   
        res=minimize(obj,np.random.rand(7)*-.0001,constraints=cons,bounds=((None,0),)*7,)
    
    if not res.success: 
        print('initial faillure... trying harder......')
        res = minimize(obj,np.random.rand(7)*-.0001,constraints=cons,bounds=((0,None),)*7,
                      method='SLSQP',options={'maxiter':9999})
        if res.success: return(res.x)
        else: # if still not, check what's happening 
            print('A',A);print('B:',B);print('AuB',AuB)
            set_trace() # to be commented later on
            return(np.nan(3))
    else: 
        return(res.x)
    
def _needs_correction(A, B, C, AuB, AuC, BuC, AuBuC, only_overlap=True):
    """
    Check if a voxel's R² inputs violate set-theoretic constraints.
    
    Parameters:
        A, B, C, AuB, AuC, BuC, AuBuC : float
            Cross-validated R² values.
        only_overlap : bool
            If True, only test the intersection (overlap) constraints.
            If False, also include A*, B*, C* unique constraints.

    Returns:
        bool: True if correction is needed, False if already valid.
    """
    # Define constraints
    violations = []
    
    if not only_overlap:
        violations.extend([
            (AuBuC - BuC) < 0,       # A* = AuBuC - BuC
            (AuBuC - AuC) < 0,       # B* = AuBuC - AuC
            (AuBuC - AuB) < 0        # C* = AuBuC - AuB
        ])
    
    violations.extend([
        (AuC + BuC - C - AuBuC) < 0,                         # A∩B*
        (AuB + BuC - B - AuBuC) < 0,                         # A∩C*
        (AuB + AuC - A - AuBuC) < 0,                         # B∩C*
        (AuBuC + A + B + C - AuB - AuC - BuC) < 0            # A∩B∩C*
    ])

    return any(violations)
    
def plot_2way_varpartven(varpartres,avgfun=np.mean,mask=None,formatter=None,
                        labels=['A \n features', 'B \n features'],
                         newfig=False,ax=None, alpha=0.5):
    """wrapper function to plot results coming from `two_way_varpart` function.
    
    in: 
    - varpartres (tuple)
        tuple of res from `three_way_varpart`:
        with order:
              (A, B, AuB,,   # input R2s (adjusted values if needed)
              A*,B*,C*,AnB*,AnC*,BnC*,AnBnC) # output partitions (index 7 and beyond)    
              7, 8, 9, 10,   11,  12, 13
    - avgfun: callable (default: np.mean)
        if you average over small number of responses, maybe use median?
    - get_res: None | int/float | callable 
        which result to get. can be either a callable (e.g. mean function) or an index (specific voxel)
        defaults to mean if None
    - labels: Sequenceof 3 strings
        labels of the three venns
    - formatter: callable | None
        function doing the label formatting. defaults to converting to % and rouning
    
    returns:
    -figure: mpl figure 
    """
    if not formatter: formatter=lambda x: "{}".format(round(x*100,4))
    get_rez=avgfun if (mask is None) else lambda x: avgfun(x[mask])            
    if newfig: plt.figure()
    varpartrez_orderd=tuple(varpartres[vp_i] for vp_i in [3, 4, 5])  # check this
    fig= venn2(subsets=tuple(get_rez(vp) for vp in varpartrez_orderd),
               set_labels=labels,subset_label_formatter=formatter,ax=ax, alpha=alpha)
    return(fig)

def plot_3way_varpartven(varpartres,avgfun=np.mean,mask=None,formatter=None,
                        labels=['A \n features', 'B \n features', 'C \n features'],
                         newfig=False,ax=None, alpha=0.5, formatround=4):
    """wrapper function to plot results coming from `three_way_varpart` function.
    
    in: 
    - varpartres (tuple)
        tuple of res from `three_way_varpart`:
        with order:
              (A, B, C,AuB,AuC,BuC,AuBuC,   # input R2s (adjusted values if needed)
              A*,B*,C*,AnB*,AnC*,BnC*,AnBnC) # output partitions (index 7 and beyond)    
              7, 8, 9, 10,   11,  12, 13
    - avgfun: callable (default: np.mean)
        if you average over small number of responses, maybe use median?
    - get_res: None | int/float | callable 
        which result to get. can be either a callable (e.g. mean function) or an index (specific voxel)
        defaults to mean if None
    - labels: Sequenceof 3 strings
        labels of the three venns
    - formatter: callable | None
        function doing the label formatting. defaults to converting to % and rouning
    
    returns:
    -figure: mpl figure 
    """
    if not formatter: formatter=lambda x: "{}".format(round(x*100,formatround))
    get_rez=avgfun if (mask is None) else lambda x: avgfun(x[mask])            
    if newfig: plt.figure()
    varpartrez_orderd=tuple(varpartres[vp_i] for vp_i in [7, 8, 10,  9,  11, 12,  13])
    fig= venn3(subsets=tuple(get_rez(vp) for vp in varpartrez_orderd),
               set_labels=labels,subset_label_formatter=formatter,ax=ax, alpha=alpha)
    return(fig)

def organise_2way_varpartres(vpres_in,lbls=['set1','set2'],ignorenegative=True):
    """From tuple of 2-way varpartres, make a dict with clear labels.
    
    
    In: 
    - vpres_in: Tuple (np.array,np.array,...)
        6 arrays; output from ``three_way_varpart``
    - lbls: List / Sequence of strings
        2 names for the fundamental feature spaces 
    -------------------
    Out:
    - varpart_dict:
        dictionary with all 6 arrays (2 sets, 1 subsets) plus the 3 pairwise intersects

    for a tuple of varpartres results and 3 lables, make a dict with transparent names.
    also include intersections."""
    
    # Use unique placeholders for formatting
    placeholders = ['{A}', '{B}']
    fmt = lambda x_str: x_str.replace('{A}', lbls[0]).replace('{B}', lbls[1])
    vpres_lbls = ('{A}', '{B}', '{A}_u_{B}', '{A}*', '{B}*', '{A}_n_{B}*')
    
    vp_dict = {vp_lbl: vpres for vpres, vp_lbl in zip(vpres_in, vpres_lbls)}
    
    if ignorenegative == True:
        for k,v in vp_dict.items(): vp_dict[k][v<0]=0. # ignore negative vals
    # string format and return. 
    return({fmt(k):v for k,v in vp_dict.items()})

def organise_3way_varpartres(vpres_in,lbls=['set1','set2','set3'],ignorenegative=True):
    """From tuple of 3-way varpartres, make a dict with clear labels.
    
    
    In: 
    - vpres_in: Tuple (np.array,np.array,...)
        14 arrays; output from ``three_way_varpart``
    - lbls: List / Sequence of strings
        3 names for the fundamental feature spaces 
    -------------------
    Out:
    - varpart_dict:
        dictionary with all 14 arrays (7 sets, 7 subsets) plus the 3 pairwise intersects

    for a tuple of varpartres results and 3 lables, make a dict with transparent names.
    also include intersections."""
    
    # Use unique placeholders for formatting
    placeholders = ['{A}', '{B}', '{C}']
    fmt = lambda x_str: x_str.replace('{A}', lbls[0]).replace('{B}', lbls[1]).replace('{C}', lbls[2])
    vpres_lbls = ('{A}', '{B}', '{C}', '{A}_u_{B}', '{A}_u_{C}', '{B}_u_{C}', '{A}_u_{B}_u_{C}',
                  '{A}*', '{B}*', '{C}*', '{A}_n_{B}*', '{A}_n_{C}*', '{B}_n_{C}*', '{A}_n_{B}_n_{C}')
    
    vp_dict = {vp_lbl: vpres for vpres, vp_lbl in zip(vpres_in, vpres_lbls)}
    vp_dict['{A}_n_{B}'] = vp_dict['{A}'] + vp_dict['{B}'] - vp_dict['{A}_u_{B}']
    vp_dict['{A}_n_{C}'] = vp_dict['{A}'] + vp_dict['{C}'] - vp_dict['{A}_u_{C}']
    vp_dict['{B}_n_{C}'] = vp_dict['{B}'] + vp_dict['{C}'] - vp_dict['{B}_u_{C}']
    
    if ignorenegative == True:
        for k,v in vp_dict.items(): vp_dict[k][v<0]=0. # ignore negative vals
    # string format and return. 
    return({fmt(k):v for k,v in vp_dict.items()})

def pimpcollors_venn3(fig, 
                idrange=['100', '010', '001', '110', '101', '011', '111'], 
                colrange=['#b3b4b5', '#f68e65', '#8ca0ca', '#d5a18d', '#a0abc1', '#c39798', '#bca7a7']):
    """change collors of all elements of a venn diagram
    input: fig: figure
           idrange: range of all ellements of 3venn diagram
           colrange: matching to idrange, set collors for all ellements
    output: adjusted figure"""
    
    for i in range(len(idrange)):
        try: fig.get_patch_by_id(idrange[i]).set_color(colrange[i])
        except AttributeError: print(idrange[i] + 'not present')
    
    return(fig)

def pimpcollors_venn2(fig, 
                idrange=['10', '01', '11'], 
                colrange=['#EE6D4A', '#475DEC', '#9B659B']):
    """change collors of all elements of a venn diagram
    input: fig: figure
           idrange: range of all ellements of 3venn diagram
           colrange: matching to idrange, set collors for all ellements
    output: adjusted figure"""
    
    for i in range(len(idrange)):
        try: fig.get_patch_by_id(idrange[i]).set_color(colrange[i])
        except AttributeError: print(idrange[i] + 'not present')
    
    return(fig)

def organize_varpar_to_df(varpar, lbls=['Baseline','Adaptation','Expectation'], nvars=3):
    """organize variance partitioning tuple of arrays into a dataframe with namings
    input varpar: tuple of arrays - obtained from 2- or 3-way varpar
          lbls: list - labels to combine
          nvars: int - 2 or 3 way varpar
    return pandas dataframe with organized labeling"""
    
    # generate dict from varpars using labels as keys
    if nvars == 3: varpardict = organise_3way_varpartres(varpar, lbls=lbls, ignorenegative=False)
    elif nvars == 2: varpardict = organise_2way_varpartres(varpar, lbls=lbls, ignorenegative=False)

    # transform into pandas dataframe
    return pd.DataFrame.from_dict(varpardict)

def mask_varpart_impossible(df, lbls=['Baseline', 'Adaptation', 'Expectation'], u='_u_', filter_flags=[1, 1, 1, 1, 1, 1, 1]):
    """
    Returns a mask (1 = valid, 0 = invalid) where any of the variance partitioning columns are zero 
    (where regressions didnt properly happen).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with main effects and interaction terms.
    lbls : list of str
        Names of the three main effects (default: ['Baseline', 'Adaptation', 'Expectation']).
    u : str
        Delimiter used in column names (default: '_u_').
    filter_flags : list or np.ndarray of 7 binary values
        Which of the 7 R² columns to consider for filtering.
        Order: [A, B, C, A∪B, A∪C, B∪C, A∪B∪C]
        Example: [0, 0, 0, 0, 0, 0, 1] -> only filter if A∪B∪C is zero.
        Default: np.ones(7) -> filter if *any* of the 7 are zero.

    Returns
    -------
    numpy.ndarray
        1D array with 1 for rows where all relevant columns are non-zero, else 0.
    """

    # warn if filter_flags are off
    filter_flags = np.array(filter_flags, dtype=int)
    if filter_flags.shape != (7,):
        raise ValueError("filter_flags must be a list or array of length 7.")
    
    # Set placeholders
    A, B, C = lbls
    # Use unique placeholders for formatting
    set_cols = [f'{A}', f'{B}', f'{C}', f'{A}{u}{B}', f'{A}{u}{C}', f'{B}{u}{C}', f'{A}{u}{B}{u}{C}']
    # Apply filter mask to only the selected columns
    cols_to_check = [col for col, use_col in zip(set_cols, filter_flags) if use_col]
    # Create boolean mask: 1 = valid, 0 = any selected col is zero
    bad_msk = 1 - (df[cols_to_check] == 0).any(axis=1).astype(int).to_numpy()

    return bad_msk

def save_to_vmp(df, vox_idx, dummy_vmp, dummy_vmp_head, vmp_path):
    """ save df obtained from organize varpar to df, as a vmp file,
        each column in the dataframe is saved as a seperate map
    
        df: pandas - dataframe with maps to save as vmp
        vox_idx: tuple of arrays - (3xindex arrays) containing linear indexes of voxel locations 
        dummy_vmp: np.array - numpy array of a vmp file of correct shape (last column doesnt have to match) [bvbabel]
        dummy_vmp_head: dict - dummy vmp dictionary [bvbabel]
        vmp_path: string - full path
    """
    
    # count number of models within dataframe
    nr_models = len(df.columns)

    # load dummy vmp image
    score_vmp_full = np.zeros((list(dummy_vmp.shape[:-1]) + [nr_models]))
    dummy_vmp_head['NrOfSubMaps'] = nr_models
    dummy_vmp_head['Map'] = []

    # loop over columns / maps
    for col in df.columns:

        # get column index
        colidx = df.columns.get_loc(col)

        # current model
        score_vmp = np.zeros(dummy_vmp.shape[:3])
        score_vmp[vox_idx] = df[col]

        # append to full dataframe
        score_vmp_full[:,:,:,colidx] = score_vmp
        dummy_vmp_head['Map'].append(set_map(col, mapthreshold=0, mapupperthreshold=df[col].max()))

    # save the vmp file
    bvbabel.vmp.write_vmp(vmp_path,dummy_vmp_head, score_vmp_full)

def closest_row(lookup, coord):
    """quick function so search for the closest coordinate in the lookup table
    then convert this to the -10 +10 range of brainvoyager"""
    lookup = np.array(lookup)
    coords = np.array(coord)
    bv_value = np.concatenate((np.arange(1,11,1), np.arange(-1,-11,-1)))
    distances = np.linalg.norm(lookup[np.newaxis, :, :] - coords[:, np.newaxis, :], axis=2)
    return bv_value[np.argmin(distances, axis=1)]

def set_map(mapname, mapthreshold=1.65, mapupperthreshold=8.0):
    FDRTableInfo = np.array([], dtype=np.float64)
    FDRTableInfo.shape = (0,3)
    returnmap = {'TypeOfMap': 1,
     'MapThreshold': mapthreshold,
     'UpperThreshold': mapupperthreshold,
     'MapName': mapname,
     'RGB positive min': np.array([255,   0,   0], dtype=np.uint8),
     'RGB positive max': np.array([255, 255,   0], dtype=np.uint8),
     'RGB negative min': np.array([255,   0, 255], dtype=np.uint8),
     'RGB negative max': np.array([  0,   0, 255], dtype=np.uint8),
     'UseVMPColor': 0,
     'LUTFileName': '<default>',
     'TransparentColorFactor': 1.0,
     'ClusterSizeThreshold': 50,
     'EnableClusterSizeThreshold': 0,
     'ShowValuesAboveUpperThreshold': 1,
     'DF1': 249,
     'DF2': 1,
     'ShowPosNegValues': 3,
     'NrOfUsedVoxels': 45555,
     'SizeOfFDRTable': 0,
     'FDRTableInfo': FDRTableInfo,
     'UseFDRTableIndex': 0}
    return(returnmap)
