# Regression and Variance Partitioning Analyses (Session 2)

This directory contains all scripts and notebooks used for the voxel-wise and ROI-based regression analyses of Session 2.

These analyses quantify how well each model — **stimulus drive**, **repetition suppression**, and **predictive models (D-REX and HGF)** — explains BOLD responses across cortical depth and auditory regions, using cross-validated regression and set-theoretic variance partitioning.

---

## Purpose

This stage links the modeled regressors from [`../ses2_modelstims/`](../ses2_modelstims) to the preprocessed laminar fMRI data.  
It estimates voxel-wise model fits, computes cross-validated \(R^2\) maps, and partitions variance into unique and shared contributions between model components.  
Group-level statistics and layer-wise comparisons are performed in a final ROI analysis step.

---

## Contents

| File / Notebook | Description |
|-----------------|--------------|
| **regression_main.ipynb** | Primary notebook performing voxel-wise regression analyses across all runs and models. Implements leave-one-run-out (LOO) cross-validation and outputs cross-validated \(R^2\) maps for each model. |
| **regression_adaptation_grid.ipynb** | Supplementary analysis exploring alternative adaptation parameterisations to assess model sensitivity. |
| **regression_split_pred.ipynb** | Supplementary analysis to test the separation of predictive components (priors vs. prediction errors) to isolate anticipatory and reactive effects. |
| **regression_IdealObserver.ipynb** | Supplementary control analysis using an Ideal-Observer model for comparison with the D-REX framework. |
| **regression.py** | Core regression engine: handles model loading, HRF-aligned predictor integration, and cross-validated voxel-wise fits. |
| **varpar.py** | Implements the variance partitioning framework (after de Heer et al., 2017). Computes unique and shared \(R^2\) components across model combinations. |
| **stats.py** | Performs participant-level bootstrap t-tests and false-discovery-rate (FDR) correction for ROI and layer-wise statistics. |
| **ROI_and_layer_analysis.ipynb** | Final analysis notebook combining individual-participant outputs into ROI-based and laminar statistics. Includes visualisation and bootstrapped group statistics. |
| **save_maps.ipynb** | Utility for saving voxel-wise regression and variance-partitioning results as BrainVoyager-compatible `.vmp` maps. |

---

## Workflow Overview

1. **Input**  
   - Preprocessed functional data (VTCs, ROIs, layer masks) from [`../preproc_fmri/`](../preproc_fmri)  
   - Modeled regressors from [`../ses2_modelstims/`](../ses2_modelstims)  

2. **Voxel-wise Regression**  
   - Perform leave-one-run-out cross-validation per model  
   - Compute cross-validated \(R^2\) maps and significance maps  

3. **Variance Partitioning**  
   - Quantify unique and overlapping variance contributions among stimulus drive, adaptation, and predictive models using set-theoretic decomposition  

4. **ROI and Layer Analyses**  
   - Aggregate voxel-wise fits within each auditory ROI (HG, PP, PT, aSTG, pSTG)  
   - Compute laminar (deep, middle, superficial) averages and bootstrapped statistics  
   - Apply FDR correction across multiple comparisons  

5. **Output**  
   - Cross-validated \(R^2\) maps  
   - Variance-partitioned model components  
   - ROI and layer-level statistics and figures  
