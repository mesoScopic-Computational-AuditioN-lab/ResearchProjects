# Single-Trial Beta Estimation (Session 1)

This directory contains the MATLAB scripts used to estimate single-trial beta maps from the population receptive-field (pRF) localizer session.  
These single-trial beta estimates form the input for subsequent pRF modeling in [`ses1_prf_estimation/`](../ses1_prf_estimation).

---

## Purpose

For each tone presented in the quasi-random localizer (Session 1), voxel-wise beta estimates were computed using a GLM approach.  
Each tone frequency was modeled as an individual event regressor, yielding one beta weight per tone per voxel.  
This produced 240 beta maps per run, which together describe the voxel’s frequency-selective response profile.

---

## Contents

| File | Description |
|------|--------------|
| **BulkBetaEstimation.m** | Main wrapper that iterates over participants and runs to estimate single-trial beta maps for all tones presented. Produces voxel-wise beta files for each tone frequency. |
| **BetaEstimation.m** | Core function that performs single-trial GLM fitting on each run, handling design matrix construction, HRF convolution, and model fitting. |
| **fitGaussianPrfbasedonRanks_MM.m** | Utility for Gaussian fitting of the resulting beta-weight frequency profiles; used in later steps to map voxel-wise frequency tuning. |
| **saveICAMap.m** | Helper function for saving beta-derived maps in BrainVoyager-compatible format (e.g., `.vmp` or `.ica` structure). |
