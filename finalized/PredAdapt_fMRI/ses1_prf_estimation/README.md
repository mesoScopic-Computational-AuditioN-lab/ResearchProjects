# Population Receptive-Field (pRF) Estimation (Session 1)

This directory contains all MATLAB scripts and supporting files used for voxel-wise population receptive-field (pRF) estimation from the quasi-random tonotopy localizer in **Session 1**.  

The estimated frequency-selective tuning parameters (preferred frequency and tuning width) are subsequently used to parameterize the voxel-wise models of auditory responses in Session 2.

---

## Purpose

This step converts the single-trial beta maps (from [`ses1_single_trial_betas/`](../ses1_single_trial_betas)) into frequency-selective tuning models per voxel.  
Each voxel’s frequency response profile is fit with a Gaussian function on a log₂-frequency axis, yielding estimates of tuning center (`μ`) and width (`σ`).  
A permutation-based fitting procedure corrects for biases in low-SNR regions and improves model reliability.

---

## Contents

| File | Description |
|------|--------------|
| **explore_prf_with_permutations_noCV_bulk_f.m** | Main wrapper script performing permutation-based pRF fitting across participants and runs using single-trial beta inputs. |
| **fitGaussianPrfbasedonRanks_MM.m** | Core fitting routine that models voxel-wise frequency tuning as a Gaussian function on a log₂ frequency scale. |
| **get_gausssian_weigthsZ_MM_f.m** | Utility for computing Gaussian weight profiles given parameterized tuning estimates. |
| **get_penaltybasedonRanks_MM.m** | Computes permutation-based penalty terms to correct for overfitting in low-SNR voxels (after Lage-Castellanos et al., 2020). |
| **tonotopy - create prt.ipynb** | Notebook for generating BrainVoyager protocol (`.prt`) files from experimental log data for each participant and run. |
| **saveICAMap.m** | Helper function for saving fitted parameter maps (preferred frequency, tuning width, fit quality) to BrainVoyager-compatible `.vmp` files. |

---

## Required Additional Files and Dependencies

The following directories and files are required for full execution but are not included in this repository due to size:

- `/fMRIEncoding/` — toolbox for voxel-wise encoding model fitting and grid search routines.  
- `/NeuroElf_v09b/` or `/NeuroElf_v10_5153/` — MATLAB-based neuroimaging library used for file I/O, HRF convolution, and statistical utilities.  
- `/ff_range.mat` — lookup table containing the log₂-spaced frequency grid (240 tones between 200–6000 Hz) used for pRF estimation.

Ensure these resources are available and added to the MATLAB path before running the scripts.
