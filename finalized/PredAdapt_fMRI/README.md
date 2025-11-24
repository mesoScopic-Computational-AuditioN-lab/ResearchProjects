# Distinct Roles of Deep and Superficial Cortical Layers in Tone Prediction, Comparison, and Adaptation in Human Auditory Cortex

This repository contains all pre- and post-processing scripts used for the analyses described in the paper  
**“Distinct Roles of Deep and Superficial Cortical Layers in Tone Prediction, Comparison, and Adaptation in Human Auditory Cortex.”**

The project combines 7 T fMRI with model-based analyses of stochastic tone sequences to dissociate the laminar organization of stimulus-driven, adaptive, and predictive responses in human auditory cortex.

---

## Repository Overview

Scripts are organized by experimental session and analysis stage:

| Folder | Description |
|--------|--------------|
| **psychtoolbox_exp/** | MATLAB scripts for auditory stimulus generation and presentation, including loudness calibration, stochastic tone sequences, and quasi-random pRF mapping. |
| **beh/** | Anonymized stimulus log data. Contains participant-specific tone presentation logs, scanner trigger timings, experimental settings, and equal-loudness calibration curves. These provide the basis for stimulus modeling and precise temporal alignment with fMRI data. |
| **preproc_fmri/** | BrainVoyager-based preprocessing pipeline: raw data import, functional and anatomical preprocessing, alignment, and VTC creation. |
| **ses1_prf_estimation/** | MATLAB routines for population receptive-field (pRF) estimation and permutation-based reliability assessment. |
| **ses1_single_trial_betas/** | Scripts for deriving single-trial beta estimates per tone in session 1. |
| **ses2_modelstims/** | Implementation of stimulus-driven, adaptation, and predictive models (including the long-trace adaptation and D-REX frameworks). |
| **ses2_regressions/** | Tone-wise regression and variance-partitioning analyses linking modeled predictors to laminar fMRI responses. |
| **misc/** | Stand-alone utilities for BrainVoyager file handling (e.g., cortical-depth mapping, VOI/POI conversion, NIfTI tools). |

---

## Preprocessing Workflow

The high-field fMRI preprocessing pipeline proceeds through:
1. General settings and participant/session selection  
2. Raw data preparation  
3. Functional preprocessing  
4. Anatomical preprocessing  
5. Functional-anatomical coregistration  
6. VTC generation for analysis  

All parameters and configurations are contained within the provided notebooks.

---

## Dependencies

- **BrainVoyager 22.2+** — required for all preprocessing and alignment steps  
- **MATLAB R2021a+** — for stimulus generation, presentation, and model scripts  
- **Psychtoolbox** — for experimental presentation within MATLAB  
- **Python 3.9+** — with standard scientific libraries (`numpy`, `scipy`, `pandas`, `nibabel`, etc.)

---

## Citation

> van Haren J., de Lange F. P., Kotz S. A., & de Martino F.  
> *Distinct Roles of Deep and Superficial Cortical Layers in Tone Prediction, Comparison, and Adaptation in Human Auditory Cortex.*  
> (Manuscript in preparation / bioRxiv preprint forthcoming)

---

## Data Availability

All datasets will be made publicly available via **[OpenNeuro.org](https://openneuro.org/datasets/ds006928)**.  
A direct link to the corresponding dataset will be provided here once the upload is complete:

---

## Notes

All analysis parameters and configurations are embedded within the notebooks to facilitate transparency and reproducibility.  
