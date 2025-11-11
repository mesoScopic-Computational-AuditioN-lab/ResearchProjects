# Stimulus Modeling (Session 2)

This directory contains all scripts and notebooks used to construct tone-by-tone model regressors for the second fMRI session.

These models translate the stochastic tone sequences into voxel-wise predictors representing **stimulus drive**, **long-trace adaptation**, and **predictive mechanisms** derived from the **D-REX** and **HGF** frameworks.

---

## Purpose

This stage implements the computational modeling of auditory sequences, generating fully time-aligned predictor time courses for the regression analyses in [`../ses2_regressions/`](../ses2_regressions).  
All predictors are convolved with a canonical HRF and resampled to the fMRI TR *within* the modeling notebook, yielding model outputs already expressed in the TR domain.

---

## Contents

| File / Folder | Description |
|----------------|-------------|
| **stim_modelling.ipynb** | Main notebook implementing the complete stimulus modeling framework. Generates TR-aligned regressors for stimulus drive, repetition suppression, and predictive (D-REX and HGF) variables including prior probability, surprisal, precision, and prediction error. |
| **Adaptation/** | Implements the double-exponential long-trace adaptation model, combining short- and long-timescale suppression components (`a_fast`, `τ_fast`, `a_slow`, `τ_slow`). |
| **DREX/** | Contains the Dynamic Regularity Extraction (D-REX) model, a Bayesian sequential inference framework producing tone-wise estimates of prior, surprisal, precision, and contextual updating. |
| **HGF/** | Supplementary implementation of the Hierarchical Gaussian Filter model for comparison with the D-REX framework. Provides an alternative probabilistic account of auditory prediction and volatility tracking. |
| **Visualisations/** | Scripts for visualising modeled trajectories (e.g., probability landscapes, adaptation traces, or surprisal dynamics). |
