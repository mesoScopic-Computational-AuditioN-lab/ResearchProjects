# Auditory Experiment Presentation (Psychtoolbox)

This directory contains all MATLAB scripts used for stimulus generation and experimental presentation in the study  

All experiments were implemented in **MATLAB R2021a+** using the **Psychophysics Toolbox** extensions for stimulus presentation and hardware synchronization.

---

## Contents

| File / Folder | Description |
|----------------|-------------|
| **maintonotopy.m** | Main script for the quasi-random population receptive-field (pRF) mapping experiment (localizer). Presents 240 tones per run with octave-spaced constraints to reduce temporal correlations between frequencies. |
| **settings_tonotopy.m** | Configuration file defining frequency range, tone duration, inter-tone interval, and run parameters for `maintonotopy.m`. |
| **mainpredsound.m** | Main script for the stochastic tone-sequence experiment (session 2). Presents tone sequences drawn probabilistically from mixtures of Gaussian frequency distributions to establish dynamic regularities. |
| **settings_main.m** | Parameter definitions for the main stochastic-sequence experiment (`mainpredsound.m`), including Gaussian centers, probability schedules, and block timing. |
| **mainequalization.m** | Script for individual loudness calibration within the scanner. Participants adjust tone intensity to achieve perceived loudness matching across frequencies. |
| **leftrightequalization.m** | Script for equalizing loudness between the left and right auditory channels using Sensimetric in-ear headphones. |
| **STARTEXP.m** | High-level wrapper script to launch the experimental session. Handles initialization, parameter loading, and run management. |
| **functions/** | Contains supporting MATLAB functions for tone generation, stimulus waveform construction, and synchronization with scanner triggers and button boxes. Includes utilities for timing control (`waitforpulse`, `waitforbitsi`, `waitfornokey`, etc.) and acoustic stimulus synthesis (`createwaveform`, `generate_frequencies_main`, `logistic_func`, etc.). |

---

## Overview

The Psychtoolbox-based experiment suite includes:

1. **Loudness Calibration** — establishes individual equal-loudness curves (nine logarithmically spaced tones between 200–6000 Hz) to normalize sound intensity across frequencies.  
2. **Tonotopy (Session 1)** — quasi-random tone presentation for estimating population receptive fields in auditory cortex.  
3. **Stochastic Sequence Experiment (Session 2)** — probabilistic tone sequences based on two-Gaussian mixtures, manipulating sampling probability to create transitions in acoustic regularity.

All sounds are generated at a 48,000 Hz sampling rate and delivered via MRI-compatible **Sensimetric s15** in-ear headphones.

---

## Dependencies

- **MATLAB R2021a+**  
- **Psychtoolbox 3+** (tested on version 3.0.17)  
- MRI-compatible auditory presentation system
