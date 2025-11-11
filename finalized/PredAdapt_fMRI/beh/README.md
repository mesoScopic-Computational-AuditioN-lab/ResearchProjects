# Behavioural and Stimulus Log Data

This directory contains all behavioural and stimulus log files

It provides the full record of tone-sequence presentations, scanner synchronization, and participant-specific calibration data used in model construction and preprocessing.

---

## Directory Overview

| Folder / File | Description |
|----------------|-------------|
| **/data/** | Contains participant- and run-specific `.mat` files with raw behavioural and stimulus logs from both experimental sessions. |
| **/loudness/** | Contains individual equal-loudness calibration curves, obtained before scanning to perceptually match tone intensity across frequencies for each participant. |

---

## `/data/` Contents

Each participant’s files are prefixed with their numeric ID (e.g., `1-mainpred.mat`, `1-r1-tonotopy.mat`, etc.).

| File type | Description |
|------------|-------------|
| **\<ID\>-mainpred.mat** | Raw behavioural log from the Session 2 stochastic-sequence experiment. Includes full tone presentation timing, sequence parameters, and scanner triggers. |
| **\<ID\>-r\*-tonotopy.mat** | Raw tone log files from the Session 1 pRF mapping runs, containing frequency and onset information for each tone. |
| **\_\<ID\>-r\*-pulses.mat** | Extracted scanner pulse logs used for synchronization with MRI acquisition. |
| **\<ID\>_settings.mat** / **\<ID\>_settings_tonotopy.mat** | Experiment parameter files defining session structure, tone frequencies, and sequence probability settings. |
| **\<ID\>_stimdf.mat** | Preprocessed stimulus dataframe containing structured, trial-by-trial timing, frequency, and probabilistic context. This serves as the modeling input for [`../ses2_modelstims/`](../ses2_modelstims/). |

---

## `/loudness/` Contents

Contains individual calibration data obtained prior to scanning.  
Each file represents the participant’s equal-loudness contour, derived from in-scanner tone matching.  
These curves were used to adjust tone intensities such that perceived loudness remained constant across the frequency range.

| File type | Description |
|------------|-------------|
| **\<ID\>_loudness.mat** | Participant-specific loudness calibration curve, interpolated across test frequencies. Used to normalise tone presentation intensity. |

---
