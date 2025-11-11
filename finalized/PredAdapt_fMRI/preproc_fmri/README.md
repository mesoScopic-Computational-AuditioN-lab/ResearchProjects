# Preprocessing of High-Field fMRI Data

This directory contains all scripts and BrainVoyager notebooks used for preprocessing the 7 T fMRI data in the study  
**“Distinct Roles of Deep and Superficial Cortical Layers in Tone Prediction, Comparison, and Adaptation in Human Auditory Cortex.”**

All preprocessing was performed in **BrainVoyager 22.2+**, using its integrated Python environment (BV Notebooks), with external dependencies on **FSL**, **ANTs**, and **Nighres** for geometric correction, alignment, and anatomical preparation.

---

## Contents

| File / Folder | Description |
|----------------|-------------|
| **bv_preproc_pipeline.bvnb** | Full preprocessing notebook implementing the complete workflow from DICOM import to fully preprocessed data. |
| **bv_preproc_vtcCreation.bvnb** | Notebook for generating VTC (voxel-time-course) files in anatomical reference space. |
| **preproc_demo.bvnb** | Minimal example pipeline illustrating the preprocessing stages on a reduced dataset. |
| **bv_preproc/** | Directory containing all supporting Python functions for automatic preprocessing, alignment, and data handling within BrainVoyager. |

---

## Preprocessing Workflow

The pipeline follows the same structure described in the manuscript and can be executed step-by-step or as a full workflow within BrainVoyager notebooks.

### 1. General Configuration
- Define participants and sessions.  
- Specify data paths, preprocessing options, and output directories.

### 2. Raw Data Preparation
- Import and rename DICOMs, organize by participant and session.  
- Extract acquisition parameters from headers and convert to BrainVoyager-compatible formats.  

### 3. Functional Preprocessing
Performed in **BrainVoyager-Notebooks** with integrated calls to **FSL** and **ANTs**:
- **Slice-timing correction** using sinc interpolation.  
- **3D motion correction** with rigid-body alignment (6 DOF) within and across runs.  
- **Temporal high-pass filtering** (cut-off = 7 cycles / run).  
- **Distortion correction** using **FSL Topup** with reversed phase-encoded field maps.  
- **Non-linear distortion correction** using **ANTs** to improve inter-session alignment.  

### 4. Anatomical Preprocessing
- Import high-resolution **MP2RAGE** T1-weighted data (0.7 mm isotropic).  
- **Skull and dura removal** using **Nighres**.  
- Intensity normalization and bias-field correction.  
- Segmentation of gray–white and pial boundaries using **BrainVoyager’s Tiramisu** deep-learning segmentation.  
- (Visual inspection and manual refinement in **3D Slicer**.)

### 5. Functional–Anatomical Coregistration
- Boundary-based registration (**BBR**) between mean functional volume and upsampled anatomical reference.  

### 6. VTC Generation
- Creation of **VTC (voxel-time-course)** files in anatomical reference space.  
- These serve as the basis for voxel-wise and laminar analyses in subsequent stages.

---

## Dependencies

All components below are required for a complete and reproducible run of the pipeline:

- **BrainVoyager 22.2+** — primary preprocessing and analysis environment  
- **FSL 6+** — geometric distortion correction (*Topup*)  
- **ANTs 2+** — non-linear registration and inter-session alignment  
- **Nighres 1+** — skull and dura removal from MP2RAGE anatomical images  
- **Python 3.9+** (within BV Notebooks) with: `numpy`, `scipy`, `pandas`, `bvbabel`, `nibabel`

---

## Notes

All parameters (paths, subject lists, and preprocessing options) are defined directly within the notebooks.  
Each stage can be executed independently or as part of the full workflow.  

The outputs from this pipeline form the foundation for:  
- `ses1_single_trial_betas/` — single-trial beta modeling  
- `ses1_prf_estimation/` — population receptive-field estimation  
- `ses2_modelstims/` — model-based stimulus and laminar regression analyses  
