# Miscellaneous Tools and Documentation

This directory collects supplementary resources used 

It includes:
- BrainVoyager plugin scripts for volume, surface, and cortical-depth operations (`Tools/`)
- Procedural guides describing the laminar and surface mapping workflows
- Visualisation notebooks for figure generation and experiment schematics

---

## Contents

| File / Folder | Description |
|----------------|-------------|
| **Layer_analyses.pdf** | Detailed guide for performing cortical-layer segmentation and depth sampling in BrainVoyager, combining automated and manual refinement steps. |
| **CBA_Mapping.pdf** | Instructions for performing cortex-based alignment (CBA) and inter-subject mapping of auditory ROIs, outlining both automatic and manual procedures. |
| **mainexp visualisation.ipynb** | Notebook visualising the overall experimental design, tone sequence structure, and analysis flow. |
| **abstract visualisation.ipynb** | Notebook used to generate graphical abstract components illustrating key methodological steps. |
| **Tools/** | Folder containing Python plugins for BrainVoyager. See below for details. |

---

## BrainVoyager Python Plugins (`Tools/`)

A collection of BrainVoyager-compatible Python scripts extending the software’s functionality.  
Save the files in  
`BrainVoyager/Extensions/PythonPlugins/`  
or  
`BrainVoyager/Extensions/PythonScripts/`,  
then run them via **Python → Python Development** (or press `Ctrl+P`) in BrainVoyager.

For dependencies, check each script’s import section (commonly `bvbabel`, `nibabel`, `numpy`, etc.).

| Script | Description |
|---------|-------------|
| **Isovoxel_Nearest.py** | IsoVoxel volumes (and VOIs) using nearest-neighbour interpolation. |
| **Nifti_Tools.py** | Convert between BrainVoyager and NIfTI formats, ensuring correct spatial orientation. |
| **VOI_Tools.py** | Export BrainVoyager VOIs (names/colours) to ITK-Snap and 3D-Slicer compatible formats. |
| **VTC_BOX.py** | Read `.vtc` headers and draw NIfTI bounding boxes — useful for visualising functional data coverage. |
| **VMP_Cortical_depth.py** | Sample VMP files within predefined cortical depths, outputting one SMP per depth per hemisphere. |
| **LabelMap_to_WMGM_LHRH.py** | Convert DNN-derived labelmaps (split LH/RH) into white/grey matter files per hemisphere. |
| **Map_POI.py** | Map POIs between reference frames using `.ssm` files (supports one-to-one and one-to-many mappings). |
| **BIDS_Structure_Generator.py** | Automatically generate an empty BIDS-compliant folder hierarchy with chosen datatypes and sessions. |

---

Windows users: 
`bvbabel` isnt always playing ball when intalling within a conda environment, a solution is to manually copy paste the bvbabel directory somewhere and within the scripts (before `import bvbabel`) add `import sys` & `sys.path.append('C:/path/to/parentdirectory/ofbvbabel')`.
When using this approach, please remove the last two lines of code from the `__init__.py` file within the bvbabel folder (starting from `import pkg_resources`).
