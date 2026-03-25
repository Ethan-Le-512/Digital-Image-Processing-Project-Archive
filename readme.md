# MRI/PET Medical Image Fusion

A Python implementation of a multi-algorithm medical image fusion pipeline that combines structural detail from MRI scans with metabolic activity information from PET scans — producing a single fused image that preserves both anatomical and functional data for enhanced clinical interpretation.

## Overview

MRI and PET imaging capture complementary information: MRI provides high spatial resolution of anatomical structures, while PET reveals metabolic and functional activity. Fusing these modalities into a single image can support more informed medical diagnosis without requiring manual comparison between two separate scans.

This project implements a three-stage fusion pipeline built entirely from scratch in Python, combining classical signal processing techniques with a bio-inspired optimization algorithm.

## Pipeline

The fusion is performed in three stages:

**Stage 1 — Frequency Decomposition (Weighted Mean Curvature Filter)**  
Each input image is decomposed into low-frequency (coarse structure) and high-frequency (fine detail/edges) components using a Weighted Mean Curvature Filter (WMCF). This preserves structural integrity better than standard Gaussian decomposition.

**Stage 2 — High-Frequency Fusion (XDoG + Structure Tensor + Local Energy)**  
High-frequency components are fused using an eXtended Difference of Gaussians (XDoG) filter combined with Structure Tensor analysis. A Local Energy Decision Map determines, pixel-by-pixel, which source image contributes stronger edge information at each location, ensuring the fused result retains the sharpest details from both modalities.

**Stage 3 — Low-Frequency Fusion (Coyote Optimization Algorithm)**  
A bio-inspired metaheuristic — the Coyote Optimization Algorithm (COA) — adaptively optimises the weighting coefficients for combining the low-frequency components. The fitness function maximises image variance (information content), ensuring the fused base layer is not a naive average.

The final fused image is reconstructed in YUV colour space, preserving PET chrominance (metabolic colour mapping) while replacing the luminance channel with the fused output.

## Algorithms Implemented

| Algorithm | Purpose |
|---|---|
| Weighted Mean Curvature Filter (WMCF) | Image decomposition into frequency bands |
| eXtended Difference of Gaussians (XDoG) | Edge-preserving high-frequency enhancement |
| Structure Tensor + Local Energy | Pixel-wise edge saliency decision mapping |
| Coyote Optimization Algorithm (COA) | Adaptive low-frequency fusion parameter search |

## Tech Stack

- Python 3
- OpenCV (`cv2`) — image I/O and colour space conversion
- NumPy — numerical computation
- SciPy (`ndimage`) — convolution and filtering
- Recommended IDE: Spyder

## Usage

**To produce the final fused image:**
```bash
# Open final.py or Finalbonus.py
# Set the MRI and PET input file paths
# Run the script — output saved as Fused_Image.jpg
python final.py
```

**To visualise each intermediate step:**
```bash
# Open decisionmapfinal.py or decisionmapbonus.py
# Set the input file path
# The script will plot each stage of the pipeline
python decisionmapfinal.py
```

## Repository Structure

```
├── final.py               # Main fusion pipeline (Algorithm 1+2+3)
├── Finalbonus.py          # Extended/bonus variant
├── decisionmapfinal.py    # Step-by-step visualisation
├── decisionmapbonus.py    # Step-by-step visualisation (bonus variant)
├── Dataset/               # Sample MRI and PET input images
├── Final_result/          # Output images from main pipeline
├── Improved_result/       # Output images from bonus pipeline
├── Tutorial.pdf           # Project report and methodology
└── Source.txt             # References and citations
```

## Background

This project was completed as part of a Digital Image Processing course. The fusion methodology draws on published literature in multi-modal medical image fusion, combining frequency-domain decomposition with activity-level measurement and metaheuristic optimisation — areas actively researched in medical imaging.
