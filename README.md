# CAS Toolbox

**Consensus Activation Score (CAS)** Toolbox is a utility for computing agreement metrics between activation heatmaps. It is especially useful in explainable AI (XAI) pipelines to quantify consistency across visual explanations produced by neural network models.

## Features

- Multiple CAS scoring strategies:
  - **Heatmap-based**
    - **region_iou**: Intersection-over-Union (IoU) across thresholded regions
    - **ssim**: Pairwise Structural Similarity Index (SSIM) 
  - **Vector-based**  
    - **Pixel-summary-based**: Consensus map is generated via pixelwise statistical operations e.g. mean, maximum and CAS metric is computed  
    - **latent_consensus**: Our proposed strategy to quantify and visualise classwise agreement between explanations.
- Supports `.npy`, `.png`, `.jpg`, and `.jpeg` heatmap formats
- Optionally saves consensus visualizations and latent t-SNE plots
- Handles invalid or missing latent vectors gracefully
- Designed to work with class-wise structured input directories


## Installation

Install the required dependencies with:

```bash
pip install numpy pillow scikit-image scikit-learn matplotlib

## Usage
from cas_toolbox.cas import compute_cas_for_dir

cas = compute_cas_for_dir(
    root_dir="LayerCAM-ResNet18-OCTID",             # Path to heatmap data organized by class
    strategy="ssim",                                # Options: mean, union, region_iou, ssim, latent_consensus
    threshold=0.2,                                   # Threshold for region_iou strategy
    save_vis_dir="Consensus_maps_LayerCam_ResNet18-OCTID",  # Output directory for visualizations
    latent_root="latent_vectors_ResNet18-C8"        # Required for 'latent_consensus' strategy
)

print("CAS Scores:")
for cls, score in cas.items():
    print(f"{cls}: {score:.4f}")

## Output
CAS Scores per class:
ClassA: 0.8421
ClassB: 0.7935
ClassC: 0.9112

## 📖 Citation
If you use this toolbox in your research or publications, please cite it as:

@misc{cas_toolbox,
author = {Abbas Haider et. al.},
title = {CAS Toolbox: A Tool for Computing Consensus Activation Scores from Heatmaps},
year = {2025},
published = {\url{https://github.com/your-username/cas_toolbox}},
}
