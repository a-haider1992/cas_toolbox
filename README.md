# 💡 CAS Toolbox

The **Consensus Activation Score (CAS)** Toolbox is a utility for computing agreement between activation heatmaps of individual instances per class. This GitHub repository is the official implementation of **ConsensusXAI: A framework to examine class-wise agreement in medical imaging** paper accepted at **IEEE/CVF Winter Conference on Applications of Computer Vision 2026**.

**Abstract**
Explainable AI (XAI) is essential for trust and transparency in deep learning, especially in medical imaging. Existing local explanation methods provide per-instance insights but fail to show whether similar explanations hold across samples of the same class. This limits global interpretability and demands time-consuming manual review by clinicians to trust models in practice. We introduce the **Consensus Alignment Score (CAS)**, a novel metric that quantifies consistency of explanations at the class level. We also present ConsensusXAI, an open-source, modeland method-agnostic framework that evaluates explanation agreement **quantitatively (via CAS)** and **qualitatively (through consensus heatmaps) per class**. Unlike prior benchmarks, ConsensusXAI uses a latent-space clustering approach, Latent Consensus, to identify dominant explanation patterns, exposing biases and inconsistencies towards certain classes. Evaluated across four benchmark datasets and two imaging modalities, our method consistently reveals meaningful class-level insights, outperforming traditional metrics like SSIM and IoU, and enabling faster, more confident clinical adoption of AI models.



## ✨ Features

- 🔍 **Multiple CAS scoring strategies:**
  - 🔥 **Heatmap-based**
    - `region_iou` — Intersection-over-Union (IoU) of thresholded regions
    - `ssim` — Structural Similarity Index (SSIM) between heatmaps
  - 📊 **Vector-based**
    - `pixel-summary-based` — Uses statistical operations (mean, max) on pixel values
    - `latent_consensus` — *Proposed* method to quantify & visualize **class-wise agreement** using latent vectors

- 📂 Supports `.npy`, `.png`, `.jpg`, `.jpeg` heatmap formats  
- 💾 Option to save:
  - Consensus visualizations
  - t-SNE plots of latent representations
- ⚠️ Graceful handling of missing or invalid latent vectors  
- 🗂️ Designed for class-wise structured directories

---

## ⚙️ Installation

Install the required dependencies with:

```bash
pip install numpy pillow scikit-image scikit-learn matplotlib

## Usage

from cas_toolbox.cas import compute_cas_for_dir

cas = compute_cas_for_dir(
    root_dir="LayerCAM-ResNet18-OCTID",                # Directory of heatmaps, organized by class
    strategy="ssim",                                   # Options: mean, union, region_iou, ssim, latent_consensus
    threshold=0.2,                                     # For region_iou strategy
    save_vis_dir="Consensus_maps_LayerCam_ResNet18-OCTID",  
    latent_root="latent_vectors_ResNet18-C8"           # Required only for 'latent_consensus'
)

## Output
print("CAS Scores:")
for cls, score in cas.items():
    print(f"{cls}: {score:.4f}")

## Citation
@misc{cas_toolbox,
  author       = {Abbas Haider et al.},
  title        = {ConsensusXAI: A framework to examine class-wise agreement in medical imaging},
  Inproceedings = {Winter Conference on Applications of Computer Vision 2026},
  year         = {2026},
  note         = {\url{https://github.com/a-haider1992/cas_toolbox}}
}

