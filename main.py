from cas_toolbox.cas import compute_cas_for_dir

# Number of classes on which the classifier is trained
num_classes = 8

# Compute Consensus Alignment Score (CAS) for all classes in the given directory
# Parameters:
# - root_dir: Directory containing CAM visualizations
# - strategy: Consensus strategy (options: mean, union, region_iou, ssim, latent_consensus (ours))
# - threshold: Binary Threshold for IoU strategy
# - save_vis_dir: Directory to save consensus maps
# - latent_root: Directory containing corresponding latent vectors of CAMs
# - num_classes: Total number of classes
cas = compute_cas_for_dir(
    root_dir="LayerCAM-ResNet18-OCTID",
    strategy="ssim",  # Using Structural Similarity Index (SSIM)
    threshold=0.2,
    save_vis_dir="Consensus_maps_LayerCam_ResNet18-OCTID",
    latent_root="latent_vectors_ResNet18-C8",
    num_classes=num_classes,
)

# Write CAS scores to a text file
output_file = "cas_scores.txt"
with open(output_file, "w") as f:
    f.write("CAS Scores:\n")
    for cls, score in cas.items():
        f.write(f"{cls}: {score:.4f}\n")
