from cas_toolbox.cas import compute_cas_for_dir

cas = compute_cas_for_dir(
    root_dir="LayerCAM-ResNet18-OCTID",
    strategy="ssim",  # Options: mean, union, region_iou, ssim, hausdorff
    threshold=0.2,
    save_vis_dir="Consensus_maps_LayerCam_ResNet18-OCTID",
    latent_root="latent_vectors_ResNet18-C8"
)

print("CAS Scores:")
for cls, score in cas.items():
    print(f"{cls}: {score:.4f}")
