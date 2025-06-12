import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects

def visualize_and_save_heatmap(consensus, shape, class_name, output_dir, strategy, gamma=0.5, threshold_percentile=10, top_k=5):
    os.makedirs(output_dir, exist_ok=True)
    heatmap_img = consensus.reshape(shape)

    # Normalize and apply gamma correction
    heatmap_img = (heatmap_img - np.min(heatmap_img)) / (np.max(heatmap_img) - np.min(heatmap_img))
    heatmap_img = np.power(heatmap_img, gamma)

    # Thresholding
    threshold = np.percentile(heatmap_img, threshold_percentile)
    binary_mask = heatmap_img >= threshold
    labeled, num_features = label(binary_mask)
    slices = find_objects(labeled)

    region_sizes = [(i+1, np.sum(labeled == i+1)) for i in range(num_features)]
    top_regions = sorted(region_sizes, key=lambda x: x[1], reverse=True)[:top_k]

    top_k_mask = np.zeros_like(heatmap_img)
    for region_id, _ in top_regions:
        top_k_mask[labeled == region_id] = heatmap_img[labeled == region_id]

    # Plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=600)
    im = ax.imshow(top_k_mask, cmap='jet')
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    plt.title(f'Consensus - {class_name} ({strategy})\nTop {top_k} Regions > {threshold_percentile}th %ile')

    save_path = os.path.join(output_dir, f'{class_name}_{strategy}_top{top_k}_thr{threshold_percentile}.png')
    plt.savefig(save_path)
    plt.close()
