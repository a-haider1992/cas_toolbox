import os
from glob import glob
import numpy as np
import pickle

from .utils import load_heatmap
from .visualization import visualize_and_save_heatmap

def compute_cas_for_dir(root_dir, strategy="mean", threshold=0.2, save_vis_dir=None, latent_root="latent_vectors_ResNet18-C8"):
    from skimage.metrics import structural_similarity as ssim
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from sklearn.manifold import TSNE

    cas_scores = {}

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        heatmap_paths = glob(os.path.join(class_path, '*'))
        heatmap_paths = [p for p in heatmap_paths if p.endswith(('.npy', '.png', '.jpg', '.jpeg'))]
        if len(heatmap_paths) < 2:
            print(f"Skipping {class_name} (not enough heatmaps)")
            continue

        heatmaps, shaped_maps = [], []
        for p in heatmap_paths:
            h, shape = load_heatmap(p, return_shape=True)
            heatmaps.append(h)
            shaped_maps.append(np.reshape(h, shape))

        heatmaps = np.stack(heatmaps, axis=0)

        if strategy == "mean":
            consensus = np.mean(heatmaps, axis=0)
            consensus /= np.linalg.norm(consensus) if np.linalg.norm(consensus) != 0 else 1
            sims = [np.dot(h, consensus) for h in heatmaps]
            cas_score = float(np.mean(sims))

        elif strategy == "maximum":
            consensus = np.max(heatmaps, axis=0)
            consensus /= np.linalg.norm(consensus) if np.linalg.norm(consensus) != 0 else 1
            sims = [np.dot(h, consensus) for h in heatmaps]
            cas_score = float(np.mean(sims))

        elif strategy == "region_iou":
            binary_maps = [(h > threshold).astype(np.uint8) for h in shaped_maps]
            intersection = np.logical_and.reduce(binary_maps).astype(np.uint8)
            union = np.logical_or.reduce(binary_maps).astype(np.uint8)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
            cas_score = float(iou)
            consensus = union.astype(np.float32)

        elif strategy == "ssim":
            best_score = -1
            total_score = 0
            count = 0
            best_pair = (0, 1)
            for i in range(len(shaped_maps)):
                for j in range(i + 1, len(shaped_maps)):
                    score = ssim(shaped_maps[i], shaped_maps[j], data_range=1.0)
                    total_score += score
                    count += 1
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
            cas_score = total_score / count if count > 0 else 0.0
            consensus = shaped_maps[best_pair[0]]

        elif strategy == "latent_consensus":
            num_classes = 8  # adjust as needed or make configurable

            latent_vectors = []
            file_list = []
            heatmap_dict = {}

            for p in heatmap_paths:
                filename = os.path.basename(p).rsplit('.', 1)[0]
                latent_path = os.path.join(latent_root, class_name, f"{filename}.pkl")
                if not os.path.exists(latent_path):
                    print(f"Missing latent vector: {latent_path}")
                    continue
                with open(latent_path, 'rb') as f:
                    vec = pickle.load(f)
                if not np.all(np.isfinite(vec)):
                    print(f"Invalid latent vector (NaN/Inf): {latent_path}")
                    continue
                norm = np.linalg.norm(vec)
                if norm < 1e-8:
                    print(f"Low-norm latent vector: {latent_path}")
                    continue
                latent_vectors.append(vec)
                file_list.append(filename)

            if len(latent_vectors) == 0:
                print(f"No valid latent vectors found for class {class_name}")
                continue

            latent_vectors = np.stack(latent_vectors, axis=0)

            for fname, shaped_map in zip(file_list, shaped_maps):
                heatmap_dict[fname] = shaped_map

            if len(latent_vectors) < num_classes:
                print(f"Only {len(latent_vectors)} latent vectors for class {class_name}. Using mean consensus.")
                consensus_vec = np.mean(latent_vectors, axis=0)
                consensus = np.min(shaped_maps, axis=0)
            else:
                gmm = GaussianMixture(n_components=num_classes, random_state=42, covariance_type='full')
                labels = gmm.fit_predict(latent_vectors)
                centroids = gmm.means_

                class_cluster_idx = np.bincount(labels).argmax()
                cluster_indices = np.where(labels == class_cluster_idx)[0]
                consensus_vec = centroids[class_cluster_idx]

                dists = cdist([consensus_vec], latent_vectors[cluster_indices])[0]
                closest_idx_within_cluster = cluster_indices[np.argmin(dists)]
                consensus_filename = file_list[closest_idx_within_cluster]
                consensus = heatmap_dict[consensus_filename]

                # Visualization (optional, requires save_vis_dir)
                if save_vis_dir:
                    tsne = TSNE(n_components=2, perplexity=min(30, len(latent_vectors) // 2), n_iter=2000, random_state=42)
                    latent_tsne = tsne.fit_transform(latent_vectors)
                    centroid_tsne = tsne.fit_transform(np.vstack([latent_vectors, centroids]))[-num_classes:]

                    plt.figure(figsize=(10, 8))
                    for cluster_id in range(num_classes):
                        cluster_mask = labels == cluster_id
                        plt.scatter(
                            latent_tsne[cluster_mask, 0], latent_tsne[cluster_mask, 1],
                            label=f"Cluster {cluster_id}", alpha=0.5, s=50
                        )
                    plt.scatter(
                        centroid_tsne[:, 0], centroid_tsne[:, 1],
                        marker='X', c='black', s=120, linewidths=2, edgecolors='white', label='GMM Centroids'
                    )
                    dominant_centroid_tsne = centroid_tsne[class_cluster_idx]
                    plt.scatter(
                        dominant_centroid_tsne[0], dominant_centroid_tsne[1],
                        c='red', s=150, marker='*', edgecolors='black', linewidths=1.5,
                        label='Dominant Cluster Centroid'
                    )
                    closest_tsne_coord = latent_tsne[closest_idx_within_cluster]
                    plt.scatter(
                        closest_tsne_coord[0], closest_tsne_coord[1],
                        c='blue', s=120, marker='o', edgecolors='black', linewidths=1.5,
                        label='Closest to Centroid (Consensus Map)'
                    )
                    plt.title(f"t-SNE of Latent Vectors (Class {class_name})")
                    plt.xlabel("t-SNE Dimension 1")
                    plt.ylabel("t-SNE Dimension 2")
                    plt.legend()
                    plt.tight_layout()
                    os.makedirs(save_vis_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_vis_dir, f"Latent_TSNE_{class_name}.png"))
                    plt.close()

            consensus_norm = np.linalg.norm(consensus_vec)
            if consensus_norm > 0:
                consensus_vec = consensus_vec / consensus_norm

            sims = []
            for vec in latent_vectors:
                vec_norm = np.linalg.norm(vec)
                if vec_norm < 1e-8:
                    sims.append(0.0)
                else:
                    sims.append(np.dot(vec / vec_norm, consensus_vec))

            cas_score = float(np.mean(sims))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        cas_scores[class_name] = cas_score

        if save_vis_dir and 'consensus' in locals():
            visualize_and_save_heatmap(consensus, shape, class_name, save_vis_dir, strategy)

    return cas_scores
