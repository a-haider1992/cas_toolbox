import numpy as np
from PIL import Image

def load_heatmap(path, return_shape=False):
    """Loads and normalizes a heatmap."""
    if path.endswith('.npy'):
        heatmap = np.load(path)
    elif path.endswith(('.png', '.jpg', '.jpeg')):
        heatmap = Image.open(path).convert('L')
        heatmap = np.array(heatmap, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    shape = heatmap.shape
    flat = heatmap.flatten()
    norm = np.linalg.norm(flat)
    normed = flat / norm if norm != 0 else flat
    return (normed, shape) if return_shape else normed
