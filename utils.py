import numpy as np

HAND_LANDMARKS = 21
FEAT_DIM = HAND_LANDMARKS * 3

def normalize_landmarks(flat_xyz):
    pts = np.array(flat_xyz, dtype=np.float32).reshape(-1, 3)
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts, axis=1).max()
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.reshape(-1)

def row_to_features(row):
    flat = list(map(float, row[1:1+FEAT_DIM]))
    return normalize_landmarks(flat)
