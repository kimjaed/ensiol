import numpy as np


def layered_stairs(z_interfaces, values):
    """(values, depths) of a per-layer profile drawn as a staircase against depth."""
    z_interfaces = np.asarray(z_interfaces, float)
    values = np.asarray(values, float)
    depth_stair = np.empty(2 * values.size)
    depth_stair[0::2] = z_interfaces[:-1]
    depth_stair[1::2] = z_interfaces[1:]
    return np.repeat(values, 2), depth_stair


def pca_2d(X):
    """(mean, components (2, n_features)); project with (X - mean) @ components.T"""
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt[:2]


def pad_lims(arrs):
    """((xmin, xmax), (ymin, ymax)) around the stacked points, padded by 8%."""
    allpts = np.vstack(arrs)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    pad = (hi - lo) * 0.08
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])
