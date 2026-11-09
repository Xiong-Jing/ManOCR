"""Direction-dependent inward edge offsets for convex word polygons, Eq. (5)."""
import numpy as np


def edge_asymmetric_shrink(points, shrink_ratio=0.4, gamma=1 / 3):
    if not 0 < shrink_ratio < 1 or not 0 < gamma <= 1:
        raise ValueError("Expected shrink_ratio in (0,1), gamma in (0,1]")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        return None
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-8):
        return None
    signed_area = 0.5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
    if abs(signed_area) <= 1:
        return None
    orientation = 1 if signed_area > 0 else -1
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.any(orientation * crosses < -1e-8):
        raise ValueError("Edge-offset AS expects convex word polygons (the paper uses rectangular boxes)")
    dx = abs(signed_area) * (1 - shrink_ratio ** 2) / lengths.sum()
    result = points.copy()
    for origin, edge, length in zip(points, edges, lengths):
        inward = orientation * np.array([-edge[1], edge[0]]) / length
        distance = dx if abs(edge[1]) > abs(edge[0]) else gamma * dx
        boundary = np.dot(inward, origin) + distance
        clipped = []
        for a, b in zip(result, np.roll(result, -1, axis=0)):
            da, db = np.dot(inward, a) - boundary, np.dot(inward, b) - boundary
            if da >= -1e-9:
                clipped.append(a)
            if (da >= 0) != (db >= 0):
                clipped.append(a + (b - a) * da / (da - db))
        if len(clipped) < 3:
            return None
        result = np.asarray(clipped)
    return result.astype(np.float32)
