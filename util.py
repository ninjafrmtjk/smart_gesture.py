"""Geometry utilities for hand landmark processing."""

import numpy as np


def get_angle(a, b, c) -> float:
    """
    Angle (degrees) at point *b* formed by the vectors b→a and b→c.

    Args:
        a, b, c: (x, y) coordinate tuples.
    """
    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    return float(np.abs(np.degrees(radians)))


def get_distance(landmark_list) -> float | None:
    """
    Scaled distance between two landmark coordinates.

    Args:
        landmark_list: list of two (x, y) tuples in normalised [0, 1] space.

    Returns:
        Distance mapped to [0, 1000], or None if input is too short.
    """
    if len(landmark_list) < 2:
        return None
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    raw = np.hypot(x2 - x1, y2 - y1)
    return float(np.interp(raw, [0, 1], [0, 1000]))
