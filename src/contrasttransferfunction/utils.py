import numpy as np


def calculate_diagonal_radius(box_size: int = 512) -> int:
    return int(np.around(np.sqrt(2 * (box_size / 2 - 0.5) ** 2)))


def distance_from_center_array(size: int = 512) -> np.ndarray:
    x, y = np.ogrid[0:size, 0:size]
    r = np.hypot(x - (size - 1) / 2, y - (size - 1) / 2)
    return r
