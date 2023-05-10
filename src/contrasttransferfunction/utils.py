import numpy as np


def calculate_diagonal_radius(box_size: int = 512) -> int:
    return int(np.sqrt(2 * (box_size / 2) ** 2))
