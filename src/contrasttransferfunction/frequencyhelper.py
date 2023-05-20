from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, validator

from contrasttransferfunction.utils import calculate_diagonal_radius, distance_from_center_array


class FrequencyHelper(BaseModel):
    size: Union[int, tuple[int, int], tuple[int, int, int]] = calculate_diagonal_radius(512)
    pixel_size_angstroms: float = 1.0
    low_frequency_cutoff_angstroms: Optional[float] = None
    high_frequency_cutoff_angstroms: Optional[float] = None

    @validator("size")
    def must_be_square(cls, v):
        if type(v) is tuple:
            if len(v) == 2 and v[0] != v[1]:
                msg = "Size must be square"
                raise ValueError(msg)
            if len(v) == 3 and (v[0] != v[1] or v[0] != v[2]):
                msg = "Size must be cube"
                raise ValueError(msg)
        return v

    @property
    def spatial_frequency_pixels(self) -> np.ndarray:
        if type(self.size) is int:
            return np.linspace(0, 0.5 * np.sqrt(2.0), self.size)
        elif type(self.size) is tuple and len(self.size) == 2:
            return distance_from_center_array(self.size[0]) / (self.size[0] - 1)
        else:
            raise NotImplementedError

    @property
    def spatial_frequency_angstroms(self) -> np.ndarray:
        return self.spatial_frequency_pixels / self.pixel_size_angstroms

    @property
    def spatial_wavelength_angstroms(self) -> np.ndarray:
        return 1.0 / (self.spatial_frequency_angstroms + 1e-10)

    @property
    def azimuth(self) -> np.ndarray:
        if type(self.size) is int:
            msg = "Azimuth only defined for 2D/3D"
            raise ValueError(msg)
        elif type(self.size) is tuple and len(self.size) == 2:
            x_inds, y_inds = np.ogrid[: self.size[1], : self.size[0]]
            mid_x, mid_y = (np.array(self.size[::-1]) - 1) / float(2)
            # TODO: Test this convention (depends on whether Y is up or down)
            azimuth = -np.arctan2(x_inds - mid_x, y_inds - mid_y)
            return azimuth
        else:
            raise NotImplementedError
