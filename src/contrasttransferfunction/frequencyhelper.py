import numpy as np
from pydantic import BaseModel


class FrequencyHelper1D(BaseModel):
    size: int = int(np.sqrt(512**2 + 512**2))
    pixel_size_angstroms: float = 1.0

    @property
    def spatial_frequency_pixels(self):
        return np.linspace(0, 0.5 * np.sqrt(2.0), self.size)

    @property
    def spatial_frequency_angstroms(self):
        return self.spatial_frequency_pixels * self.pixel_size_angstroms

    @property
    def spatial_wavelength_angstroms(self):
        return 1.0 / (self.spatial_frequency_angstroms + 1e-10)
