from typing import Union, overload

import numpy as np
from pydantic import BaseModel

from contrasttransferfunction.frequencyhelper import FrequencyHelper
from contrasttransferfunction.utils import calculate_diagonal_radius

DEFAULT_BOXSIZE = 512


class ContrastTransferFunction(BaseModel):
    # These are the parameters that are used to generate the CTF
    defocus1_angstroms: Union[float, np.ndarray] = 10000.0
    defocus2_angstroms: Union[float, np.ndarray] = 10000.0
    defocus_angle_degrees: float = 0.0
    additional_phase_shift_degrees: float = 0.0
    voltage_kv: float = 300.0
    spherical_aberration_mm: float = 2.7
    amplitude_contrast: float = 0.07
    pixel_size_angstroms: float = 1.0

    # These are the parameters used for plotting and fitting
    box_size: int = DEFAULT_BOXSIZE

    class Config:
        arbitrary_types_allowed = True

    @property
    def defocus1_pixels(self):
        return self.defocus1_angstroms / self.pixel_size_angstroms

    @property
    def defocus2_pixels(self):
        return self.defocus2_angstroms / self.pixel_size_angstroms

    @property
    def defocus_angle_radians(self):
        return np.deg2rad(self.defocus_angle_degrees)

    @property
    def additional_phase_shift_radians(self):
        return np.deg2rad(self.additional_phase_shift_degrees)

    @property
    def wavelength_angstroms(self):
        return 12.2639 / np.sqrt(self.voltage_kv * 1000 + 0.97845e-6 * (self.voltage_kv * 1000) ** 2)

    @property
    def wavelength_pixels(self):
        return self.wavelength_angstroms / self.pixel_size_angstroms

    @property
    def precomputed_amplitude_contrast(self):
        if np.abs(self.amplitude_contrast - 1.0) < np.finfo(self.amplitude_contrast).eps:
            return np.pi / 2.0
        else:
            return np.arctan(self.amplitude_contrast / np.sqrt(1.0 - self.amplitude_contrast**2))

    @property
    def spherical_aberration_pixels(self):
        return (self.spherical_aberration_mm * 1000 * 1000 * 10) / self.pixel_size_angstroms

    def defocus_given_azimuth(self, azimuth: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 0.5 * (
            self.defocus1_pixels
            + self.defocus2_pixels
            + np.cos(2.0 * (azimuth - self.defocus_angle_radians)) * (self.defocus1_pixels - self.defocus2_pixels)
        )

    def phase_shift(
        self, frequency: Union[float, np.ndarray], azimuth: Union[float, np.ndarray] = 0.0
    ) -> Union[float, np.ndarray]:
        return (
            np.pi
            * self.wavelength_pixels
            * frequency**2
            * (
                self.defocus_given_azimuth(azimuth)
                - 0.5 * self.wavelength_pixels**2 * frequency**2 * self.spherical_aberration_pixels
            )
            + self.additional_phase_shift_radians
            + self.precomputed_amplitude_contrast
        )

    @overload
    def evaluate(self, frequency: float, azimuth: float = 0.0) -> float:
        ...

    @overload
    def evaluate(self, frequency: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
        ...

    @overload
    def evaluate(self, frequency: float, azimuth: np.ndarray) -> np.ndarray:
        ...

    @overload
    def evaluate(self, frequency: np.ndarray) -> np.ndarray:
        ...

    def evaluate(
        self, frequency: Union[float, np.ndarray], azimuth: Union[float, np.ndarray] = 0.0
    ) -> Union[float, np.ndarray]:
        return -np.sin(self.phase_shift(frequency, azimuth))

    @property
    def powerspectrum_1d(self) -> np.ndarray:
        fh = FrequencyHelper(
            size=calculate_diagonal_radius(self.box_size), pixel_size_angstroms=self.pixel_size_angstroms
        )
        return self.evaluate(fh.spatial_frequency_pixels) ** 2

    @property
    def frequency_pixels_1d(self) -> np.ndarray:
        fh = FrequencyHelper(
            size=calculate_diagonal_radius(self.box_size), pixel_size_angstroms=self.pixel_size_angstroms
        )
        return fh.spatial_frequency_pixels

    @property
    def frequency_angstroms_1d(self) -> np.ndarray:
        fh = FrequencyHelper(
            size=calculate_diagonal_radius(self.box_size), pixel_size_angstroms=self.pixel_size_angstroms
        )
        return fh.spatial_frequency_angstroms

    @property
    def powerspectrum_2d(self) -> np.ndarray:
        fh = FrequencyHelper(size=(self.box_size, self.box_size), pixel_size_angstroms=self.pixel_size_angstroms)
        return self.evaluate(fh.spatial_frequency_pixels, azimuth=fh.azimuth) ** 2
