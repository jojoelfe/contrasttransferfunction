import numpy as np
from pydantic import BaseModel

class ContrastTransferFunction(BaseModel):
    defocus1_A: float = 10000.0
    defocus2_A: float = 10000.0
    defocus_angle_degrees: float = 0.0
    additional_phase_shift_degrees: float = 0.0
    voltage_kV: float = 300.0
    spherical_aberration_mm: float = 2.7
    amplitude_contrast: float = 0.07
    pixel_size_A: float = 1.0

    @property
    def defocus1_pixels(self):
        return self.defocus1_A / self.pixel_size_A
    
    @property
    def defocus2_pixels(self):
        return self.defocus2_A / self.pixel_size_A

    @property
    def defocus_angle_radians(self):
        return np.deg2rad(self.defocus_angle_degrees)
    
    @property
    def additional_phase_shift_radians(self):
        return np.deg2rad(self.additional_phase_shift_degrees)
    
    @property
    def wavelength_A(self):
        return 12.2643247 / np.sqrt(self.voltage_kV * 1000  + 0.978466e-6 * (self.voltage_kV * 1000) **2)

    @property
    def wavelength_pixels(self):
        return self.wavelength_A / self.pixel_size_A
    
    @property
    def precomputed_amplitude_contrast(self):
        if np.abs(self.amplitude_contrast - 1.0) < 1e-3:
            return np.pi / 2.0
        else:
            return np.arctan(self.amplitude_contrast / np.sqrt(1.0 - self.amplitude_contrast ** 2))

    @property
    def spherical_aberration_pixels(self):
        return (self.spherical_aberration_mm * 1000 * 1000 * 10) / self.pixel_size_A
    
    def defocus_given_azimuth(self, azimuth):
        return 0.5 * (self.defocus1_pixels + self.defocus2_pixels + np.cos(2.0 * (azimuth - self.defocus_angle_radians)) * (self.defocus1_pixels - self.defocus2_pixels))
    
    def phase_shift(self, frequency, azimuth= 0.0):
        return np.pi * self.wavelength_pixels * frequency**2 * (self.defocus_given_azimuth(azimuth) - 0.5 * self.wavelength_pixels ** 2 * frequency ** 2 * self.spherical_aberration_pixels) + self.additional_phase_shift_radians + self.precomputed_amplitude_contrast
    
    def evaluate1D(self, frequency: np.ndarray = None, azimuth=0.0):
        return -np.sin(self.phase_shift(frequency, azimuth))
    
    def evaluate2D(self, frequency: np.ndarray = None):
        raise NotImplementedError


