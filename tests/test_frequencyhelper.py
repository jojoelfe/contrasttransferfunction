import numpy as np
import pytest

from contrasttransferfunction import FrequencyHelper
from contrasttransferfunction.utils import calculate_diagonal_radius
from contrasttransferfunction.spectrumhelpers import radial_average


def test_size():
    f1 = FrequencyHelper(size=10)
    assert len(f1.spatial_frequency_pixels) == 10
    f1 = FrequencyHelper(size=(10, 10))
    assert f1.spatial_frequency_pixels.shape == (10, 10)
    with pytest.raises(ValueError):
        f1 = FrequencyHelper(size=(20, 10))


def test_frequency_pixels_1d():
    fh = FrequencyHelper(size=25)
    assert fh.spatial_frequency_pixels[-1] == 0.5 * np.sqrt(2.0)


def test_frequency_pixels2d():
    fh = FrequencyHelper(size=(25, 25))
    assert fh.spatial_frequency_pixels[12, 12] == 0
    assert np.isclose(fh.spatial_frequency_pixels[-1, -1], 0.5 * np.sqrt(2.0))
    assert np.isclose(fh.spatial_frequency_pixels[0, 12], 0.5)


def test_frequency_angstroms_1d():
    fh = FrequencyHelper(size=25, pixel_size_angstroms=3.0)
    assert fh.spatial_frequency_angstroms[-1] == 0.5 * np.sqrt(2.0) / 3.0


def test_wavelength_angstroms_1d():
    fh = FrequencyHelper(size=25, pixel_size_angstroms=3.0)
    assert np.isclose(fh.spatial_wavelength_angstroms[-1], 3.0 / (0.5 * np.sqrt(2.0)))


def test_wavelength_angstroms_2d():
    fh = FrequencyHelper(size=(25, 25), pixel_size_angstroms=3.0)
    assert np.isclose(fh.spatial_wavelength_angstroms[-1, -1], 3.0 / (0.5 * np.sqrt(2.0)))
    assert np.isclose(fh.spatial_wavelength_angstroms[0, 12], 3.0 / 0.5)


def test_azimuth():
    fh = FrequencyHelper(size=(25, 25))
    # 0 is to the right
    assert fh.azimuth[12, 24] == 0.0
    # 90 is on top
    assert fh.azimuth[0, 12] == np.pi / 2.0

@pytest.mark.parametrize("size", [8,32,128,512,1024])
def test_2d_equals_1d(size):
    fh2d = FrequencyHelper(size=(size, size))
    fh1d = FrequencyHelper(size=calculate_diagonal_radius(size))
    print(radial_average(fh2d.spatial_frequency_pixels))
    print(fh1d.spatial_frequency_pixels)
    assert np.allclose(radial_average(fh2d.spatial_frequency_pixels), fh1d.spatial_frequency_pixels,rtol=0.1)

