import numpy as np

from contrasttransferfunction import ContrastTransferFunction
from contrasttransferfunction.spectrumhelpers import radial_average
from contrasttransferfunction.utils import calculate_diagonal_radius


def test_radial_average_length():
    ctf = ContrastTransferFunction()
    result = radial_average(ctf.powerspectrum_2d)
    assert len(result) == calculate_diagonal_radius(ctf.powerspectrum_2d.shape[0])


def test_radial_average():
    # Create sample spectrum
    spectrum = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # Calculate radial average
    radial_avg = radial_average(spectrum)
    # Define expected result
    expected_result = np.array([8.5, 8.5])
    # Check that result matches expected result
    np.testing.assert_array_equal(radial_avg, expected_result)

def test_radial_average_of_2d_powerspectrum():
    ctf = ContrastTransferFunction(box_size=1024,defocus=10000,pixel_size_angstroms=4.0)
    radial_avg = radial_average(ctf.powerspectrum_2d)
    assert np.allclose(radial_avg, ctf.powerspectrum_1d,atol=0.05)

