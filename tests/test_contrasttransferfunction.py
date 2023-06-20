import numpy as np
import pytest
from pycistem.core import CTF

from contrasttransferfunction import ContrastTransferFunction
from contrasttransferfunction.utils import calculate_diagonal_radius


def test_1dpowerspectrum():
    ctf = ContrastTransferFunction()
    powerspectrum = ctf.powerspectrum_1d
    assert len(powerspectrum) == calculate_diagonal_radius(ctf.powerspectrum_2d.shape[0])
    assert np.isclose(np.min(powerspectrum), 0.0, atol=1e-4)
    print(np.max(powerspectrum))
    assert np.isclose(np.max(powerspectrum), 1.0,atol=1e-4)


def test_2dpowerspectrum():
    ctf = ContrastTransferFunction()
    powerspectrum = ctf.powerspectrum_2d
    assert powerspectrum.shape == (512, 512)
    assert np.isclose(np.min(powerspectrum), 0.0, atol=1e-4)
    assert np.isclose(np.max(powerspectrum), 1.0, atol=1e-4)

@pytest.mark.parametrize("defocus1,defocus2,asti_angle,kV,cs,ac,pixel_size", [
    (12000,12000,0.0,300.0,2.7,0.07,1.0),
    (12000,12000,0.0,200.0,0.01,0.12,1.3),
    (1200,1200,0.0,300.0,2.7,0.07,1.5),
    (24000,12000,30.0,300.0,2.7,0.07,0.9),
    (24000,24000,0.0,300.0,2.7,0.07,2.0),
    (9000,7000,180.0,300.0,2.7,0.07,1.0),
    (12000,9000,0.0,200.0,2.7,0.07,0.9),
    (12000,12000,60.0,200.0,2.7,0.02,0.75),
    (12000,3895,45.0,200.0,2.7,0.07,2.2),
])
def test_1dpowerspectrum_consisten_cistem(defocus1,defocus2,asti_angle,kV,cs,ac,pixel_size):
    ctf = ContrastTransferFunction(defocus1_angstroms=defocus1, defocus2_angstroms=defocus2, defocus_angle_degrees=asti_angle,voltage_kv=kV,spherical_aberration_mm=cs,amplitude_contrast=ac,pixel_size_angstroms=pixel_size)
    powerspectrum = ctf.powerspectrum_1d
    frequency_pixel = ctf.frequency_pixels_1d

    cisCTF = CTF(kV=kV,cs=cs,ac=ac,defocus1=defocus1,defocus2=defocus2,astig_angle=asti_angle,pixel_size=pixel_size)
    cisTEM_powerspectrum = np.array([cisCTF.Evaluate(freq**2.0,0.0)**2.0 for freq in frequency_pixel])

    assert np.allclose(powerspectrum,cisTEM_powerspectrum,atol=3e-4)
