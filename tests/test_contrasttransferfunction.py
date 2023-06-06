import numpy as np
import pytest

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
