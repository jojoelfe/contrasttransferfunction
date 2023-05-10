import numpy as np

from contrasttransferfunction import ContrastTransferFunction


def test_1dpowerspectrum():
    ctf = ContrastTransferFunction()
    powerspectrum = ctf.get_powerspectrum_1d()
    assert len(powerspectrum) == int(256 * np.sqrt(2.0))
    assert np.isclose(np.min(powerspectrum), 0.0, atol=1e-4)
    assert np.isclose(np.max(powerspectrum), 1.0)


def test_2dpowerspectrum():
    ctf = ContrastTransferFunction()
    powerspectrum = ctf.get_powerspectrum_2d()
    assert powerspectrum.shape == (512, 512)
    assert np.isclose(np.min(powerspectrum), 0.0, atol=1e-4)
    assert np.isclose(np.max(powerspectrum), 1.0)
