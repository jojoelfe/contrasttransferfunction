import pytest
import numpy as np

from contrasttransferfunction.contrasttransferfunction import ContrastTransferFunction
from contrasttransferfunction.ctffit import CtfFit

@pytest.mark.parametrize("defocus,expected", [(a, a) for a in np.linspace(8000,20000,20)])
def test_fit(defocus, expected):
    ctf = ContrastTransferFunction(defocus1_angstroms=defocus, defocus2_angstroms=defocus, pixel_size_angstroms=2.0)

    fit_d = CtfFit.fit_1d(ctf.powerspectrum_2d,pixel_size_angstrom=2.0)

    assert(np.abs(fit_d.ctf.defocus1_angstroms-expected)<= fit_d.ctf_accuracy*2)
