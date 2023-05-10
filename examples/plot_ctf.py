import sys

import matplotlib.pyplot as plt

from contrasttransferfunction import ContrastTransferFunction

ctf = ContrastTransferFunction(
    pixel_size_angstroms=4.24, defocus1_angstroms=8000, defocus2_angstroms=16000, defocus_angle_degrees=30.0
)

plt.plot(ctf.get_frequency_angstroms_1d(), ctf.get_powerspectrum_1d())
plt.show()

plt.imshow(ctf.get_powerspectrum_2d())
plt.show()
sys.exit()
