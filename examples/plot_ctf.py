from contrasttransferfunction import ContrastTransferFunction as CTF
from contrasttransferfunction import FrequencyHelper1D as Freq1D
import matplotlib.pyplot as plt

my_ctf = CTF(pixel_size_A=2.0)

my_freq = Freq1D()

print(my_freq.spatial_wavelength_A)
plt.plot(my_freq.spatial_frequency_A, my_ctf.evaluate1D(my_freq.spatial_frequency_pixels))
plt.show()