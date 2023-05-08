import matplotlib.pyplot as plt

from contrasttransferfunction import ContrastTransferFunction
from contrasttransferfunction import FrequencyHelper1D as Freq1D

my_ctf = ContrastTransferFunction(pixel_size_A=2.0)

my_freq = Freq1D()

plt.plot(my_freq.spatial_frequency_A, my_ctf.evaluate(my_freq.spatial_frequency_pixels))
plt.show()
