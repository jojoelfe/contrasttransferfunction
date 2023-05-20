import numpy as np
from scipy.ndimage import mean, uniform_filter

from contrasttransferfunction.utils import distance_from_center_array

# These functions aim to replicate ctffinds preprocessing of the amplitude spectrum


def downscale_spectrum(spectrum: np.ndarray, box_size: int = 512) -> np.ndarray:
    # Downscale the spectrum to the desired box size
    half_box_size = box_size // 2
    spectrum = np.fft.rfft2(spectrum)
    spectrum = np.concatenate(
        (spectrum[:half_box_size, 0:half_box_size], spectrum[-half_box_size:, 0:half_box_size]), axis=0
    )
    spectrum = np.fft.irfft2(spectrum, s=(box_size, box_size))
    return spectrum


def adjust_central_cross(spectrum: np.ndarray, pixel_size_angstroms: float = 1.0, minimum_resolution: float = 30.0):
    # Reduce the high intensity values in the central cross to improve
    # artifacts in later processing
    if spectrum.shape[0] != spectrum.shape[1]:
        msg = "Spectrum must be square"
        raise ValueError(msg)
    if (spectrum.shape[0] % 2) != 0:
        msg = "Spectrum must have even dimensions"
        raise ValueError(msg)

    distance_center = distance_from_center_array(spectrum.shape[0])
    # Get the mean and standard deviation of the spectrum between desired
    # low resolution and Nyquist
    to_analyze = spectrum[
        (distance_center > spectrum.shape[0] / 2 * pixel_size_angstroms / minimum_resolution)
        & (distance_center <= spectrum.shape[0] / 2)
    ]
    mean = int(to_analyze.mean())
    stddev = to_analyze.std()

    spectrum /= stddev
    boxsize = spectrum.shape[0]
    central_cross = np.zeros_like(spectrum, dtype=bool)
    central_cross[boxsize - 1 : boxsize + 1, :] = 1
    central_cross[:, boxsize - 1 : boxsize + 1] = 1
    spectrum[central_cross & (spectrum > mean + 10 * stddev)] = mean + 10 * stddev
    return spectrum


def subtract_baseline(spectrum: np.ndarray):
    # TODO: size should be derived from size of spectrum and lowres-limit
    a = uniform_filter(spectrum, size=100, mode="nearest")
    return spectrum - a


def cosine_highpass(spectrum: np.ndarray, pixel_size_angstroms: float = 1.0) -> np.ndarray:
    mask_radius = float(spectrum.shape[0]) * pixel_size_angstroms / 16.0
    mask_edge = float(spectrum.shape[0]) * pixel_size_angstroms / max(pixel_size_angstroms * 2.0, 4.0)
    mask_radius = mask_radius - mask_edge * 0.5
    mask_radius_plus_edge = mask_radius + mask_edge

    distance_center = distance_from_center_array(spectrum.shape[0])

    cos_filter = np.zeros_like(spectrum)
    outside_mask = distance_center >= mask_radius
    cos_filter[outside_mask] = 1 - (
        (1 + np.cos(np.pi * (distance_center[outside_mask] - mask_radius) / mask_edge)) / 2.0
    )
    cos_filter[distance_center >= mask_radius_plus_edge] = 1.0

    return spectrum * cos_filter


def radial_average(spectrum: np.ndarray) -> np.ndarray:
    distance_from_center = distance_from_center_array(spectrum.shape[0])
    bins = np.around(distance_from_center).astype(np.int32)
    return mean(spectrum, labels=bins, index=np.arange(1, bins.max() + 1))


def ctffind_1d_preproc(spectrum: np.ndarray, pixel_size_angstroms) -> np.ndarray:
    spectrum = downscale_spectrum(spectrum, 512)
    spectrum = adjust_central_cross(spectrum)
    spectrum = subtract_baseline(spectrum)
    spectrum = cosine_highpass(spectrum, pixel_size_angstroms=pixel_size_angstroms)
    return radial_average(spectrum)
