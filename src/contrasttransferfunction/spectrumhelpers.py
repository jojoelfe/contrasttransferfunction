import numpy as np

# These functions aim to replicate ctffiinds preprocessing of the amplitude spectrum

def downscale_spectrum(spectrum: np.ndarray, box_size: int = 512) -> np.ndarray:
    half_box_size = box_size // 2
    spectrum = np.fft.rfft2(spectrum)
    spectrum = np.concatenate((spectrum[:half_box_size,0:half_box_size],
                               spectrum[-half_box_size:,0:half_box_size]), axis=0)
    spectrum = np.fft.irfft2(spectrum, s=(box_size,box_size))

    return spectrum

def adjust_central_cross(spectrum: np.ndarray,
                         pixel_size_angstroms: float=1.0,
                         minimum_resolution: float=30.0):
    x_inds, y_inds = np.ogrid[: spectrum.shape[1], : spectrum.shape[0]]
    mid_x, mid_y = (np.array(spectrum.shape[::-1]) - 1) / float(2)
    distance_center = ((y_inds - mid_y) ** 2 + (x_inds - mid_x) ** 2) ** 0.5
    to_analyze = spectrum[(distance_center > spectrum.shape[0]/2 *
                                             pixel_size_angstroms/ minimum_resolution) &
                          (distance_center <= spectrum.shape[0]/2)]
    mean = int(to_analyze.mean())
    stddev = to_analyze.std()
    spectrum /= stddev
    boxsize = spectrum.shape[0]
    central_cross = np.zeros_like(spectrum,dtype=bool)
    central_cross[boxsize-1:boxsize+1,:] = 1
    central_cross[:,boxsize-1:boxsize+1] = 1
    spectrum[central_cross & (spectrum > mean + 10*stddev)] = mean + 10*stddev
    return spectrum

def convolve2d(img, kernel):
    # Determine amount of zero-padding required for each dimension
    pad_height = int((kernel.shape[0]-1)/2)
    pad_width = int((kernel.shape[1]-1)/2)

    # Pad the input image with the avergae value of the image borders
    average_border_value = np.mean(img[0,:])
    img_padded = np.pad(img,
                        ((pad_height,pad_height),(pad_width,pad_width)),
                        mode="constant",
                        constant_values=average_border_value)

    # Create an array of submatrices
    sub_shape = tuple(np.subtract(img_padded.shape, kernel.shape) + 1)     # alias for the function
    strd = np.lib.stride_tricks.as_strided

    # make an array of submatrices
    submatrices = strd(img_padded,kernel.shape + sub_shape,img_padded.strides * 2)

    # Perform convolution
    convolved_matrix = np.einsum("ij,ijkl->kl", kernel, submatrices)

    return convolved_matrix

def subtract_baseline(spectrum: np.ndarray):
    kernel = np.zeros((111,111))
    y, x = np.ogrid[:111, :111]
    dist_from_center = np.sqrt((x - 35.5)**2 + (-35.5)**2)
    kernel[dist_from_center<=35.5] = 1.0
    kernel = kernel/np.sum(kernel)

    a = convolve2d(spectrum, kernel)

    return a


def cosine_highpass(spectrum: np.ndarray, pixel_size_angstroms: float=1.0) -> np.ndarray:

    mask_radius = float(spectrum.shape[0]/2.0) * pixel_size_angstroms / 8.0
    mask_edge = float(spectrum.shape[0]/2.0) * pixel_size_angstroms / max(pixel_size_angstroms*2.0, 8.0)
    mask_radius = mask_radius - mask_edge * 0.5
    mask_radius_plus_edge = mask_radius + mask_edge

    x_inds, y_inds = np.ogrid[: spectrum.shape[1], : spectrum.shape[0]]
    mid_x, mid_y = (np.array(spectrum.shape[::-1]) - 1) / float(2)
    distance_center = ((y_inds - mid_y) ** 2 + (x_inds - mid_x) ** 2) ** 0.5

    cos_filter = np.zeros_like(spectrum)
    cos_filter[distance_center >= mask_radius] = (1-((1 + np.cos(np.pi * (distance_center[distance_center >= mask_radius] - mask_radius) / mask_edge)) / 2.0))
    cos_filter[distance_center>=mask_radius_plus_edge] = 1.0

    return cos_filter

def radial_average(spectrum: np.ndarray) -> np.ndarray:
    sx, sy = spectrum.shape
    x, y = np.ogrid[0:sx, 0:sy]


    r = np.hypot(x - sx/2, y - sy/2)

    rad = np.arange(1, np.max(r), 1)
    intensity = np.zeros(len(rad))
    index = 0
    bin_size = 3
    for i in rad:
        mask = (np.greater(r, i - bin_size) & np.less(r, i + bin_size))
        values = spectrum[mask]
        intensity[index] = np.mean(values)
        index += 1
    return intensity

