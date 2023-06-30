from pydantic import BaseModel
import numpy as np
from contrasttransferfunction import ContrastTransferFunction
from contrasttransferfunction.spectrumhelpers import ctffind_preproc, radial_average
from contrasttransferfunction.utils import calculate_diagonal_radius

DEFAULT_BOXSIZE=512

class CtfFit(BaseModel):
    #spectrum: np.ndarray
    ctf: ContrastTransferFunction
    cross_correlation: float
    ctf_accuracy: float
    median_cc: float
    #cross_correlation_array: np.ndarray
    #defocus_array: np.ndarray
    

    class Config:
        arbitrary_types_allowed = True


    
    
    @classmethod
    def fit_1d(cls, spectrum:np.ndarray, pixel_size_angstrom: float, low_defocus:float=3000, high_defocus:float=120000,**kwargs):
        if spectrum.ndim == 2:
            spectrum  = ctffind_preproc(spectrum, pixel_size_angstroms=pixel_size_angstrom)
            spectrum_1d = radial_average(spectrum)
        elif spectrum.ndim == 1:
            spectrum_1d = spectrum
        else:
            raise ValueError('Only fits from 1D or 2D powespectr are supported')
        
        exp_mean = spectrum_1d.mean()
        exp_std = spectrum_1d.std()
        spectrum_1d -= exp_mean
        
        defocus = np.logspace(np.log10(low_defocus), np.log10(high_defocus), DEFAULT_BOXSIZE)
        defocus = np.tile(defocus,(calculate_diagonal_radius(DEFAULT_BOXSIZE),1))
        defocus = np.transpose(defocus)

        my_ctf = ContrastTransferFunction(pixel_size_angstroms=pixel_size_angstrom, defocus1_angstroms=defocus, defocus2_angstroms=defocus,**kwargs)

        frequency = np.tile(my_ctf.frequency_pixels_1d,(512,1))

        comparison_array = my_ctf.evaluate(frequency=frequency)**2
        comparison_array_mean = np.mean(comparison_array,axis=1,keepdims=True)
        comparison_array_std = np.std(comparison_array,axis=1)
        comparison_array -= comparison_array_mean

        correlation = np.dot(comparison_array[:,150:],spectrum_1d[150:])
        correlation /= (comparison_array_std*exp_std)
        fit_index = np.argmax(correlation)
        fit_defocus = defocus[fit_index,0]
        
        left_delta = right_delta = 0.0
        if fit_index > 0:
            left_delta = np.abs(fit_defocus - defocus[fit_index-1,0])
        if fit_index < correlation.shape[0]-1:
            right_delta = np.abs(fit_defocus - defocus[fit_index+1,0])
        fit_accuracy = max(left_delta,right_delta)
        return(
            cls(
                #spectrum=spectrum,
                ctf=ContrastTransferFunction(
                    defocus1_angstroms=fit_defocus,
                    defocus2_angstroms=fit_defocus,
                    pixel_size_angstroms=pixel_size_angstrom
                ),
                cross_correlation=correlation[fit_index],
                ctf_accuracy=fit_accuracy,
                median_cc=np.median(correlation),                
                #cross_correlation_array=correlation,
                #defocus_array=defocus[:,0]
            )
        )
    
    def create_overlay(self) -> np.ndarray:
        overlay = self.spectrum.copy()
        overlay[overlay.shape[0]//2:,:]  = self.ctf.powerspectrum_2d[overlay.shape[0]//2:,:]
        return(overlay)
