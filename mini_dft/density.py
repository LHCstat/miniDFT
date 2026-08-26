import numpy as np
import fft_grid

def get_density(c):
    psi_r = fft_grid.reciprocal_to_real(c)
    density = 2*np.abs(psi_r)**2
    return density