import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pysm3
import pysm3.units as u
import scipy.constants as const
from scipy.integrate import quad

import warnings
warnings.filterwarnings("ignore")

kB = const.k
c = const.c
h = const.h
Tcmb = 2.275

# Blackbody at temperature T
def BB(nu, T):
    
    x = h * nu / kB / T
    
    return 2. * h * nu * nu * nu / c / c / (np.exp(x) - 1.)

# Blackbody derivative
def dBdT(nu, T):
    
    x = h * nu / kB / T
    
    return 2. * kB * nu * nu / c / c * x * x * np.exp(x) / (np.exp(x) - 1.) / (np.exp(x) - 1.)

# Rayleigh-Jeans approx derivative
def dBrjdT(nu):
        
    return 2. * kB * nu * nu / c / c

#Using description of dust and synch from: https://arxiv.org/abs/2111.02425
# Modified Blackbody for dust in RJ units
def dust(nu, T, nu0, A, beta):
    
    return A * (nu/nu0)**(beta - 2) *  BB(nu, T) / BB(nu0, T)

# Synchrotron (simple) in RJ units
def synch(nu, nu0, A, beta):
    
    return A * (nu / nu0)**beta

# Zodiacal Light Emission in brightness (Maris et al. 2006)
def zodiac(nu, T):

	zle = np.zeros(nu.size)
	
	for i in range(nu.size):

		if nu[i] > 1875.e9:
	
			pre_factor = 1.
			
		elif nu[i] <= 1875.e9:
		
			# Pre-factor fixed to 1 to agree with Ludo's model
		
			pre_factor = 1. #pre_factor = (nu[i] / 1875.e9)**2
			
		zle[i] = 3.e-7 * BB(nu[i], T) * pre_factor

	return zle
