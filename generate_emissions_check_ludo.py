import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pysm3
import pysm3.units as u
import scipy.constants as const
from scipy.integrate import quad
import define_spectra as ds

import warnings
warnings.filterwarnings("ignore")

############### Parameters ##############################
nside = 128 # healpy nside

# Start, step and end frequencies
nu_i = 0.1 # start frequency (GHz)
nu_f = 1000000 # end frequency (GHz)
dnu = 10 # frequency step (GHz)

# Sky components
# Dust
nu_dust = 353 # dust pivot frequency (GHz)
dust_model = "d1" # dust pysm model
Td = 19.5 # dust temperature
beta_d = 1.55 # dust index
# Synchrotron
nu_synch = 30 # synch pivot frequency (GHz)
synch_model = "s1" # synch pysm model
beta_s1 = -3.1 # synch index
beta_s2 = -3.2 # synch index Ludo
# Zodiacal Light
Tzle = 240.

# CMB temperature
Tcmb = 2.725

# Star at 1000 K
Tstar = 1000
theta_star = 0.01 # size in arcminutes

# Telescope Resolution
theta_beam = 30 # size in arcminutes

# HWP
Thwp = 20 # hwp temperature

#######################################################

# Frequency array
nu = np.linspace(nu_i, nu_f, int((nu_f-nu_i)/dnu+1))*1.e9 

# Create dust model with pysm3 at 353 GHz (pivot) to define dust amplitude
sky = pysm3.Sky(nside=nside, preset_strings=[dust_model])
map_dust_353GHz = sky.get_emission(nu_dust * u.GHz)
# Define dust amplitude as:
dust_rms = np.sqrt(np.mean(map_dust_353GHz[0]*map_dust_353GHz[0])) # I map rms 
dust_max = max(map_dust_353GHz[0]) # I map max
dust_mean = np.mean(map_dust_353GHz[0]) # I map mean
Ad = dust_max / 1.e6 # amplitude in Krj

# Create synch model with pysm3 at 30 GHz (pivot) to define synch amplitude
sky = pysm3.Sky(nside=nside, preset_strings=[synch_model])
map_synch_30GHz = sky.get_emission(nu_synch * u.GHz)
# Define synch amplitude as:
synch_rms = np.sqrt(np.mean(map_synch_30GHz[0]*map_synch_30GHz[0])) # I map rms 
synch_max = max(map_synch_30GHz[0]) # I map max
synch_mean = np.mean(map_synch_30GHz[0]) # I map mean
As = synch_max / 1.e6 # amplitude in Krj

# Generate emissions
cmb = ds.BB(nu, Tcmb) # CMB
star = ds.BB(nu, Tstar) * (theta_star / theta_beam) * (theta_star / theta_beam) # Star seen by LiteBIRD
dust = ds.dust(nu, Td, nu_dust * 1.e9, Ad, beta_d) * ds.dBrjdT(nu) # Dust (include conversion from RJ to radiance)
synch1 = ds.synch(nu, nu_synch * 1.e9, As, beta_s1) * ds.dBrjdT(nu) # Synchrotron (include conversion from RJ to radiance)
synch2 = ds.synch(nu, nu_synch * 1.e9, As, beta_s2) * ds.dBrjdT(nu) # Synchrotron (include conversion from RJ to radiance)
zodiac = ds.zodiac(nu, Tzle)
hwp = ds.BB(nu, Thwp) # hwp
stop5K = ds.BB(nu, 5) # 5K stop
stop2K = ds.BB(nu, 2) # 2K stop

# Plot
plt.figure(figsize=(16, 9))
plt.loglog(nu/1e9, cmb, label='cmb')
plt.loglog(nu/1e9, star, label='star')
plt.loglog(nu/1e9, dust, label='dust')
plt.loglog(nu/1e9, zodiac, label='zodiacal')
plt.loglog(nu/1e9, synch1, label='synch -3.1')
plt.loglog(nu/1e9, synch2, label='synch -3.2')
plt.loglog(nu/1e9, hwp, label='HWP - emissivity = 1')
plt.loglog(nu/1e9, stop5K, label='5K - emissivity = 1')
plt.loglog(nu/1e9, stop2K, label='2K - emissivity = 1')

plt.ylabel('[W $m^{-2} Hz^{-1} sr^{-1}$]', fontsize=16)
plt.xlabel('$\\nu$ [GHz]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16)
plt.ylim(1.e-35, 1.e-10)

plt.show()
