#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare different time (in)dependent HUXt boundary conditions 
Created on Tue Mar 11 14:40:10 2025
@author: mrutala
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import datetime
import os
import sys

sys.path.append('HUXt/code/')
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
from scipy import ndimage
from scipy import stats
import huxt_inputs_wsa as Hin_wsa

from astroquery.jplhorizons import Horizons

try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

# =============================================================================
# Time independent boundaries: MAS vs. GONG-WSA vs. ADAPT-WSA
# =============================================================================

# Setup runtime
runstart = datetime.datetime(2015, 1, 1)
runstop = datetime.datetime(2015, 5, 1)
runtime = (runstop - runstart).days


# # Get a single Carrington Rotation for a stationary solution
# start_cr, start_cr_lon = Hin.datetime2huxtinputs(start)
# start_cr_lat = 0.0*u.deg
# vr_in = Hin.get_MAS_long_profile(start_cr, start_cr_lat)

# # Get all Carrington Rotations (CRs) in simtime
# vcarr_MAS = np.zeros([*vr_in.shape, simtime])
# for day in range(simtime): 
#     cr, cr_lon = Hin.datetime2huxtinputs(start + datetime.timedelta(days=day))
#     vcarr_MAS[:,day] = Hin.get_MAS_long_profile(cr, 0.0*u.deg)
    
# # Download extra data so we can smooth vcarr in time
# vcarr_MAS_forsmooth = np.zeros([*vr_in.shape, simtime+52])
# for day in np.arange(-26, simtime+26): 
#     cr, cr_lon = Hin.datetime2huxtinputs(start + datetime.timedelta(days=int(day)))
#     print(cr, cr_lon.value)
#     vcarr_MAS_forsmooth[:,day+26] = Hin.get_MAS_long_profile(cr, 0.0*u.deg)
# kernel = np.full([1,26], 1)/26.
# vcarr_MAS_smooth = ndimage.convolve(vcarr_MAS_forsmooth, kernel, mode='constant', cval=0.0)[:, 26:-26]

# GONG-WSA and ADAPT-WSA are both initialized from the Integrated Space Weather Analysis System
time_wsa, vcarr_wsa, bcarr_wsa = Hin_wsa.get_WSA_time_dependent_boundary(runstart, runstop,  lat=0*u.deg, source='GONG_Z')

# Get back-mapped OMNI for comparison
time_omni, vcarr_omni, bcarr_omni = Hin.generate_vCarr_from_OMNI(runstart, runstop)
vcarr_omni_bm = vcarr_omni.copy() 
horizons_time = {'start': runstart.strftime("%Y-%m-%d %H:%M:%S"),
                 'stop': runstop.strftime("%Y-%m-%d %H:%M:%S"),
                 'step': '1d'}
earth_pos = Horizons(id = '399', location = '500@0', epochs = horizons_time).vectors().to_pandas()
rmin = 21.5 * u.solRad
for i in range(vcarr_omni_bm.shape[1]): 
    Earth_R_km = (earth_pos['range'].iloc[i] * u.AU).to(u.km)
    vcarr_omni_bm[:,i] = Hin.map_v_boundary_inwards(vcarr_omni[:,i]*u.km/u.s, 
                                                    Earth_R_km.to(u.solRad), rmin)
    
    
fig, axs = plt.subplots(ncols=2)
axs[0].imshow(vcarr_wsa, vmin=400, vmax=600)
axs[0].set(title='WSA/GONG Z')
axs[1].imshow(vcarr_omni_bm, vmin=400, vmax=600)
axs[1].set(title='Backmapped OMNI')


# %% Plot the results at Earth
# =============================================================================
# 
# =============================================================================

# WSA
earth_wsa_dfs = []
for l in np.arange(-10, 10+1, 2):
    time_wsa, vcarr_wsa, bcarr_wsa = Hin_wsa.get_WSA_time_dependent_boundary(runstart, runstop,  lat=l*u.deg, source='GONG_Z')
    model_wsa = Hin.set_time_dependent_boundary(vcarr_wsa, time_wsa, runstart, runtime*u.day, 
                                                r_min=21.5*u.solRad, r_max=1290*u.solRad, dt_scale=10, latitude=0*u.deg,
                                                ) # bgrid_Carr = bcarr_omni)
    model_wsa.solve([])
    earth_wsa = HA.get_observer_timeseries(model_wsa, observer = 'Earth')
    earth_wsa_dfs.append(earth_wsa)
    print(l)

# OMNI
model_omni = Hin.set_time_dependent_boundary(vcarr_omni_bm, time_omni, runstart, runtime*u.day, 
                                             r_min=21.5*u.solRad, r_max=1290*u.solRad, dt_scale=10, latitude=0*u.deg,
                                             ) # bgrid_Carr = bcarr_omni)
model_omni.solve([])
earth_omni = HA.get_observer_timeseries(model_omni, observer = 'Earth')

data = Hin.get_omni(runstart, runstop)

fig, ax = plt.subplots()
for df in earth_wsa_dfs:
    ax.plot(df['time'], df['vsw'], color='C4', alpha=0.3)
ax.plot(earth_omni['time'], earth_omni['vsw'], label='bOMNI-HUXt')
ax.scatter(data['datetime'], data['V'], marker='.', color='black', label='Data')
ax.set(xlim=[runstart, runstop])
ax.legend()

plt.show()
