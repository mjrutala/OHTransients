#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 13:09:46 2025

@author: mrutala
"""


import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import datetime
import os
import sys
import tqdm

from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries
from sunpy.coordinates import sun
from astropy.time import Time

sys.path.append('../HUXt/code/')
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
from scipy import ndimage
from scipy import stats

from astroquery.jplhorizons import Horizons

import huxt_inputs_wsa as Hin_wsa
import queryDONKI

try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

# HUXt can be easily initiated MAS, by specifying a carrington rotation number. 
# Data are downloaded from the Pred Sci Inc archive on demand
runstart = datetime.datetime(2024, 1, 1)
runstop = datetime.datetime(2024, 7, 1)
runtime = (runstop - runstart)


def fetch_omni(starttime, endtime):
    """
    A function to grab and process the OMNI COHO1HR data using FIDO
    
    Args:
        starttime : datetime for start of requested interval
        endtime : datetime for start of requested interval

    Returns:
        omni: Dataframe of the OMNI timeseries

    """
    trange = attrs.Time(starttime, endtime)
    dataset = attrs.cdaweb.Dataset('OMNI_COHO1HR_MERGED_MAG_PLASMA')
    result = Fido.search(trange, dataset)
    downloaded_files = Fido.fetch(result, path='../Data/OMNI/')

    # Import the OMNI data
    omni = TimeSeries(downloaded_files, concatenate=True).to_dataframe()
    omni = omni.rename(columns = {'ABS_B': 'B',
                                  'V': 'U'})
    
    # Set invalid data points to NaN
    id_bad = omni['U'] == 9999.0
    omni.loc[id_bad, 'U'] = np.NaN

    # create a BX_GSE field that is expected by some HUXt fucntions
    omni['BX_GSE'] = -omni['BR']
    
    # add an mjd column too
    omni['mjd'] = Time(omni.index).mjd

    return omni

# %% Get raw OMNI data ========================================================
# =============================================================================
padding = datetime.timedelta(days=27)
omni_df = fetch_omni(runstart-padding, runstop+padding)

# %% Get DONKI ICMEs @ OMNI ===================================================
# Assume a generous ICME duration
# =============================================================================
icme_df = queryDONKI.ICME(runstart, runstop, location='Earth', duration=2.0*u.day)

# %% OMNI - ICMEs =============================================================
# 
# =============================================================================
# Reformat icme_df for subtraction
_icme_df = icme_df.rename(columns = {'eventTime': 'Shock_time'})
_icme_df['ICME_end'] = [row['Shock_time'] + datetime.timedelta(days=(row['duration'])) for _, row in _icme_df.iterrows()]

qomni_df = Hin.remove_ICMEs(omni_df, _icme_df, interpolate=False, icme_buffer=0.5 * u.day, interp_buffer=1 * u.day,
                            params=['U', 'BX_GSE'], fill_vals=None)

qomni_df['carringtonLongitude'] = sun.L0(qomni_df.index).to(u.deg).value

# %% Gaussian Process Data Imputation =========================================
# 
# =============================================================================

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from scipy.spatial.distance import cdist
# import tensorflow_probability  as     tfp
import gpflow
from sklearn.cluster import KMeans

#set global variable scaling constants for GP
maxRescale = 100
subsetSize = 1000
min_l      = 0.0 
mid_l      = 0.05
max_l      = 0.2
init_var   = 3
period = 27/(runtime.total_seconds() / (24*60*60)) * maxRescale
n_samples = 30

X = qomni_df.dropna(axis='index', how='any')['mjd'].to_numpy('float64')[:, None]
Y = qomni_df.dropna(axis='index', how='any')['U'].to_numpy('float64')[:, None]

time_rescaler = MinMaxScaler((0, maxRescale))
time_rescaler.fit(X)
X_scaled = time_rescaler.transform(X)

val_rescaler = StandardScaler()
val_rescaler.fit(Y)
Y_scaled = val_rescaler.transform(Y)

# Simplify

XY_scaled = np.array(list(zip(X_scaled.flatten(), Y_scaled.flatten())))
n_clusters = int((runtime.total_seconds()/3600) / 12)
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(XY_scaled)
XY_clustered = kmeans.cluster_centers_

X_clustered, Y_clustered = XY_clustered.T[0], XY_clustered.T[1]
cluster_sort = np.argsort(X_clustered)
X_clustered = X_clustered[cluster_sort][:, None]
Y_clustered = Y_clustered[cluster_sort][:, None]

# =============================================================================
# scipy GP stuff, no longer used
# =============================================================================
# long_term_kernel = 2.0**2 * RBF(length_scale=1.0)
# # short_term_kernel = 1.0**2 * RBF(length_scale=0.1)
# irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=0.1, alpha=1.0)
# carrington_kernel = (1.0**2 * RBF(length_scale=10.0) * ExpSineSquared(length_scale=1.0, periodicity=period, periodicity_bounds="fixed")
# )
# kernel = long_term_kernel + irregularities_kernel + carrington_kernel
# # kernel = 1 * RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1e1))
# gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=1**2, n_restarts_optimizer=9)
# gaussian_process.fit(X_clustered, Y_clustered)
# gaussian_process.kernel_
# Xp = np.arange(qomni_df['mjd'].iloc[0], qomni_df['mjd'].iloc[-1], 1/24.)[:, None]
# Xp_scaled = time_rescaler.transform(Xp)
# mean_prediction, std_prediction = gaussian_process.predict(Xp_scaled, return_std=True)
# fig, ax = plt.subplots()
# ax.scatter(X_scaled, Y_scaled, label="Observations", color='black', marker='.', s=2)
# ax.scatter(X_clustered, Y_clustered, label='Inducing Points', color='C0', marker='o', s=6)
# ax.plot(Xp_scaled, mean_prediction, label="Mean prediction", color='C3')
# ax.fill_between(
#     Xp_scaled.ravel(),
#     mean_prediction - 1.96 * std_prediction,
#     mean_prediction + 1.96 * std_prediction,
#     alpha=0.5, color='C3',
#     label=r"95% confidence interval",
# )
# ax.legend()
# # ax.xlabel("$x$")
# # ax.ylabel("$f(x)$")
# # _ = plt.title("Gaussian process regression on noise-free dataset")
# ax.set(xlim=[40,60])

# =============================================================================
# GPFlow GP stuff
# =============================================================================
# signal_kernel = gpflow.kernels.RationalQuadratic(variance = init_var)
small_scale_kernel = gpflow.kernels.SquaredExponential(variance=2**2, lengthscales=1.0)
large_scale_kernel = gpflow.kernels.SquaredExponential(variance=2**2, lengthscales=10.0)
irregularities_kernel = gpflow.kernels.SquaredExponential(variance=1**2, lengthscales=1.0)
carrington_kernel = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=period)

signal_kernel = small_scale_kernel + large_scale_kernel + irregularities_kernel + carrington_kernel

# model = gpflow.models.GPR((X_clustered, Y_clustered), kernel=signal_kernel)
model = gpflow.models.GPR((X_clustered, Y_clustered), kernel=signal_kernel, noise_variance=1)

opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

# gpflow.utilities.print_summary(model)

Xp = qomni_df['mjd'].to_numpy('float64')[:, None]
Xp_scaled = time_rescaler.transform(Xp)

Yp_scaled_mean, Yp_scaled_var = model.predict_y(Xp_scaled)
Yp_scaled_mean, Yp_scaled_var = np.array(Yp_scaled_mean), np.array(Yp_scaled_var)
Yp_scaled_std = np.sqrt(Yp_scaled_var)

f_scaled_samples = model.predict_f_samples(Xp_scaled, n_samples, full_cov=False)
f_scaled_samples = np.array(f_scaled_samples)

Xo = time_rescaler.inverse_transform(Xp_scaled)
Yo_mu = val_rescaler.inverse_transform(Yp_scaled_mean)
Yo_sig = Yp_scaled_std * val_rescaler.scale_
Fo_samples = np.array([val_rescaler.inverse_transform(f) for f in f_scaled_samples])

fig, ax = plt.subplots()
ax.scatter(X, Y, label="Observations", color='black', marker='.', s=2, zorder=2)
# ax.scatter(X_clustered, Y_clustered, label='Inducing Points', color='C0', marker='o', s=6, zorder=4)

ax.plot(Xo, Yo_mu, label="Mean prediction", color='C3', zorder=0)
ax.fill_between(
    Xo.ravel(),
    (Yo_mu - 1.96 * Yo_sig).ravel(),
    (Yo_mu + 1.96 * Yo_sig).ravel(),
    alpha=0.1, color='C3',
    label=r"95% confidence interval", zorder=-2)

for Fo in Fo_samples[0:2]:
    ax.plot(Xo.ravel(), Fo.ravel(), lw=0.5, color='C5', alpha=0.75, zorder=1)

ax.legend()
ax.set(xlabel='Date [MJD]', ylabel='Solar Wind Speed [km/s]', 
       title='OMNI (@ 1AU), no ICMEs, GP Data Imputation')


# %% Backmap samples to 21.5 Rs, add CMEs =====================================
# 
# =============================================================================

def map_omni_inwards(runstart, runstop, df):
    
    # Format the OMNI DataFrame as HUXt expects it
    df['V'] = df['U']
    df['datetime'] = df.index
    df = df.reset_index()
    
    # Generate boundary conditions from omni dataframe
    time_omni, vcarr_omni, bcarr_omni = Hin.generate_vCarr_from_OMNI(runstart, runstop, omni_input=df)
    
    # Get the position of the Earth from JPL Horizons
    epoch_dict = {'start': runstart.strftime('%Y-%m-%d %H:%M:%S'), 
                  'stop': runstop.strftime('%Y-%m-%d %H:%M:%S'), 
                  'step': '{:1.0f}d'.format(np.diff(time_omni).mean())}
    earth_pos = Horizons(id = '399', location = '500@0', epochs = epoch_dict).vectors().to_pandas()
    
    # Map to 210 solar radii, then 21.5 solar radii
    vcarr_210 = vcarr_omni.copy()
    vcarr_21p5 = vcarr_omni.copy()
    for i, time in enumerate(time_omni):
        
        # Lookup the helicentric distance of the Earth
        Earth_r_AU = (earth_pos['range'].iloc[i] * u.AU)
        
        # Map to 210 solar radii
        vcarr_210[:,i] = Hin.map_v_boundary_inwards(vcarr_omni[:,i]*u.km/u.s, 
                                                    Earth_r_AU.to(u.solRad), 
                                                    210 * u.solRad)
        
        # And map to 21.5 solar radii
        vcarr_21p5[:,i] = Hin.map_v_boundary_inwards(vcarr_210[:,i]*u.km/u.s,
                                                     210 * u.solRad,
                                                     21.5 * u.solRad)
        
    return time_omni, vcarr_21p5, bcarr_omni

def create_CME_list(runstart, runstop):
    
    # Get the CMEs
    cmes = queryDONKI.CME(runstart, runstop)
    
    cmelist = []
    
    for index, row in cmes.iterrows():
        info = row['cmeAnalyses']
        t = (datetime.datetime.strptime(info['time21_5'], "%Y-%m-%dT%H:%MZ") - runstart).total_seconds()
        lon = info['longitude']
        lat = info['latitude']
        w = 2*info['halfAngle']
        v = info['speed']
        thick = 4
        if lon is not None:
            cme = H.ConeCME(t_launch=t*u.s, 
                            longitude=lon*u.deg, 
                            latitude=lat*u.deg, 
                            width=w*u.deg, 
                            v=v*(u.km/u.s), 
                            thickness=thick*u.solRad, 
                            initial_height=21.5*u.solRad)
            
            cmelist.append(cme)
            
    return cmelist
        
# Unmodified omni for comparison
import time

t_start = time.time()
time_21p5_control, vcarr_21p5_control, bcarr_21p5_control = map_omni_inwards(runstart, runstop, omni_df)

print('Time backmapping: ', time.time() - t_start) # About 20s
model = Hin.set_time_dependent_boundary(vcarr_21p5_control, time_21p5_control, 
                                        runstart, simtime = runtime.days*u.day, 
                                        r_min = 21.5 * u.solRad, 
                                        r_max = 2150 * u.solRad, 
                                        latitude=0*u.deg,
                                        bgrid_Carr = bcarr_21p5_control, 
                                        lon_start = (350 * u.deg).to(u.rad), 
                                        lon_stop =  (10* u.deg).to(u.rad),
                                        frame = 'sidereal')
print('Time initializing model: ', time.time() - t_start) # About 26s
model.solve([])
print('Time solving model: ', time.time() - t_start)

# # Now map each sample back
# time_samples, vcarr_samples, bcarr_samples = [], [], []
# for Fo in tqdm.tqdm(Fo_samples):
    
#     # Create a version of omni_df with the sample in place of U
#     omni_df_sample = omni_df.copy(deep=True)
#     omni_df_sample['U'] = Fo.ravel()
    
#     # Backmap
#     time, vcarr, bcarr = map_omni_inwards(runstart, runstop, omni_df_sample)
    
#     time_samples.append(time)
#     vcarr_samples.append(vcarr)
#     bcarr_samples.append(bcarr)
    
# time_samples = np.array(time_samples)
# vcarr_samples = np.array(vcarr_samples)
# bcarr_samples = np.array(bcarr_samples)

# # Get the list of relevant CMEs, which won't change across samples
# cmelist = create_CME_list(runstart, runstop)
    

