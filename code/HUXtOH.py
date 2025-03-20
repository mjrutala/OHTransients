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
import pandas as pd

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
runstart = datetime.datetime(2024, 4, 1)
runstop = datetime.datetime(2024, 6, 1)
runtime = (runstop - runstart)


def fetch_omni(starttime, stoptime):
    """
    A function to grab and process the most up-to-date OMNI data from 
    COHOWeb directly
    
    Args:
        starttime : datetime for start of requested interval
        endtime : datetime for start of requested interval

    Returns:
        omni: Dataframe of the OMNI timeseries

    """
    # trange = attrs.Time(starttime, endtime)
    # dataset = attrs.cdaweb.Dataset('OMNI_COHO1HR_MERGED_MAG_PLASMA')
    # result = Fido.search(trange, dataset)
    # downloaded_files = Fido.fetch(result, path='../Data/OMNI/')
    
    # omni = TimeSeries(downloaded_files, concatenate=True).to_dataframe()
    # omni = omni.rename(columns = {'ABS_B': 'B',
    #                               'V': 'U'})
    # # Set invalid data points to NaN
    # id_bad = omni['U'] == 9999.0
    # omni.loc[id_bad, 'U'] = np.NaN

    
    # Get the relevant URLs
    skeleton_url = 'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni_m{YYYY:4.0f}.dat'
    years_covered = np.arange(starttime.year, stoptime.year+1, 1)
    all_urls = [skeleton_url.format(YYYY=year) for year in years_covered]
    
    # Read each OMNI file, concatenate, and set NaNs 
    null_values = {'year': None, 'doy': None, 'hour': None,
                   'lat_hgi': 9999.9, 'lon_hgi': 9999.9, 
                   'BR': 999.9, 'BT': 999.9, 'BN': 999.9, 'B': 999.9, 
                   'U': 9999., 'U_theta': 999.9, 'U_phi': 999.9,
                   'n': 999.9, 'T': 9999999.}
    df_list = []
    for url in all_urls:
        df = pd.read_csv(url, header=0, sep='\s+', names=null_values.keys())
        df_list.append(df)
    df = pd.concat(df_list, axis='index') 
    df = df.reset_index().sort_index()
    for col, null_val in null_values.items():
        null_index = df[col] == null_val
        df.loc[null_index, col] = np.nan
    
    # Add datetime, MJD
    parse_dt = datetime.datetime.strptime
    date_cols = ['year', 'doy', 'hour']
    str_fmt   = '{:04.0f}-{:03.0f} {:02.0f}'
    dt_fmt    = '%Y-%j %H'
    df['dt'] = [parse_dt(str_fmt.format(*row[date_cols]), dt_fmt) 
                      for _, row in df.iterrows()]
    df['mjd'] = Time(df['dt']).mjd

    # create a BX_GSE field that is expected by some HUXt fucntions
    df['BX_GSE'] = -df['BR']
    
    # Limit the dataframe to the requested dates
    omni = df.query('@starttime <= dt < @stoptime')
    omni = omni.rename(columns={'dt': 'datetime'})
    omni.reset_index()
    
    return omni

# %% Get raw OMNI data ========================================================
# =============================================================================
# !!!! Padding should relate to propagation time to Jupiter/Saturn
padding = datetime.timedelta(days=27)
runstart_padded = runstart - padding
runstop_padded = runstop + padding
omni_df = fetch_omni(runstart_padded, runstop_padded)

# %% Get DONKI ICMEs @ OMNI ===================================================
# Assume a generous ICME duration
# =============================================================================
icme_df = queryDONKI.ICME(runstart_padded, runstop_padded, location='Earth', duration=2.0*u.day)

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
n_samples = 10

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
n_clusters = int((runtime.total_seconds()/3600) / 6)
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
# !!!! Calculate noise_variance from clustered data
model = gpflow.models.GPR((X_clustered, Y_clustered), kernel=signal_kernel, noise_variance=0.3)

opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

# gpflow.utilities.print_summary(model)

# Xp = np.arange(qomni_df['mjd'].iloc[0], qomni_df['mjd'].iloc[-1]+30, 1)[:, None]
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
ax.scatter(omni_df['mjd'], omni_df['U'], color='C0', marker='.', s=2, zorder=1,
           label = 'OMNI Data')
ax.scatter(X, Y, color='black', marker='.', s=2, zorder=2,
           label="OMNI Data - ICMEs")
# ax.scatter(X_clustered, Y_clustered, label='Inducing Points', color='C0', marker='o', s=6, zorder=4)

ax.plot(Xo, Yo_mu, label="Mean prediction", color='C1', zorder=0)
ax.fill_between(
    Xo.ravel(),
    (Yo_mu - 1.96 * Yo_sig).ravel(),
    (Yo_mu + 1.96 * Yo_sig).ravel(),
    alpha=0.5, color='C1',
    label=r"95% confidence interval", zorder=-2)

for Fo in Fo_samples:
    ax.plot(Xo.ravel(), Fo.ravel(), lw=1, color='C4', alpha=0.1, zorder=-1)
ax.plot(Xo.ravel()[0:1], Fo.ravel()[0:1], lw=1, color='C4', alpha=1, 
        label = 'Samples about Mean')

ax.legend(scatterpoints=3)
ax.set(xlabel='Date [MJD]', ylabel='Solar Wind Speed [km/s]', 
       title='OMNI (@ 1AU), no ICMEs, GP Data Imputation')
# ax.set(xlim=[60425, 60450])


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
    breakpoint()
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
        breakpoint()
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



def HUXt_at(body, runstart, runstop, time_grid, vgrid_Carr, **kwargs):
    """
    Run HUXt for a celestial body or spacecraft, computing the correct
    longitudes to minimize runtime

    Parameters
    ----------
    body : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    body_id_dict = {'mercury': '199', 'venus': '299', 'earth': '399', 'mars': '499',
                    'jupiter': '599', 'saturn': '699', 'uranus': '799', 'neptune': '899',
                    'parker solar probe': '2018-065A', 'solar orbiter': '2020-010A',
                    'stereo-a': '2006-047A', 'stereo-b': '2006-047B',
                    'juno': '2011-040A'}
    if body.lower() in body_id_dict.keys():
        body_id = body_id_dict[body.lower()]
    else:
        print("This body is not currently supported. Supported bodies include:")
        print(list(body_id_dict.keys()))
    
    # Get the position of the Earth from JPL Horizons
    epoch_dict = {'start': runstart.strftime('%Y-%m-%d %H:%M:%S'), 
                  'stop': runstop.strftime('%Y-%m-%d %H:%M:%S'), 
                  'step': '{:1.0f}d'.format(np.diff(time_grid).mean())}
    
    earth_pos = Horizons(id = '399', location = '@SSB', epochs = epoch_dict).vectors().to_pandas()
    earth_pos['phi'] = np.arctan2(earth_pos['y'], earth_pos['x']) + np.pi 
    earth_pos['lambda'] = np.arctan2(earth_pos['z'], np.sqrt(earth_pos['x']**2 + earth_pos['y']**2)) 
    
    body_pos = Horizons(id = body_id, location='@SSB', epochs = epoch_dict).vectors().to_pandas()
    body_pos['phi'] = np.arctan2(body_pos['y'], body_pos['x']) + np.pi 
    body_pos['lambda'] = np.arctan2(body_pos['z'], np.sqrt(body_pos['x']**2 + body_pos['y']**2))
    
    delta_phi = body_pos['phi'] - earth_pos['phi'][0]
    start_lon = ((delta_phi.iloc[0] + 2*np.pi) % (2*np.pi)) * u.rad
    stop_lon = ((delta_phi.iloc[-1] + 2*np.pi) % (2*np.pi)) * u.rad
    
    model = Hin.set_time_dependent_boundary(vgrid_Carr, time_grid, 
                                            runstart, simtime = (runstop - runstart).days*u.day, 
                                            lon_start = start_lon, 
                                            lon_stop =  stop_lon,
                                            frame='sidereal',
                                            **kwargs)
    
    return model



# Get the list of relevant CMEs, which won't change across samples
cmelist = create_CME_list(runstart_padded, runstop_padded)



model = HUXt_at('mars', runstart, runstop, 
                time_grid = time_21p5_control, 
                vgrid_Carr = vcarr_21p5_control, 
                bgrid_Carr = bcarr_21p5_control, 
                r_min = 21.5 * u.solRad, 
                r_max = 2150 * u.solRad, 
                latitude=0*u.deg)

model.solve([])
HA.animate(model, 'mars_nocme')

# Now map each sample back
time_samples, vcarr_samples, bcarr_samples = [], [], []
for Fo in tqdm.tqdm(Fo_samples):
    
    # Create a version of omni_df with the sample in place of U
    omni_df_sample = omni_df.copy(deep=True)
    omni_df_sample['U'] = Fo.ravel()
    
    # Backmap
    time, vcarr, bcarr = map_omni_inwards(runstart, runstop, omni_df_sample)
    
    time_samples.append(time)
    vcarr_samples.append(vcarr)
    bcarr_samples.append(bcarr)
    
    
model = HUXt_at('mars', runstart, runstop, 
                time_grid = time_samples[0], 
                vgrid_Carr = vcarr_samples[0], 
                bgrid_Carr = bcarr_samples[0], 
                r_min = 21.5 * u.solRad, 
                r_max = 2150 * u.solRad, 
                latitude=0*u.deg)

model.solve([cmelist])
HA.animate(model, 'mars_cme_full')

# # running for a single longitude takes ~28s
# # running for 20 degrees longitude takes ~37s
# # model_spans = [0,  10, 20, 30, 40, 50, 60, 70, 80, 90]
# # model_takes = [28, 32, 37, 40, 46, 53, 59, 61, 67, 74]
model = Hin.set_time_dependent_boundary(vcarr_samples[0], time_samples[0], 
                                        runstart, simtime = runtime.days*u.day, 
                                        r_min = 21.5 * u.solRad, 
                                        r_max = 2150 * u.solRad, 
                                        latitude=0*u.deg,
                                        bgrid_Carr = bcarr_samples[0], 
                                        lon_start = 0 * u.rad, 
                                        lon_stop =  360 * u.rad,
                                        # lon_out = (0 * u.deg).to(u.rad),
                                        frame='sidereal') # frame = 'sidereal')
# print('Time initializing model: ', time.time() - t_start) # About 26s
# model.solve([])
# print('Time solving model: ', time.time() - t_start)


    
# time_samples = np.array(time_samples)
# vcarr_samples = np.array(vcarr_samples)
# bcarr_samples = np.array(bcarr_samples)

# # Get the list of relevant CMEs, which won't change across samples
# cmelist = create_CME_list(runstart_padded, runstop_padded)
    

