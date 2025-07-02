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
import time

from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries
from sunpy.coordinates import sun
from astropy.time import Time

sys.path.append('/Users/mrutala/projects/HUXt/code/')
sys.path.append('/Users/mrutala/projects/OHTransients/code/')
# os.environ['PYTHONPATH'] = '/Users/mrutala/projects/OHTransients/HUXt/code/'
# os.environ['PYTHONPATH'] = '/Users/mrutala/projects/OHTransients/code/'
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
from scipy import ndimage
from scipy import stats

from astroquery.jplhorizons import Horizons

# import huxt_inputs_wsa as Hin_wsa
import queryDONKI

try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

# %% Data Search ==============================================================
# =============================================================================
def lookup_omni(starttime, stoptime):
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
    df = df.reset_index(drop=True).sort_index()
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
    omni = omni.reset_index(drop=True)
    
    return omni

def lookup_PSPData(starttime, stoptime):
    """
    A function to grab and process the most up-to-date PSP data from 
    COHOWeb directly
    
    Args:
        starttime : datetime for start of requested interval
        endtime : datetime for start of requested interval

    Returns:
        

    """

    # Get the relevant URLs
    skeleton_url = 'https://spdf.gsfc.nasa.gov/pub/data/psp/coho1hr_magplasma/ascii/psp_merged_hr{YYYY}.txt'
    years_covered = np.arange(starttime.year, stoptime.year+1, 1)
    all_urls = [skeleton_url.format(YYYY=year) for year in years_covered]
    
    # Read each OMNI file, concatenate, and set NaNs 
    null_values = {'year': None, 'doy': None, 'hour': None,
                   'r_hgi': 999.99, 'lat_hgi': 9999.9, 'lon_hgi': 9999.9,
                   'BR': 99999.99, 'BT': 99999.99, 'BN': 99999.99, 'B': 99999.99, 
                   'UR': 99999.9, 'UT': 99999.9, 'UN': 99999.9, 'U': 99999.9, 'U_theta': 99999.9, 'U_phi': 99999.9,
                   'n': 9999.9, 'T': 99999999.}
    df_list = []
    for url in all_urls:
        # Fail gracefully
        try:
            df = pd.read_csv(url, header=0, sep='\s+', names=null_values.keys())
        except:
            print("File {} is not available. Skipping...".format(skeleton_url))
        df_list.append(df)
    df = pd.concat(df_list, axis='index') 
    df = df.reset_index(drop=True).sort_index()
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
    df = df.query('@starttime <= dt < @stoptime')
    df = df.rename(columns={'dt': 'datetime'})
    df = df.reset_index(drop=True)
    
    return df


# %% Get raw OMNI data ========================================================
# =============================================================================
# !!!! Padding should relate to propagation time to Jupiter/Saturn

# omni_df = lookup_omni(runstart_padded, runstop_padded)

# # %% Get DONKI ICMEs @ OMNI ===================================================
# # Assume a generous ICME duration
# # =============================================================================
# icme_df = queryDONKI.ICME(runstart_padded, runstop_padded, location='Earth', duration=2.0*u.day)

# # %% OMNI - ICMEs =============================================================
# # 
# # =============================================================================
# # Reformat icme_df for subtraction
# _icme_df = icme_df.rename(columns = {'eventTime': 'Shock_time'})
# _icme_df['ICME_end'] = [row['Shock_time'] + datetime.timedelta(days=(row['duration'])) for _, row in _icme_df.iterrows()]

# qomni_df = Hin.remove_ICMEs(omni_df, _icme_df, interpolate=False, icme_buffer=0.5 * u.day, interp_buffer=1 * u.day,
#                             params=['U', 'BX_GSE'], fill_vals=None)

# qomni_df['carringtonLongitude'] = sun.L0(qomni_df['datetime']).to(u.deg).value

# %% Gaussian Process Data Imputation =========================================
# =============================================================================
def make_BackgroundSolarWind_withGP(start, stop, n_samples, omni = None, icmes = None):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # from scipy.spatial.distance import cdist
    # import tensorflow_probability  as     tfp
    import gpflow
    from sklearn.cluster import KMeans
    
    # These values are currently FIXED
    # They may (should) be made adjustable
    icme_duration = 2.0 * u.day # Seems reasonable from C&R ICME List
    icme_buffer = 0.1 * u.day
    interp_buffer = 1 * u.day
    
    max_mjd_scale = 100 # mjd will be rescaled between [0, max_mjd_scale]
    subsetSize = 1000
    min_l      = 0.0 
    mid_l      = 0.05
    max_l      = 0.2
    init_var   = 3
    period = 27 * u.day
    # n_samples = 9
    average_cluster_span = 6 * u.hour # For K-means clustering to reduce data 
    
    # Calculate the span from stop - start
    span = stop - start
    
    # If no omni dataset is supplied, look one up
    if omni is None:
        omni = lookup_omni(start, stop)
    
    # If no icmes are supplied, look them up
    if icmes is None:
        icmes = queryDONKI.ICME(start, stop, location = 'Earth', duration = icme_duration)
    
    # Remove ICMEs from OMNI data, leaving NaNs behind
    if 'eventTime' in icmes.columns: 
        icmes = icmes.rename(columns = {'eventTime': 'Shock_time'})
        icmes['ICME_end'] = [row['Shock_time'] + datetime.timedelta(days=(row['duration'])) for _, row in icmes.iterrows()]
    
    omni_noicme = Hin.remove_ICMEs(omni, icmes, 
                                   params=['U', 'BX_GSE'], 
                                   interpolate = False, 
                                   icme_buffer = icme_buffer, 
                                   interp_buffer = interp_buffer, 
                                   fill_vals = np.nan)
    
    # Get the mjd and U as column vectors for GPflow
    mjd = omni_noicme.dropna(axis='index', how='any')['mjd'].to_numpy()[:, None]
    speed = omni_noicme.dropna(axis='index', how='any')['U'].to_numpy('float64')[:, None]

    # Scale the MJD (abscissa) and U (ordinate) for GP imputation
    mjd_rescaler = MinMaxScaler((0, max_mjd_scale))
    mjd_rescaler.fit(mjd)
    X = mjd_rescaler.transform(mjd)

    speed_rescaler = StandardScaler()
    speed_rescaler.fit(speed)
    Y = speed_rescaler.transform(speed)

    # K-means cluster the data (fewer data = faster processing)
    # And calculate the variance within each cluster
    XY = np.array(list(zip(X.flatten(), Y.flatten())))
    n_clusters = int((span.total_seconds()/3600) * u.hour / average_cluster_span)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(XY)
    XYc = kmeans.cluster_centers_
    Yc_var = np.array([np.var(XY[kmeans.labels_ == i, 1]) 
                       for i in range(kmeans.n_clusters)])

    # Arrange clusters to be strictly increasing in time (X)
    Xc, Yc = XYc.T[0], XYc.T[1]
    cluster_sort = np.argsort(Xc)
    Xc = Xc[cluster_sort][:, None]
    Yc = Yc[cluster_sort][:, None]
    Yc_var = Yc_var[cluster_sort][:, None]

    # Construct the signal kernel for GP
    # Again, these lengthscales could probable be calculated?
    small_scale_kernel = gpflow.kernels.SquaredExponential(variance=2**2, lengthscales=1.0)
    large_scale_kernel = gpflow.kernels.SquaredExponential(variance=2**2, lengthscales=10.0)
    irregularities_kernel = gpflow.kernels.SquaredExponential(variance=1**2, lengthscales=1.0)
    # Fixed period Carrington rotation kernel
    period_rescaled = max_mjd_scale / (span.days * u.day / period).value
    carrington_kernel = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), 
                                                period=gpflow.Parameter(period_rescaled, trainable=False))

    signal_kernel = small_scale_kernel + large_scale_kernel + irregularities_kernel + carrington_kernel
    
    # This model *could* be solved with the exact noise for each point...
    # But this would require defining a custom likelihood class...
    model = gpflow.models.GPR((Xc, Yc), kernel=signal_kernel, noise_variance=np.mean(Yc_var))
    
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    
    # gpflow.utilities.print_summary(model)
    
    # breakpoint()
    
    # Xp = np.arange(qomni_df['mjd'].iloc[0], qomni_df['mjd'].iloc[-1]+30, 1)[:, None]
    Xo = omni['mjd'].to_numpy('float64')[:, None]
    Xoc = mjd_rescaler.transform(Xo)
    
    Yoc_mu, Yoc_var = model.predict_y(Xoc)
    Yoc_mu, Yoc_var = np.array(Yoc_mu), np.array(Yoc_var)
    Yoc_sig = np.sqrt(Yoc_var)
    
    foc = model.predict_f_samples(Xoc, n_samples, full_cov=False)
    foc = np.array(foc)
    
    Xo = mjd_rescaler.inverse_transform(Xoc)
    Yo_mu = speed_rescaler.inverse_transform(Yoc_mu)
    Yo_sig = Yoc_sig * speed_rescaler.scale_
    fo_samples = np.array([speed_rescaler.inverse_transform(f) for f in foc])
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(omni['mjd'], omni['U'], color='C0', marker='.', s=2, zorder=1,
               label = 'OMNI Data')
    ax.scatter(omni_noicme['mjd'], omni_noicme['U'], color='black', marker='.', s=2, zorder=2,
               label="OMNI Data - ICMEs")
    # ax.scatter(Xc, Yc, label='Inducing Points', color='C1', marker='o', s=6, zorder=4)
    
    ax.plot(Xo, Yo_mu, label="Mean prediction", color='C5', zorder=0)
    ax.fill_between(
        Xo.ravel(),
        (Yo_mu - 1.96 * Yo_sig).ravel(),
        (Yo_mu + 1.96 * Yo_sig).ravel(),
        alpha=0.5, color='C1',
        label=r"95% confidence interval", zorder=-2)
    
    for fo_sample in fo_samples:
        ax.plot(Xo.ravel(), fo_sample.ravel(), lw=1, color='C3', alpha=0.05, zorder=-1)
    ax.plot(Xo.ravel()[0:1], fo_sample.ravel()[0:1], lw=1, color='C3', alpha=1, 
            label = 'Samples about Mean')
    
    ax.legend(scatterpoints=3)
    ax.set(xlabel='Date [MJD]', ylabel='Solar Wind Speed [km/s]', 
           title='OMNI (@ 1AU), no ICMEs, GP Data Imputation')
    # ax.set(xlim=[60425, 60450])
    
    # Add samples to omni, return as a list
    results = []
    for fo_sample in fo_samples:
        new_omni = omni_noicme.copy(deep=True)
        new_omni['U'] = fo_sample
        results.append(new_omni)
    
    return results

# %% Get CMEs =================================================================
# This only needs to be done once for the ensemble
# =============================================================================
def create_CME_list(runstart, runstop, latitude_range=[-90,90]):
    
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
        
        if (lon is not None) & (lat > latitude_range[0]) & (lat < latitude_range[1]):
            cme = H.ConeCME(t_launch=t*u.s, 
                            longitude=lon*u.deg, 
                            latitude=lat*u.deg, 
                            width=w*u.deg, 
                            v=v*(u.km/u.s), 
                            thickness=thick*u.solRad, 
                            initial_height=21.5*u.solRad)
            
            cmelist.append(cme)
            
    return cmelist

def perturb_CME_list(cme_list, n):
    import copy
    
    # Baby steps: just perturb the launch time and velocity
    scales = {'tla': 3600,
              'lon': 20,
              'lat': 10,
              'wid': 10,
              'vel': 50,
              'thi': 1}
    
    rng = np.random.default_rng()
    
    perturbed_cme_lists = []
    for cme in cme_list:
        t_launch_0 = cme.t_launch
        v_0 = cme.v
        
        t_launch_perturbed = rng.normal(t_launch_0.value, 12*3600., n) * t_launch_0.unit
        v_perturbed = rng.normal(v_0.value, 100, n) * v_0.unit
        
        new_cmes = []
        for i in range(n):
            new_cme = copy.deepcopy(cme)
            new_cme.t_lauch = t_launch_perturbed[i]
            new_cme.v = v_perturbed[i]
            new_cmes.append(new_cme)
            
        perturbed_cme_lists.append(new_cmes)
    
    perturbed_cme_lists = [list(l) for l in zip(*perturbed_cme_lists)]
    
    return perturbed_cme_lists
    

# %% Backmap samples to 21.5 Rs ===============================================
# =============================================================================

def map_omni_inwards(runstart, runstop, df, earth_pos = None):
    
    # Format the OMNI DataFrame as HUXt expects it
    df['V'] = df['U']
    # df['datetime'] = df.index
    # df = df.reset_index()
    
    # Generate boundary conditions from omni dataframe
    time_omni, vcarr_omni, bcarr_omni = Hin.generate_vCarr_from_OMNI(runstart, runstop, omni_input=df)
    
    # Get the position of the Earth from JPL Horizons
    if earth_pos is None:
        epoch_dict = {'start': runstart.strftime('%Y-%m-%d %H:%M:%S'), 
                      'stop': runstop.strftime('%Y-%m-%d %H:%M:%S'), 
                      'step': '{:1.0f}d'.format(np.diff(time_omni).mean())}
        earth_pos = Horizons(id = '399', location = '500@0', epochs = epoch_dict).vectors().to_pandas()
    
    # Map to 210 solar radii, then 21.5 solar radii
    vcarr_210 = vcarr_omni.copy()
    vcarr_21p5 = vcarr_omni.copy()
    for i, t in enumerate(time_omni):
        # Map to 210 solar radii
        vcarr_210[:,i] = Hin.map_v_boundary_inwards(vcarr_omni[:,i]*u.km/u.s, 
                                                    earth_pos.r[i], 
                                                    210 * u.solRad)
        
        # And map to 21.5 solar radii
        vcarr_21p5[:,i] = Hin.map_v_boundary_inwards(vcarr_210[:,i]*u.km/u.s,
                                                     210 * u.solRad,
                                                     21.5 * u.solRad)
        
    return time_omni, vcarr_21p5, bcarr_omni

def HUXt_at(body, runstart, runstop, time_grid, vgrid_Carr, dpadding=0.03, target_pos=None, **kwargs):
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
    
    if target_pos is None:
        target_pos = H.Observer(body, Time(time_grid, format='mjd'))
    
    # HEEQ longitude, continuous
    delta = np.unwrap(target_pos.lon.value)
    if (delta > 2*np.pi).any():
        delta -= 2*np.pi
    
    # Max and Min of delta
    dmin, dmax = np.min(delta), np.max(delta)
    if ((delta >= dmin) & (delta <= dmax)).all():
        start_lon = (dmin - dpadding) % (2*np.pi)
        stop_lon = (dmax + dpadding) % (2*np.pi)
    elif ((delta >= dmax) | (delta <= dmin)).all():
        start_lon = (dmax - dpadding) % (2*np.pi)
        stop_lon = (dmin + dpadding) % (2*np.pi)
    else:
        print("Something fishy is afoot...")
        breakpoint()
    
    # # High res time grid, check that all points lie inside bounds...
    # time_grid_hires = np.arange(time_grid[0], time_grid[-1], 0.01)
    # body_pos_hires = H.Observer(body, Time(time_grid_hires, format='mjd'))
    
    model = Hin.set_time_dependent_boundary(vgrid_Carr, time_grid, 
                                            runstart, simtime = (runstop - runstart).days*u.day, 
                                            lon_start = start_lon * u.rad, 
                                            lon_stop =  stop_lon * u.rad,
                                            frame='synodic',
                                            **kwargs)
   
    return model

def HUXt_at_(body, runstart, runstop, time_grid, vgrid_Carr, dpadding=0.03, target_pos=None, **kwargs):
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
    
    if target_pos is None:
        target_pos = H.Observer(body, Time(time_grid, format='mjd'))
    
    # HEEQ longitude, continuous
    delta = np.unwrap(target_pos.lon.value)
    if (delta > 2*np.pi).any():
        delta -= 2*np.pi
    
    # Max and Min of delta
    dmin, dmax = np.min(delta), np.max(delta)
    if ((delta >= dmin) & (delta <= dmax)).all():
        start_lon = (dmin - dpadding) % (2*np.pi)
        stop_lon = (dmax + dpadding) % (2*np.pi)
    elif ((delta >= dmax) | (delta <= dmin)).all():
        start_lon = (dmax - dpadding) % (2*np.pi)
        stop_lon = (dmin + dpadding) % (2*np.pi)
    else:
        print("Something fishy is afoot...")
        breakpoint()
    
    # # High res time grid, check that all points lie inside bounds...
    # time_grid_hires = np.arange(time_grid[0], time_grid[-1], 0.01)
    # body_pos_hires = H.Observer(body, Time(time_grid_hires, format='mjd'))
    
    model = Hin.set_time_dependent_boundary(vgrid_Carr, time_grid, 
                                            runstart, simtime = (runstop - runstart).days*u.day, 
                                            lon_start = start_lon * u.rad, 
                                            lon_stop =  stop_lon * u.rad,
                                            frame='synodic',
                                            **kwargs)
   
    return model

# # Now map each sample back
# time_samples, vcarr_samples, bcarr_samples = [], [], []
# results = []

# # Add speed samples to omni_df in a list
# omni_samples = []
# for i in range(n_samples):
    
#     Fo = Fo_samples[i]
#     omni_df_sample = omni_df.copy(deep=True)
#     omni_df_sample['U'] = Fo.ravel()
#     omni_samples.append(omni_df_sample)
    


def solve_single_model(_runstart, _runstop, _omni_df, _cme_list, _earth_pos, _target_pos):
    
    # Backmap
    time, vcarr, bcarr = map_omni_inwards(_runstart, _runstop, _omni_df, _earth_pos)
    
    # Propagate to PSP
    model = HUXt_at('parker solar probe', _runstart, _runstop, 
                    time_grid = time, 
                    vgrid_Carr = vcarr, 
                    bgrid_Carr = bcarr, 
                    target_pos = _target_pos,
                    r_min = 21.5 * u.solRad, 
                    r_max = 215 * u.solRad, 
                    latitude=0*u.deg)
    
    model.solve(_cme_list)
    
    return model


def solve_model_attarget(start, stop, tgrid, vgrid_carr, bgrid_carr, 
                         target = None, cme_list = [], dpadding = 0.03, **kwargs):
    from scipy.interpolate import RegularGridInterpolator
    from tqdm import tqdm
    
    # Parse the target
    if target is None:
        target = 'EARTH'
    # elif type(target) == str:
    target_obs = H.Observer(target, Time(tgrid, format='mjd'))
    # else:
    #     target_obs = target
        
    # HEEQ longitude, continuous
    delta = np.unwrap(target_obs.lon.value)
    if (delta > 2*np.pi).any():
        delta -= 2*np.pi
    
    # Max and Min of delta
    dmin, dmax = np.min(delta), np.max(delta)
    if ((delta >= dmin) & (delta <= dmax)).all():
        start_lon = (dmin - dpadding) % (2*np.pi)
        stop_lon = (dmax + dpadding) % (2*np.pi)
    elif ((delta >= dmax) | (delta <= dmin)).all():
        start_lon = (dmax - dpadding) % (2*np.pi)
        stop_lon = (dmin + dpadding) % (2*np.pi)
    else:
        print("Something fishy is afoot...")
        breakpoint()
        
    # # High res time grid, check that all points lie inside bounds...
    # time_grid_hires = np.arange(time_grid[0], time_grid[-1], 0.01)
    # body_pos_hires = H.Observer(body, Time(time_grid_hires, format='mjd'))
    
    model = Hin.set_time_dependent_boundary(vgrid_carr, tgrid, 
                                            start, simtime = (stop - start).days*u.day, 
                                            bgrid_Carr = bgrid_carr,
                                            lon_start = start_lon * u.rad, 
                                            lon_stop =  stop_lon * u.rad,
                                            r_max = np.max(target_obs.r.to(u.solRad)),
                                            frame='synodic',
                                            **kwargs)
    
    model.solve(cme_list)
    
    # Sample the model at the target
    time_series = HA.get_observer_timeseries(model, target)
    
    cme_arrivals = []
    for cme in model.cmes:
        cme_stats = cme.compute_arrival_at_body(target)
        if cme_stats['t_arrive'].jyear > 1000:
            datestr = cme_stats['t_arrive'].isot
            cme_arrivals.append(datetime.datetime.fromisoformat(datestr))
    
    # # This produces the ~same results as HA.get_observer_timeseries
    # t_model = (model.time_init + model.time_out).mjd
    # r_model = model.r_grid[:,0].to(u.solRad).value
    # l_model = model.lon_grid[0,:].to(u.rad).value
    
    # interp_v = RegularGridInterpolator((t_model, r_model, l_model), model.v_grid, bounds_error=False, fill_value=np.nan)
    # interp_b = RegularGridInterpolator((t_model, r_model, l_model), model.b_grid, bounds_error=False, fill_value=np.nan)
    # del model # Free memory
    
    # trl_sample = [[t, r, l] for t, r, l in zip(target_obs.time.mjd, target_obs.r.value, target_obs.lon.value)]
    # v_sample = interp_v(trl_sample)
    # b_sample = interp_b(trl_sample)

    # output = pd.DataFrame(data={'time': target_obs.time, 
    #                             'r': target_obs.r.value, 
    #                             'lon': target_obs.lon.value, 
    #                             'vsw': v_sample, 
    #                             'bpol': b_sample, 
    #                             'mjd': target_obs.time.mjd})
    
    return time_series

def solve_model_attarget_dask(start, stop, tgrids, vgrids_carr, bgrids_carr, 
                              target, cme_lists = [], dpadding = 0.03, **kwargs):
    from dask.distributed import Client

    client = Client(n_workers=4,
                    threads_per_worker=1,
                    )

    futures = []
    for tg, vg, bg, cmes in zip(tgrids, vgrids_carr, bgrids_carr, cme_lists):
        future = client.submit(solve_model_attarget, start, stop, tg, vg, bg, target, cmes, dpadding, **kwargs)
        futures.append(future)

    breakpoint()
    
    return 



def solve_model_attarget_parallel(start, stop, tgrids, vgrids_carr, bgrids_carr, 
                                  target = None, cme_lists = [], dpadding = 0.03, **kwargs):
    import multiprocessing as mp
    
    def solve_model_attarget_star(args):
        return solve_model_attarget(*args)
    
    arg_list = [(start, stop, tgrid, vgrid_carr, bgrid_carr, target, cme_list, kwargs) 
                for tgrid, vgrid_carr, bgrid_carr, cme_list in zip(tgrids, vgrids_carr, bgrids_carr, cme_lists)]
    
    n_cpus = int(0.5 * mp.cpu_count())
    results = []
    with mp.Pool(n_cpus) as pool:
        generator = pool.imap(solve_model_attarget_star, arg_list)

        for element in tqdm(generator, total=len(arg_list)):
            results.append(element)
            
    return results
    

# test = solve_single_model(runstart, runstop, omni_samples[0], perturbed_cmelist[0], earth_pos, target_pos)
# %% Single processing



# single_result = solve_single_model(runstart_padded, runstop_padded, omni_samples[0], cme_samples[0], earth_pos, target_pos)


# t_start = timer.time()
# results = []
# for i in range(len(omni_samples)):
#     print(i)
#     r = solve_single_model(runstart_padded, runstop_padded, omni_samples[i], cme_samples[i], earth_pos, target_pos)
#     results.append(r)
# print(timer.time() - t_start)

# psp = lookup_PSPData(runstart, runstop)
# samples_psp = []
# for r in results:
#     sample = HA.get_observer_timeseries(r, 'PSP')
#     samples_psp.append(sample)
    
# fig, ax = plt.subplots()
# ax.scatter(psp['mjd'], psp['U'], color='black', marker='.', s=2, zorder=1,
#            label = 'PSP Data')

# mean_U = np.zeros(len(samples_psp[0]['vsw']))
# for sample in samples_psp:
#     ax.plot(sample['mjd'], sample['vsw'], color='C4', alpha=0.2, lw=1, zorder=-1)
#     mean_U += sample['vsw']
# ax.plot(sample['mjd'][0:1], sample['vsw'][0:1], color='C4', alpha=1, lw=1, label='bOMNI-HUXt Samples')
# ax.plot(sample['mjd'], mean_U/n_samples, color='C5', alpha=1, lw=1, zorder=0, label='Mean bOMNI-HUXt')

# ax.legend(scatterpoints=3)
# ax.set(xlabel='Date [MJD]', ylabel='Solar Wind Speed [km/s]', 
#        title='Ensemble Forecast for PSP, w/ Variable Background and CMEs')
# ax.set(xlim = [psp['mjd'].iloc[0], psp['mjd'].iloc[-1]])

# %% Multiprocess?
# args = [(a, b, c, d, e, f) for a, b, c, d, e, f in zip([runstart]*n_samples, 
#                                                        [runstop]*n_samples,
#                                                        omni_samples,
#                                                        perturbed_cmelist,
#                                                        [earth_pos]*n_samples,
#                                                        [target_pos]*n_samples)]

# import multiprocessing as mp

# def solve_single_model_star(arg):
#     return solve_single_model(*arg)

# n_cores = mp.cpu_count()-1
# results = []
# with mp.Pool(n_cores) as pool:
#     generator = pool.imap(solve_single_model_star, args)

#     for element in tqdm.tqdm(generator, total=len(args)):
#         results.append(element)

# # %% Multiprocess with Ray
# import ray
# import time
# import huxt
# import huxt_inputs
# # os.environ['PYTHONPATH'] = '/Users/mrutala/projects/OHTransients/HUXt/code/'
# # os.environ['PYTHONPATH'] = '/Users/mrutala/projects/OHTransients/code/'
# # import huxt as H
# # import huxt_inputs as Hin

# runtime_env = {"working_dir": "/Users/mrutala/projects/OHTransients/HUXt/code/",
#                "py_modules": ["/Users/mrutala/projects/OHTransients/HUXt/code/huxt.py",
#                               "/Users/mrutala/projects/OHTransients/HUXt/code/huxt_inputs.py"]}
# ray.init(runtime_env = runtime_env)

# # ray.init(num_cpus=4)
# @ray.remote
# def backmap_withray(start, stop, omni_sample, earth_obs):
    
#     # Format the OMNI DataFrame as HUXt expects it
#     omni_sample['V'] = omni_sample['U']
    
#     # Generate boundary conditions from omni dataframe
#     time_omni, vcarr_omni, bcarr_omni = huxt_inputs.generate_vCarr_from_OMNI(start,stop, omni_input=omni_sample)
    
#     # Map to 210 solar radii, then 21.5 solar radii
#     vcarr_210 = vcarr_omni.copy()
#     vcarr_21p5 = vcarr_omni.copy()
#     for i, t in enumerate(time_omni):
#         # Map to 210 solar radii
#         vcarr_210[:,i] = Hin.map_v_boundary_inwards(vcarr_omni[:,i]*u.km/u.s, 
#                                                     earth_obs.r[i], 
#                                                     210 * u.solRad)
        
#         # And map to 21.5 solar radii
#         vcarr_21p5[:,i] = Hin.map_v_boundary_inwards(vcarr_210[:,i]*u.km/u.s,
#                                                      210 * u.solRad,
#                                                      21.5 * u.solRad)
    
#     return time_omni, vcarr_21p5, bcarr_omni

# futures = [backmap_withray.remote(runstart_padded, runstop_padded, sample, earth_pos) for sample in omni_samples]

# t_start = time.time()
# results = ray.get(futures)
# print(time.time() - t_start)

# # future = backmap_and_solve.remote(start = runstart_padded, 
# #                                   stop = runstop_padded,
# #                                   omni_samples = omni_samples,
# #                                   cme_samples = cme_samples,
# #                                   earth_state = earth_pos,
# #                                   target_state = target_pos)

# # results = ray.get(future)

#%% Multiprocess with Dask: Setup
import dask
import time

def print_url(filename, url):
    with open(filename, 'w') as f:
        print('<html>', file=f)
        print(' <body>', file=f)
        print('  <script type="text/javascript">', file=f)
        print('   window.location.href = "{}"'.format(url), file=f)
        print('  </script>', file=f)
        print(' </body>', file=f)
        print('</html>', file=f)
    return

@dask.delayed(nout=3)
def _backmap(start, stop, omni_sample, earth_pos):
    import sys
    sys.path.append('/Users/mrutala/projects/OHTransients/HUXt/code/')
    import huxt_inputs as Hin
    
    # Format the OMNI DataFrame as HUXt expects it
    omni_sample['V'] = omni_sample['U']
    
    # Generate boundary conditions from omni dataframe
    time_omni, vcarr_omni, bcarr_omni = Hin.generate_vCarr_from_OMNI(start, stop, omni_input=omni_sample)
    
    # Map to 210 solar radii, then 21.5 solar radii
    vcarr_210 = vcarr_omni.copy()
    vcarr_21p5 = vcarr_omni.copy()
    for i, t in enumerate(time_omni):
        # Map to 210 solar radii
        vcarr_210[:,i] = Hin.map_v_boundary_inwards(vcarr_omni[:,i]*u.km/u.s, 
                                                    earth_pos.r[i], 
                                                    210 * u.solRad)
        
        # And map to 21.5 solar radii
        vcarr_21p5[:,i] = Hin.map_v_boundary_inwards(vcarr_210[:,i]*u.km/u.s,
                                                     210 * u.solRad,
                                                     21.5 * u.solRad)
        
    return time_omni, vcarr_21p5, bcarr_omni

def backmap(start, stop, omni_samples, earth_pos):
    import multiprocessing as mp
    from dask.distributed import Client
    
    # Determine number of cores to use
    num_cores = 4# int(0.75 * mp.cpu_count())
    
    # Create a client with explicit configuration
    client = Client(
        n_workers = num_cores, 
        threads_per_worker = 1,  # Recommended for Spyder
        processes = True  # Use multiprocessing
        )
       
    # Write the dashboard link
    print_url('backmap_url.html', client.dashboard_link)
    
    # Create delayed futures
    futures = [_backmap(start, stop, s, earth_pos) 
               for s in omni_samples]
    
    # Compute with multiple workers
    results = dask.compute(*futures)
    

    client.close()
    
    return results
        

@dask.delayed(nout=1)
def _dask_propagate_totarget(start, stop, target_obs, t_grid, vc_grid, bc_grid, cme_sample, dpadding=0.03, **kwargs):
    import sys
    sys.path.append('/Users/mrutala/projects/OHTransients/HUXt/code/')
    import huxt_inputs as Hin
    import huxt as H
    import huxt_analysis as HA
    
    # Isolate the range of HEEQ longitudes covered by the target
    # HEEQ longitude, continuous
    delta = np.unwrap(target_obs.lon.value)
    if (delta > 2*np.pi).any():
        delta -= 2*np.pi
    
    # Max and Min of delta
    dmin, dmax = np.min(delta), np.max(delta)
    if ((delta >= dmin) & (delta <= dmax)).all():
        start_lon = (dmin - dpadding) % (2*np.pi)
        stop_lon = (dmax + dpadding) % (2*np.pi)
    elif ((delta >= dmax) | (delta <= dmin)).all():
        start_lon = (dmax - dpadding) % (2*np.pi)
        stop_lon = (dmin + dpadding) % (2*np.pi)
    else:
        print("Something fishy is afoot...")
        breakpoint()
        
    model = Hin.set_time_dependent_boundary(vc_grid, t_grid, 
                                            start, simtime = (stop - start).days*u.day, 
                                            bgrid_Carr = bc_grid,
                                            lon_start = start_lon * u.rad, 
                                            lon_stop =  stop_lon * u.rad,
                                            frame='synodic',
                                            r_min = 21.5 * u.solRad, r_max = target_obs.r.max(),
                                            latitude = 0.0 * u.deg,
                                            dt_scale = 50,
                                            **kwargs)
    
    model.solve(cme_sample)
    
    ts = HA.get_observer_timeseries(model, observer='PSP', obs_pos=target_obs)
    del model
    
    return ts

def propagate_totarget(start, stop, target_obs, t_grids, vc_grids, bc_grids, cme_samples):
    import multiprocessing as mp
    from dask.distributed import Client
    
    # Determine number of cores to use
    num_cores =  int(0.75 * mp.cpu_count())
    
    # Create a client with explicit configuration
    client = Client(
        n_workers = num_cores, 
        threads_per_worker = 1,  # Recommended for Spyder
        processes = True  # Use multiprocessing
        )
   
    # Write the dashboard link
    print_url('propagate_url.html', client.dashboard_link)
    
    # Create delayed futures
    futures = [_dask_propagate_totarget(start, stop, target_obs, tg, vg, bg, cmes) 
               for tg, vg, bg, cmes in zip(t_grids, vc_grids, vc_grids, cme_samples)]
    
    # Compute with multiple workers
    results = dask.compute(*futures)
    
    client.close()
    
    return results
       
if __name__ == '__main__':
    
    runstart = datetime.datetime(2024, 4, 1)
    runstop = datetime.datetime(2024, 6, 1)
    n_samples = 8
    target = 'PSP'

    # Get padded runtimes
    # !!!! This padding should be dynamic to account for propagation time to destination (Jupiter/Saturn)
    padding = datetime.timedelta(days=27)
    runstart_padded = runstart - padding
    runstop_padded = runstop + padding

    # Remove ICMEs from OMNI, and sample possible background solar wind conditions
    omni_samples = make_BackgroundSolarWind_withGP(runstart_padded, runstop_padded, n_samples)

    # Sample possible CME parameters
    # !!!!! CHECK MODEL RUN TIME WITH DIFFERENT NUMBERS OF CMES (for same duration)
    # Should we get rid of CMEs highly unlikely to matter?

    cmes = create_CME_list(runstart_padded, runstop_padded)

    cme_samples = perturb_CME_list(cmes, n_samples)

    # Get Earth, Target Body Observer classes at 4h resolution (these don't change per sample)
    times_for_pos = Time(omni_samples[0]['mjd'].to_numpy(), format='mjd')
    earth_obs = H.Observer('EARTH', times_for_pos)
    target_obs = H.Observer(target, times_for_pos)
    
    t0 = time.time()
    
    results = backmap(runstart_padded, runstop_padded, omni_samples, earth_obs)
    t_grids, vc_grids, bc_grids = list(map(list, zip(*results)))
    del results
    
    print(time.time() - t0)
    
    results = propagate_totarget(runstart_padded, runstop_padded, target_obs, 
                                 t_grids, vc_grids, bc_grids, cme_samples)
    
    print(time.time() - t0)


# # t_start = time.time()
# # futures = []
# # for 
# # futures = [propagate_totarget_withdask(runstart_padded)]
# # results = 

# # %% Actually run DASK
# # client = Client()
# delayed_results = []
# for omni_sample, cme_sample in zip(omni_samples, cme_samples):
#     dr = solve_single_model_parallel(runstart_padded, runstop_padded,
#                                      earth_pos, target_pos, 
#                                      omni_sample, cme_sample)
#     delayed_results.append(dr)
    

# t_start = timer.time()    
# results = dask.compute(*delayed_results)
# print(timer.time() - t_start)

# # %%
# # import multiprocessing as mp

# # if __name__ == "__main__":
    
# #     args = [(a, b, c, d) for a, b, c, d in zip([runstart]*n_samples, 
# #                                                [runstop]*n_samples,
# #                                                omni_samples,
# #                                                perturbed_cmelist)]
    
# #     cpus_available = mp.cpu_count()
# #     results = []
# #     with mp.Pool(cpus_available-1) as pool:
# #         generator = pool.imap(worker, args)

# #         for element in tqdm.tqdm(generator, total=len(args)):
            
# #             results.append(element)
# #     print("Program finished!")

# # test = solve_single_model(runstart, runstop, omni_samples[0], perturbed_cmelist[0])


# # # Plot PSP data
# # psp_data = lookup_PSPData(runstart, runstop)

# # fig, ax = plt.subplots()
# # # ax.scatter(psp_data['mjd'], psp_data['U'], 
# # #            color='black', marker='x', lw=1.0, s=8,
# # #            label = 'PSP Measurements')
# # ax.plot(psp_data['mjd'], psp_data['U'], 
# #            color='black', lw=1.0,
# #            label = 'PSP Measurements')


# # for i, result in enumerate(results):
# #     body_ts = HA.get_observer_timeseries(result, 'PSP')
# #     ax.plot(body_ts['mjd'], body_ts['vsw'],
# #             color='C4', lw=1, alpha=0.33)
# #     if i == 0:
# #         ax.plot(body_ts['mjd'][0:1], body_ts['vsw'][0:1],
# #                 color='C4', lw=1, alpha=1,
# #                 label = 'bOMNI-HUXt Ensemble')
# # ax.plot()
    
# # ax.legend()
    
# # ax.set(ylim = (200,800), ylabel="Solar Wind Flow Speed [km/s]",
# #        xlabel = "Date [MJD]")
# # plt.suptitle("bOMNI-HUXt @ Parker Solar Probe, ICME-subtracted, CME-added, GP data imputation")



