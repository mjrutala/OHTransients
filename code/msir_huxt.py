#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 12:20:39 2025

@author: mrutala
"""
import astropy.units as u
from astropy.time import Time
import datetime
import datetime
import os
import astropy.units as u
import glob
import re
import numpy as np
import time
from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries
import requests
import matplotlib.pyplot as plt
import pandas as pd
from astroquery.jplhorizons import Horizons
import dask

import sys
sys.path.append('/Users/mrutala/projects/OHTransients/HUXt/code/')
sys.path.append('/Users/mrutala/projects/OHTransients/code/')
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
import huxt_atObserver as hao
from scipy import ndimage
from scipy import stats
from sklearn.metrics import root_mean_squared_error as rmse

from astroquery.jplhorizons import Horizons

# import huxt_inputs_wsa as Hin_wsa
import queryDONKI

try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

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
    # omni = omni.rename(columns={'dt': 'datetime'})
    omni = omni.reset_index(drop=True)
    
    return omni

def msir_huxt(start, stop):
    """
    Multipoint Sequential Importance Resampling (MSIR) HUXt
    Estimates the best HUXt inputs by comparing to available in-situ data

    Parameters
    ----------
    start : TYPE
        DESCRIPTION.
    stop : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Get all in-situ data availability for [start, stop)
    possible_sources = ['Parker Solar Probe',
                        'Solar Orbiter',
                        'OMNI', 
                        'STEREO-A',
                        'STEREO-B',
                        ]
    
class msir_inputs:
    def __init__(self, start, stop, rmax=1, latmax=10):
        self.start = start
        self.stop = stop
        self.radmax = rmax * u.AU
        self.latmax = latmax * u.deg
        self.innerbound= 21.5 * u.solRad
        
        self.usw_minimum = 200 * u.km/u.s
        self.SiderealCarringtonRotation = 27.28 * u.day
        self.SynodicCarringtonRotation = 25.38 * u.day
        
        # Input data initialization
        self.availableSources = []
        
        self.path_cohodata = '/Users/mrutala/projects/OHTransients/data/'
        return
        
    @property
    def starttime(self):
        return Time(self.start)
    
    @property
    def stoptime(self):
        return Time(self.stop)
    
    @property
    def simpadding(self):
        return (self.radmax / self.usw_minimum).to(u.day)
    
    @property 
    def simstart(self):
        return self.start - datetime.timedelta(days=self.simpadding.to(u.day).value)
    
    @property 
    def simstop(self):
        return self.stop + datetime.timedelta(days=self.simpadding.to(u.day).value)
    
    @property
    def simstarttime(self):
        return self.starttime - self.simpadding
    
    @property
    def simstoptime(self):
        return self.stoptime + self.simpadding
    
    def _identify_source(self, source):
        
        source_aliases = {'omni': ['omni'],
                          'parker solar probe': ['parkersolarprobe', 'psp', 'parker solar probe'],
                          'stereo a': ['stereoa', 'stereo a', 'sta'],
                          'stereo b': ['stereob', 'stereo b', 'stb'],
                          # 'helios1': ['helios1', 'helios 1'],
                          # 'helios2': ['helios2', 'helios 2'],
                          'ulysses': ['ulysses', 'uy'],
                          # 'maven': ['maven'],
                          'voyager 1': ['voyager1', 'voyager 1'],
                          'voyager 2': ['voyager2', 'voyager 2']}
        
    def get_availableBackgroundData(self, sources=None):
        
        source_functions = {'omni': self.get_omni,
                            'parker solar probe': self.get_parkersolarprobe,
                            'stereo a': self.get_stereoa,
                            'stereo b': self.get_stereob,
                            # 'helios1': self.get_helios1,
                            # 'helios2': self.get_helios2,
                            'ulysses': self.get_ulysses,
                            # 'maven': self.get_maven,
                            'voyager 1': self.get_voyager1,
                            'voyager 2': self.get_voyager2}
        
        all_sources = list(source_functions.keys())
        
        # Check if sources are specified; if not, use them all
        if sources is None:
            sources = all_sources
        else:
            breakpoint()
            #!!!! Add functionality to search alias dictionary
        
        # Read each source into a dictionary
        available_sources = []
        available_data_dict = {}
        for source in sources:
            data_df = source_functions[source]()
            if data_df is not None: 
                available_sources.append(source)
                available_data_dict[source] = data_df
                
        available_data_df = pd.concat(available_data_dict, axis='columns')
        available_data_df['mjd'] = Time(available_data_df.index).mjd
        
        self.availableSources.extend(available_sources)
        self.availableData = available_data_df
            
        return
    
    def _fetchFromCOHOWeb(self, source, fetch_datasetID):
        # Mute warnings
        import warnings
        warnings.filterwarnings("ignore")
                
        # Simulation time range in Fido format
        timerange = attrs.Time(self.simstarttime, self.simstoptime)
        
        # 
        dataset = attrs.cdaweb.Dataset(fetch_datasetID)
        result = Fido.search(timerange, dataset)
        downloaded_files = Fido.fetch(result, path = self.path_cohodata + source.upper() + '/{file}')
        
        # If there's any files to download, download them and get the dataframe
        if len(downloaded_files) > 0:
            # Combine the data for this source into one df
            df = TimeSeries(downloaded_files, concatenate=True).to_dataframe()
           
            # Truncate to the appropriate start/stop time
            df = df.query("@self.simstart <= index < @self.simstop")
            
        else:
            df = None

        return df
    
    def get_omni(self):
        
        # Source and Dataset names
        source = 'omni'
        fetch_datasetID = 'OMNI_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'BR': 'BR',
                      'V': 'U',
                      'elevAngle': 'Uel',
                      'azimuthAngle': 'Uaz',
                      'rad_HGI': 'rad_HGI',
                      'heliographicLatitude': 'lat_HGI',
                      'heliographicLongitude': 'lon_HGI'}
        
        # Get the data
        data_df = self._fetchFromCOHOWeb(source, fetch_datasetID)
        
        # If data exists, rename columns & add distance
        if data_df is not None:
            data_df.rename(columns = column_map, inplace=True)
            # Add the radial distance
            data_df['rad_HGI'] = 1.0
            data_df = data_df[column_map.values()]
        
        return data_df
    
    def get_stereoa(self):
        # Source and Dataset names
        source = 'stereo a'
        fetch_datasetID = 'STA_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'BR': 'BR',
                      'plasmaSpeed': 'U',
                      'lat': 'Uel',
                      'lon': 'Uaz',
                      'radialDistance': 'rad_HGI',
                      'heliographicLatitude': 'lat_HGI',
                      'heliographicLongitude': 'lon_HGI'}
        
        # Get the data
        data_df = self._fetchFromCOHOWeb(source, fetch_datasetID)
        
        # If data exists, rename columns
        if data_df is not None:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
        
        return data_df
        
    def get_stereob(self):
        # Source and Dataset names
        source = 'stereo b'
        fetch_datasetID = 'STB_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'BR': 'BR',
                      'plasmaSpeed': 'U',
                      'lat': 'Uel',
                      'lon': 'Uaz',
                      'radialDistance': 'rad_HGI',
                      'heliographicLatitude': 'lat_HGI',
                      'heliographicLongitude': 'lon_HGI'}
        
        # Get the data
        data_df = self._fetchFromCOHOWeb(source, fetch_datasetID)
        
        # If data exists, rename columns
        if data_df is not None:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
        
        return data_df    
    
    def get_voyager1(self):
        
        # Source and Dataset names
        source = 'voyager 1'
        fetch_datasetID = 'VOYAGER1_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'BR': 'BR',
                      'V': 'U',
                      'elevAngle': 'Uel',
                      'azimuthAngle': 'Uaz',
                      'heliocentricDistance': 'rad_HGI',
                      'heliographicLatitude': 'lat_HGI',
                      'heliographicLongitude': 'lon_HGI'}
        
        # Get the data
        data_df = self._fetchFromCOHOWeb(source, fetch_datasetID)
        
        # If data exists, rename columns
        if data_df is not None:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
        
        return data_df
    
    def get_voyager2(self):
        
        # Source and Dataset names
        source = 'voyager 2'
        fetch_datasetID = 'VOYAGER2_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'BR': 'BR',
                      'V': 'V',
                      'elevAngle': 'Uel',
                      'azimuthAngle': 'Uaz',
                      'heliocentricDistance': 'rad_HGI',
                      'heliographicLatitude': 'lat_HGI',
                      'heliographicLongitude': 'lon_HGI'}
        
        # Get the data
        data_df = self._fetchFromCOHOWeb(source, fetch_datasetID)
        
        # If data exists, rename columns
        if data_df is not None:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
        
        return data_df
    
    def get_ulysses(self):
        # Source and Dataset names
        source = 'ulysses'
        fetch_datasetID = 'UY_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'BR': 'BR',
                      'plasmaFlowSpeed': 'U',
                      'elevAngle': 'Uel',
                      'azimuthAngle': 'Uaz',
                      'heliocentricDistance': 'rad_HGI',
                      'heliographicLatitude': 'lat_HGI',
                      'heliographicLongitude': 'lon_HGI'}
        
        # Get the data
        data_df = self._fetchFromCOHOWeb(source, fetch_datasetID)
        
        # If data exists, rename columns
        if data_df is not None:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
        
        return data_df    
    
    def get_parkersolarprobe(self):
        # Source and Dataset names
        source = 'parker solar probe'
        fetch_datasetID = 'PSP_COHO1HR_MERGED_MAG_PLASMA'
        
        # Which columns to keep, and what to call them
        column_map = {'BR': 'BR',
                      'ProtonSpeed': 'U',
                      'flow_theta': 'Uel',
                      'flow_lon': 'Uaz',
                      'radialDistance': 'rad_HGI',
                      'heliographicLatitude': 'lat_HGI',
                      'heliographicLongitude': 'lon_HGI'}
        
        # Get the data
        data_df = self._fetchFromCOHOWeb(source, fetch_datasetID)
        
        # If data exists, rename columns
        if data_df is not None:
            data_df.rename(columns = column_map, inplace=True)
            data_df = data_df[column_map.values()]
        
        return data_df    
    
    def filter_availableBackgroundData(self):
        
        sources_to_remove = []
        for source in self.availableSources:
            
            # Where is the source out of radial and latitudinal range?
            out_of_range = (np.abs(self.availableData[(source, 'lat_HGI')]) > np.abs(self.latmax)) &\
                           (self.availableData[(source, 'rad_HGI')] > self.radmax)
            
            # Set these as NaNs
            self.availableData.loc[out_of_range, source] = np.nan
            
            # If no data is in range, delete the source and columns entirely
            if out_of_range.all() == True:
                sources_to_remove.append(source)
                self.availableData.drop(columns = source, level = 0, inplace = True)
                          
        for source in sources_to_remove:
            self.availableSources.remove(source)
            
        return
    
    def sort_availableSources(self, column='rad_HGI'):
        
        mean_vals = {}
        for source in self.availableSources:
            mean_vals[source] = np.mean(self.availableData[(source, column)])
            
        order = np.argsort(list(mean_vals.values()))
        self.availableSources = np.array(list(mean_vals.keys()))[order]
        
        return
    
    # def _print_url(self, filename, url):
    #     with open(filename, 'w') as f:
    #         print('<html>', file=f)
    #         print(' <body>', file=f)
    #         print('  <script type="text/javascript">', file=f)
    #         print('   window.location.href = "{}"'.format(url), file=f)
    #         print('  </script>', file=f)
    #         print(' </body>', file=f)
    #         print('</html>', file=f)
    #     return
    
    def get_availableTransientData(self, sources=None, duration=2*u.day):
        
        location_aliases = {'omni': 'Earth',
                            'stereo a': 'STEREO%20A',
                            'stereo b': 'STEREO%20B',
                            'maven': 'Mars'}
        
        all_sources = list(location_aliases.keys())
        
        # Parse which sources to lookup transients for
        if sources is None:
            # Either use the sources we have data for, or all of them
            if len(self.availableSources) > 0:
                sources = list(set(all_sources).intersection(set(self.availableSources)))  
            else:
                sources = all_sources
        else:
            breakpoint()
            #!!!! Add functionality to search alias dictionary
        
        # Lookup ICMEs for each source
        availableTransientData_list = []
        for source in sources:
            location = location_aliases[source]
            icmes = queryDONKI.ICME(self.simstart, 
                                    self.simstop, 
                                    location = location, 
                                    duration = duration)
            icmes['affiliated_source'] = source
            
            availableTransientData_list.append(icmes)
        
        availableTransientData_df = pd.concat(availableTransientData_list, axis='rows')
        availableTransientData_df['mjd'] = Time(availableTransientData_df['eventTime'])
        availableTransientData_df.reset_index(inplace=True)

        self.availableTransientData = availableTransientData_df 
        
        return
    
    def generate_backgroundDistributions(self, insitu=None, icmes=None):
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
        # subsetSize = 1000
        # min_l      = 0.0 
        # mid_l      = 0.05
        # max_l      = 0.2
        # init_var   = 3
        period = 27 * u.day
        n_samples = 10
        average_cluster_span = 6 * u.hour # For K-means clustering to reduce data 
        
        # Calculate the span from stop - start
        span = self.simstop - self.simstart
        
        breakpoint()
        
        # If no omni dataset is supplied, look one up
        if insitu is None:
            insitu = self.get_omni()
        # omni = lookup_omni(self.simstart, self.simstop)
        # omni = self.get_omni()
        if 'mjd' not in insitu.columns:
            insitu['mjd'] = Time(insitu.index).mjd # needed for Hin.remove_ICMEs()
        
        # If no icmes are supplied, look them up
        if icmes is None:
            icmes = queryDONKI.ICME(self.simstart, self.simstop, location = 'Earth', duration = icme_duration)
        
        # Remove ICMEs from OMNI data, leaving NaNs behind
        if 'eventTime' in icmes.columns: 
            icmes = icmes.rename(columns = {'eventTime': 'Shock_time'})
            icmes['ICME_end'] = [row['Shock_time'] + datetime.timedelta(days=(row['duration'])) for _, row in icmes.iterrows()]
        
        insitu_noicme = Hin.remove_ICMEs(insitu, icmes, 
                                         params=['U', 'BR'], 
                                         interpolate = False, 
                                         icme_buffer = icme_buffer, 
                                         interp_buffer = interp_buffer, 
                                         fill_vals = np.nan)
        
        # Get the mjd and U as column vectors for GPflow
        mjd = insitu_noicme.dropna(axis='index', how='any')['mjd'].to_numpy()[:, None]
        speed = insitu_noicme.dropna(axis='index', how='any')['U'].to_numpy('float64')[:, None]

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
        model = gpflow.models.GPR((Xc, Yc), 
                                  kernel=signal_kernel, 
                                  noise_variance=np.percentile(Yc_var, 90))
        
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        
        # gpflow.utilities.print_summary(model)
        
        # Xp = np.arange(qomni_df['mjd'].iloc[0], qomni_df['mjd'].iloc[-1]+30, 1)[:, None]
        Xo = insitu['mjd'].to_numpy('float64')[:, None]
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
        
        # Add samples to omni, return as a list
        # results = []
        # for fo_sample in fo_samples:
        #     new_omni = omni_noicme.copy(deep=True)
        #     new_omni['U'] = fo_sample
        #     results.append(new_omni)
        
        new_insitu = insitu_noicme.copy(deep=True)
        # new_omni['U'] = Yo_mu.ravel()
        # v_omni_dict = {'mu-sig': Yo_mu.ravel() - Yo_sig.ravel(),
        #                'mu': Yo_mu.ravel(),
        #                'mu+sig': Yo_mu.ravel() + Yo_sig.ravel()}
        
        new_insitu['U_mu'] = Yo_mu.ravel()
        new_insitu['U_sig'] = Yo_sig.ravel()
        new_insitu.drop(columns='U', inplace=True)
        self.backgroundDistribution = new_insitu
        
        # =============================================================================
        # Visualization
        # =============================================================================
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(insitu['mjd'], insitu['U'], color='C0', marker='.', s=2, zorder=1,
                   label = 'OMNI Data')
        ax.scatter(insitu_noicme['mjd'], insitu_noicme['U'], color='black', marker='.', s=2, zorder=2,
                   label="OMNI Data - ICMEs")
        # ax.scatter(Xc, Yc, label='Inducing Points', color='C1', marker='o', s=6, zorder=4)
        
        ax.plot(Xo, Yo_mu, label="Mean prediction", color='C5', zorder=0)
        ax.fill_between(
            Xo.ravel(),
            (Yo_mu - 1.96 * Yo_sig).ravel(),
            (Yo_mu + 1.96 * Yo_sig).ravel(),
            alpha=0.33, color='C1',
            label=r"95% confidence interval", zorder=-2)
        
        for fo_sample in fo_samples:
            ax.plot(Xo.ravel(), fo_sample.ravel(), lw=1, color='C3', alpha=0.2, zorder=-1)
        ax.plot(Xo.ravel()[0:1], fo_sample.ravel()[0:1], lw=1, color='C3', alpha=1, 
                label = 'Samples about Mean')
        
        ax.legend(scatterpoints=3, loc='upper right')
        
        ax.set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        ax.set(xlabel='Date [MJD], from {}'.format(datetime.datetime.strftime(self.start, '%Y-%m-%d %H:%M')), 
               ylabel='Solar Wind Speed [km/s]', 
               title='Gaussian Process (GP) Data Imputation: OMNI (@ 1AU) without ICMEs',)
        
        plt.show()
        
        return
    
    def generate_boundaryDistribution(self, nSamples=10, constant_sig=50):
        
        # Format the OMNI DataFrame (backgroundDistribution) as HUXt expects it
        _insitu_df = self.backgroundDistribution.copy(deep=True)
        _insitu_df['BX_GSE'] =  -_insitu_df['BR']
        _insitu_df['datetime'] = _insitu_df.index
        _insitu_df = _insitu_df.reset_index()
        
        # Define an inner function to be run in parallel
        def map_vBoundaryInwards(insitu_df, corot_type):
            
            # Generate the Carrington grids
            t, vcarr, bcarr = Hin.generate_vCarr_from_OMNI(self.simstart, self.simstop, 
                                                           omni_input=insitu_df, corot_type=corot_type)
            
            # Map to 210 solar radii, then to the inner boundary for the model
            vcarr_210 = vcarr.copy()
            vcarr_21p5 = vcarr.copy()
            for i, _ in enumerate(t):
                
                vcarr_210[:,i] = Hin.map_v_boundary_inwards(vcarr[:,i]*u.km/u.s, 
                                                            (insitu_df['rad_HGI'][i]*u.AU).to(u.solRad), 
                                                            210 * u.solRad)
                
                vcarr_21p5[:,i] = Hin.map_v_boundary_inwards(vcarr_210[:,i]*u.km/u.s,
                                                             210 * u.solRad,
                                                             self.innerbound)
                
            return vcarr_21p5
        
        
        # Map inwards once for a "nominal" boundary
        _insitu_df['V'] = _insitu_df['U_mu']
        t, vcarr, bcarr = Hin.generate_vCarr_from_OMNI(self.simstart, self.simstop, omni_input=_insitu_df)
        
        # # Dask client for monitoring & worker setup
        # from dask.distributed import Client
        # client = Client(threads_per_worker=1, n_workers=8)
        # print(client.dashboard_link)
        
        # # Random generator to resample U from the backgroundDistribution
        # rng = np.random.default_rng()
        # delayed_results = []
        # for _ in range(nSamples):
        #     _insitu_df_copy = _insitu_df.copy(deep=True)
        #     _insitu_df_copy['V'] = rng.normal(loc=_insitu_df['U_mu'], scale=_insitu_df['U_sig'])
        #     delayed_results.append(map_vBoundaryInwards(_insitu_df_copy, 'back'))
        #     delayed_results.append(map_vBoundaryInwards(_insitu_df_copy, 'forward'))
        #     delayed_results.append(map_vBoundaryInwards(_insitu_df_copy, 'both'))
        
        # # Compute the delayed results
        # with dask.config.set({'distributed.admin.tick.limit': '60s'}):
        #     vcarrs = dask.compute(delayed_results)[0]
        
        # # Characterize the resulting samples as one distribution
        # vcarr_mu = np.mean(vcarrs, axis=0)
        # vcarr_sig = np.sqrt(np.std(vcarrs, axis=0)**2 + constant_sig**2)
        
        # # self.backgroundDistribution = {'t': time_omni,
        # #                                'u_mu': vgrids_dict['mu'], 
        # #                                'u_sig': (vgrids_dict['mu+sig'] - vgrids_dict['mu-sig'])/2,
        # #                                'B': bcarr_omni}
        
        # breakpoint()
        
        from tqdm import tqdm
        # from dask.distributed import Client, as_completed, LocalCluster
        import multiprocessing as mp
        import logging
        
        nCores = int(0.75 * mp.cpu_count()) 

        rng = np.random.default_rng()
        
        # dask.config.set({
        #                 'distributed.logging.distributed': 'error',
        #                 'distributed.logging.distributed.core': 'error', 
        #                 'distributed.logging.distributed.worker': 'error',
        #                 'distributed.logging.distributed.scheduler': 'error',
        #                 'distributed.admin.log-level': 'error'
        #             })
        
        # with dask.config.set({'distributed.admin.tick.limit': '60s'}):
    
        #     # logging.getLogger('dask').setLevel(logging.WARNING)
        #     # logging.getLogger('distributed').setLevel(logging.WARNING)

            
        #     cluster = LocalCluster(n_workers=n_cores, 
        #                            threads_per_worker=1, 
        #                            silence_logs=logging.ERROR, 
        #                            dashboard_address=None
        #                            )
        #     client = Client(cluster)
            
        #     futures = []
        #     for _ in range(nSamples):
                
        #         _insitu_df_copy = _insitu_df.copy(deep=True)
        #         _insitu_df_copy['V'] = rng.normal(loc=_insitu_df['U_mu'], scale=_insitu_df['U_sig'])
                
        #         # delayed_results.append(map_vBoundaryInwards(_insitu_df_copy, 'back'))
        #         # delayed_results.append(map_vBoundaryInwards(_insitu_df_copy, 'forward'))
        #         # delayed_results.append(map_vBoundaryInwards(_insitu_df_copy, 'both'))
                
        #         future = client.submit(map_vBoundaryInwards, _insitu_df_copy, 'both')
                
        #         futures.append(future)
            
        # # Append the results
        # ordered_dict = {}
        # for future, result in tqdm(as_completed(futures, with_results=True), total=len(futures)):
        #     ordered_dict[future.key] = result
    
        # # Now reorder them based on the original futures order
        # ordered_list = [ordered_dict[future.key] for future in futures]
        # del futures
        
        from joblib import Parallel, delayed
        from tqdm import tqdm
        
        # Calculate boundary distributions by backmapping each sample
        def process_sample(U_sample, method_sample):
            insitu_df_copy = _insitu_df.copy(deep=True)
            insitu_df_copy['V'] = U_sample
            return map_vBoundaryInwards(insitu_df_copy, method_sample)
        
        # Sample the velocity distribution and assign random mapping directions (method)
        # Randomly assigning these is equivalent to performing each mapping for each sample (for large numbers of samples)
        # Having a single random population should be better mathematically
        uSamples = [rng.normal(loc=_insitu_df['U_mu'], 
                               scale=_insitu_df['U_sig']
                               ) for _ in range(nSamples)]
        # methodSamples = rng.choice(['forward', 'back', 'both'], nSamples)
        methodSamples = rng.choice(['both'], nSamples)
        
        vcarrGenerator = Parallel(return_as='generator', n_jobs=nCores)(
            delayed(process_sample)(u, method) 
            for u, method in zip(uSamples, methodSamples)
            )
        vcarrSamples = list(tqdm(vcarrGenerator, total=nSamples))

        # Characterize the resulting samples as one distribution
        vcarr_mu = np.mean(vcarrSamples, axis=0)
        vcarr_sig = np.sqrt(np.std(vcarrSamples, axis=0)**2 + constant_sig**2)
        
        self.boundaryDistribution = {'t_grid': t,
                                     'U_mu_grid': vcarr_mu,
                                     'U_sig_grid': vcarr_sig,
                                     'B_grid': bcarr}
        
        # =============================================================================
        # Visualization 
        # =============================================================================
        fig, axs = plt.subplots(figsize=(6,4.5), ncols=2)
        
        mu_img = axs[0].imshow(vcarr_mu, 
                               extent=[self.simstarttime.mjd, self.simstoptime.mjd, 0, 360], 
                               origin='lower', aspect=0.2)
        axs[0].set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        fig.colorbar(mu_img, ax=axs[0])
        
        sig_img = axs[1].imshow(vcarr_sig, 
                                extent=[self.simstarttime.mjd, self.simstoptime.mjd, 0, 360], 
                                origin='lower', aspect=0.2)
        axs[1].set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        fig.colorbar(sig_img, ax=axs[1])
        
        axs[0].set(ylabel='Heliolongitude [deg.]', xlabel='Date [MJD]')
        axs[1].set(xlabel='Date [MJD]')
        
        plt.show()
        
        return
    
    def generate_cmeDistribution(self):
        
        # 
        t_sig_init = 36000 # seconds
        lon_sig_init = 15 # degrees
        lat_sig_init = 15 # degrees
        width_sig_init = 30 # degrees
        thick_mu_init = 4 # solar radii
        thick_sig_init = 1 # solar radii
        speed_sig_init = 400 # km/s
        
        # Get the CMEs
        cmes = queryDONKI.CME(self.simstart, self.simstop)

        cmeDistribution_dict = {'t_mu': [], 't_sig': [],
                                'lon_mu': [], 'lon_sig': [],
                                'lat_mu': [], 'lat_sig': [],
                                'width_mu': [], 'width_sig': [],
                                'speed_mu': [], 'speed_sig': [],
                                'thickness_mu': [], 'thickness_sig': [],
                                'innerbound': []}
        
        for index, row in cmes.iterrows():
            info = row['cmeAnalyses']
            
            t = (datetime.datetime.strptime(info['time21_5'], "%Y-%m-%dT%H:%MZ") - self.simstart).total_seconds()
            cmeDistribution_dict['t_mu'].append(t)
            cmeDistribution_dict['t_sig'].append(t_sig_init)
            
            cmeDistribution_dict['lon_mu'].append(info['longitude'])
            cmeDistribution_dict['lon_sig'].append(lon_sig_init)
            
            cmeDistribution_dict['lat_mu'].append(info['latitude'])
            cmeDistribution_dict['lat_sig'].append(lat_sig_init)
            
            cmeDistribution_dict['width_mu'].append(2*info['halfAngle'])
            cmeDistribution_dict['width_sig'].append(width_sig_init)
            
            cmeDistribution_dict['speed_mu'].append(info['speed'])
            cmeDistribution_dict['speed_sig'].append(speed_sig_init)
            
            cmeDistribution_dict['thickness_mu'].append(thick_mu_init)
            cmeDistribution_dict['thickness_sig'].append(thick_sig_init)
            
            cmeDistribution_dict['innerbound'].append(21.5)
         
        cmeDistribution = pd.DataFrame(cmeDistribution_dict)
        
        # Drop CMEs at high lat
        lat_cutoff = np.abs(cmeDistribution['lat_mu']) > 2.0*self.latmax
        cmeDistribution.loc[lat_cutoff, 'lat_mu'] = np.nan
        
        # Drop NaNs
        cmeDistribution.dropna(how='any', axis='index', inplace = True)
        
        self.cmeDistribution = cmeDistribution
        
        return
    

    
    
    def sample(self, weights):
        
        n_samples = len(weights)
        
        rng = np.random.default_rng()
        
        # Plain normal samples
        # backgroundSamples = rng.normal(loc=self.backgroundDistribution['u_mu'],
        #                                scale=self.backgroundDistribution['u_sig'])
        
        # Offset normal samples
        boundarySamples_U = []
        offsets = rng.normal(loc=0, scale=1, size=n_samples)
        offsets_ratio = 0.1
        for offset in offsets:
            boundarySamples_U.append(rng.normal(loc=self.boundaryDistribution['U_mu_grid'] + offsets_ratio*offset*self.boundaryDistribution['U_sig_grid'],
                                              scale=(1-offsets_ratio)*self.boundaryDistribution['U_sig_grid'],
                                              )) 
        
        # To sample the CMEs
        cmeSamples = []
        n_cmes = len(self.cmeDistribution)
        for i in range(n_samples):
            
            cmeSample = {}
            cmeSample['t'] = rng.normal(self.cmeDistribution['t_mu'], 
                                        self.cmeDistribution['t_sig'])
            
            cmeSample['lon'] = rng.normal(self.cmeDistribution['lon_mu'],
                                          self.cmeDistribution['lon_sig'])
            
            cmeSample['lat'] = rng.normal(self.cmeDistribution['lat_mu'],
                                          self.cmeDistribution['lat_sig'])
            
            cmeSample['width'] = rng.lognormal(self.cmeDistribution['width_mu'],
                                               self.cmeDistribution['width_sig'])
            
            cmeSample['thickness'] = rng.lognormal(self.cmeDistribution['thickness_mu'],
                                                   self.cmeDistribution['thickness_sig'])
            
            cmeSample['speed'] = rng.normal(loc=self.cmeDistribution['speed_mu'],
                                            scale=self.cmeDistribution['speed_sig'])
            
            cmeSample['innerbound'] = self.cmeDistribution['innerbound']
            
            cmeSamples.append(pd.DataFrame(data=cmeSample))
        
        # self.nSamples = n_samples
        # # self.boundarySamples = boundarySamples
        # # self.cmeSamples = cmeSamples
        
        return boundarySamples_U, cmeSamples
    
    def predict(self, boundarySamples_U, cmeSamples, observer_name, dpadding=0.03):
        import multiprocessing as mp
        from tqdm import tqdm
        from dask.distributed import Client, wait, progress, as_completed
        import logging
        logging.disable(logging.INFO)
        # dask.config.set({'logging.distributed': 'error'})
        # dask.config.set({'logging.futures': 'error'})
        
        # DO NOT loop over this bit
        observer = H.Observer(observer_name, Time(self.boundaryDistribution['t_grid'], format='mjd'))
        
        n_cores = int(0.75 * mp.cpu_count()) 
        client = Client(n_workers = n_cores,
                        threads_per_worker = 1,
                        silence_logs = 40)
        
        futures = []
        for boundarySample_U, cmeSample in zip(boundarySamples_U, cmeSamples):
        # for i in range(self.nSamples):
            # DO loop over these bits
            cme_list = []
            for index, row in cmeSample.iterrows():
                
                cme = H.ConeCME(t_launch=row['t']*u.s, 
                                longitude=row['lon']*u.deg, 
                                latitude=row['lat']*u.deg, 
                                width=row['width']*u.deg, 
                                v=row['speed']*(u.km/u.s), 
                                thickness=row['thickness']*u.solRad, 
                                initial_height=row['innerbound']*u.solRad,
                                cme_expansion=False,
                                cme_fixed_duration=True)
                
                cme_list.append(cme)
            
            future = client.submit(hao.huxt_atObserver, self.simstart, self.simstop,
                                   self.boundaryDistribution['t_grid'], 
                                   boundarySample_U,
                                   self.boundaryDistribution['B_grid'], 
                                   observer_name, observer,
                                   dpadding = dpadding, 
                                   cme_list = cme_list,
                                   r_min=self.innerbound)
            
            futures.append(future)
            
        t0 = time.time()
        
        # Append the results, after interpolating to internal data index
        ordered_dict = {}
        for future, result in tqdm(as_completed(futures, with_results=True), total=len(futures)):
            interp_result = pd.DataFrame(index=self.availableData.index,
                                         columns=result.columns)
            for col in interp_result.columns:
                interp_result[col] = np.interp(self.availableData['mjd'], result['mjd'], result[col])
                
            ordered_dict[future.key] = interp_result
        
        # Now reorder them based on the original futures order
        ensemble = [ordered_dict[future.key] for future in futures]
        del futures
        
        print("{} HUXt forecasts completed in {}s".format(len(ensemble), time.time()-t0))
        
        # =============================================================================
        # Visualize    
        # =============================================================================
        fig, ax = plt.subplots(figsize=(6,4.5))
        
        for member in ensemble:
            ax.plot(member['mjd'], member['U'], color='C3', lw=1, alpha=0.2)
        ax.plot(member['mjd'][0:1], member['U'][0:1], lw=1, color='C3', alpha=1, 
                label = 'Ensemble Members')
        
        
        ax.legend(scatterpoints=3, loc='upper right')
        
        ax.set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        ax.set(xlabel='Date [MJD], from {}'.format(datetime.datetime.strftime(self.start, '%Y-%m-%d %H:%M')), 
               ylabel='Solar Wind Speed [km/s]', 
               title='HUXt Ensemble @ {}'.format(observer_name))
        
        plt.show()
            
            
        return ensemble
    
    def predict2(self, boundarySamples_U, cmeSamples, observer_name, dpadding=0.03):
        import multiprocessing as mp
        from tqdm import tqdm
        from joblib import Parallel, delayed
        
    
        t0 = time.time()
        
        # DO NOT loop over this bit
        observer = H.Observer(observer_name, Time(self.boundaryDistribution['t_grid'], format='mjd'))
        
        nCores = int(0.75 * mp.cpu_count()) 
        
        # Calculate boundary distributions by backmapping each sample
        def runHUXt(boundarySample_U, cmeSample):
            
            cme_list = []
            for index, row in cmeSample.iterrows():
                
                cme = H.ConeCME(t_launch=row['t']*u.s, 
                                longitude=row['lon']*u.deg, 
                                latitude=row['lat']*u.deg, 
                                width=row['width']*u.deg, 
                                v=row['speed']*(u.km/u.s), 
                                thickness=row['thickness']*u.solRad, 
                                initial_height=row['innerbound']*u.solRad,
                                cme_expansion=False,
                                cme_fixed_duration=True)
                
                cme_list.append(cme)
                
            future = hao.huxt_atObserver(self.simstart, self.simstop,
                                         self.boundaryDistribution['t_grid'], 
                                         boundarySample_U,
                                         self.boundaryDistribution['B_grid'], 
                                         observer_name, observer,
                                         dpadding = dpadding, 
                                         cme_list = cme_list,
                                         r_min=self.innerbound)
            
            
            futureInterpolated = pd.DataFrame(index=self.availableData.index,
                                              columns=future.columns)
            for col in futureInterpolated.columns:
                futureInterpolated[col] = np.interp(self.availableData['mjd'], future['mjd'], future[col])
            
            return futureInterpolated
        
        futureGenerator = Parallel(return_as='generator', n_jobs=nCores)(
            delayed(runHUXt)(boundarySample_U, cmeSample) 
            for boundarySample_U, cmeSample in zip(boundarySamples, cmeSamples)
            )
        
        ensemble = list(tqdm(futureGenerator, total=nSamples))
        
        print("{} HUXt forecasts completed in {}s".format(len(ensemble), time.time()-t0))
        
        # =============================================================================
        # Visualize    
        # =============================================================================
        fig, ax = plt.subplots(figsize=(6,4.5))
        
        for member in ensemble:
            ax.plot(member['mjd'], member['U'], color='C3', lw=1, alpha=0.2)
        ax.plot(member['mjd'][0:1], member['U'][0:1], lw=1, color='C3', alpha=1, 
                label = 'Ensemble Members')
        
        
        ax.legend(scatterpoints=3, loc='upper right')
        
        ax.set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        ax.set(xlabel='Date [MJD], from {}'.format(datetime.datetime.strftime(self.start, '%Y-%m-%d %H:%M')), 
               ylabel='Solar Wind Speed [km/s]', 
               title='HUXt Ensemble @ {}'.format(observer_name))
        
        plt.show()
            
        return ensemble
    
    # def score_propagatedInputs(self, ensemble, source):
    #     import numpy as np
    #     breakpoint()
        
    #     fig, ax = plt.subplots()
    #     ax.scatter(self.availableData['mjd'], self.availableData[(source, 'U')],
    #                color='black', marker='x', label=source)
        
    #     for member in ensemble[:-1]:
    #         ax.plot(member['mjd'], member['U'],
    #                 color='C0', lw=1, alpha=0.2)
    #     ax.plot(ensemble[-1]['mjd'], ensemble[-1]['U'],
    #             color='C0', alpha=0.25, label='bOMNI-HUXt Ensemble')
        
    #     fig, ax = plt.subplots()
    #     for member in ensemble:
    #         ax.plot()
        
    #     ax.set(xlabel = 'MJD, June 2012', xlim=(self.starttime.mjd, self.stoptime.mjd),
    #            ylabel = 'Solar Wind Speed [km/s]', ylim=[200, 800])
    #     ax.legend()
    #     plt.show()
        
    #     new_weights = []
    #     for member in ensemble:
        
    #         # Interpolate the member to match the source timing
    #         member_interp = np.inter
        
    #         # Get a score for the member
    #         # w = score_function(self.availableData[(source, 'U')], member_interp)
                                                  
    #         # Record the scores as weights
    #         # new_weights.append(w)
            
    #     # Get the new net weights
    #     weights = self.innerBoundaryWeights * np.array(new_weights)
        
    #     breakpoint()
        
    #     return
        
    def intialize(self):
        return
    # def predict(): # in loop
        
    
    #     return
    def update_weights(self, ensemble, weights, intrinsic_uncertainty, data): # in loop
        
        from sklearn.metrics import r2_score
    
        # # Get the distance at each point between the meta-model and data
        # # !!!! Should estimate come first, so we have this?
        # metamodel = pd.concat(ensemble).groupby(ensemble[0].index.name).median()
        
        # metadistances = data['U'] - metamodel['U']
        
        # mask = ~np.isnan(data['U']) & ~np.isnan(metamodel['U'])
        
        # # Treat each data point (time step) as a 'landmark'
        # ensemble_U = np.array([member.loc[mask,'U'] for member in ensemble])
        # l = []
        # for i, value in enumerate(data.loc[mask,'U']):
        #     distances = value - ensemble_U[:,i]
        #     likelihood = norm(distances, intrinsic_uncertainty).pdf(metadistances[mask][i])
        #     weights *= likelihood/likelihood.sum()
        #     l.append(likelihood/likelihood.sum())
    
        # weights += 1e-300       # avoid round-off to zero
        # weights /= weights.sum() # normalize
        
        # Consider one 'landmark' per interval
        # Measure the 'distance' between time series
        distances = []
        residuals = []
        for member in ensemble:
            mask =  ~np.isnan(data['U']) & ~np.isnan(member['U'])
            # dist = rmse(data.loc[mask, 'U'], member.loc[mask, 'U'])
            try:
                dist = r2_score(data.loc[mask, 'U'], member.loc[mask, 'U'])
            except:
                breakpoint()
            distances.append(dist)
            residuals.append(np.abs(data.loc[mask, 'U'] - member.loc[mask, 'U']))
        distances = np.array(distances)
        
        likelihood = distances/distances.sum()
        
        posterior = (weights * likelihood)/(weights * likelihood).sum() 
            
        # =============================================================================
        # Visualize    
        # =============================================================================
        fig, ax = plt.subplots(figsize=(6,4.5))
        
        for member in ensemble:
            ax.plot(member['mjd'], member['U'], color='C3', lw=1, alpha=0.2)
        ax.plot(member['mjd'][0:1], member['U'][0:1], lw=1, color='C3', alpha=1, 
                label = 'Ensemble Members')
        
        best_indx = np.argmax(posterior)
        ax.plot(ensemble[best_indx]['mjd'], ensemble[best_indx]['U'], color='C1', lw=1.5, label='Highest Posterior Probability')
        worst_indx = np.argmin(posterior)
        ax.plot(ensemble[worst_indx]['mjd'], ensemble[worst_indx]['U'], color='C0', lw=1.5, label='Lowest Posterior Probability')
        
        ax.scatter(member['mjd'], data['U'], s=1, marker='.', label='stereo a', color='black')
        
        ax.legend(scatterpoints=3, loc='upper right')
        
        ax.set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        ax.set(xlabel='Date [MJD], from {}'.format(datetime.datetime.strftime(self.start, '%Y-%m-%d %H:%M')), 
               ylabel='Solar Wind Speed [km/s]', 
               title='HUXt Ensemble @ {}'.format('stereo a'))
        

        plt.show()
        
        # Time-dynamic weighting
        
        
        return posterior/posterior.sum()
        
     
    def estimate(self, ensemble, weights): # in loop
        """
        Return a weighted median metamodel
    
        Parameters
        ----------
        ensemble : TYPE
            DESCRIPTION.
        weights : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        metamodel = ensemble[0].copy(deep=True)
        
        for col in metamodel.columns:
            for index in metamodel.index:
                vals = [m.loc[index, col] for m in ensemble]
                valsort_indx = np.argsort(vals)
                cumsum_weights = np.cumsum(weights[valsort_indx])
                weighted_median = vals[valsort_indx[np.searchsorted(cumsum_weights, 0.5 * cumsum_weights[-1])]]
                
                metamodel.loc[index, col] = weighted_median
        
        return
    
    def update_distributions(self, weights, boundarySamples, cmeSamples): 
        
        n_samples = len(weights)
        
        # For the Background Solar Wind
        # Get a new mean and standard deviation
        boundary_mu = np.average(boundarySamples, 0, weights=weights)
        boundary_var = np.average((np.array(boundarySamples) - np.array([boundary_mu]*n_samples))**2, 0, weights=weights)
        boundary_sig = np.sqrt(boundary_var)
        
        self.boundaryDistribution['U_mu_grid'] = boundary_mu
        self.boundaryDistribution['U_sig_grid'] = boundary_sig
        
        # For the CMEs
        # Loop over each parameter, getting new mean and standard deviation
        for column in ['t', 'lon', 'lat', 'width', 'thickness', 'speed']:
            [w * cmeSample[column] for w, cmeSample in zip(weights, cmeSamples)]
            
            all_samples = np.array([cmeSample[column].to_numpy() for cmeSample in cmeSamples])
            column_mu = np.average(all_samples, 0, weights=weights)
            
            column_var = np.average((all_samples - np.array([column_mu]*n_samples))**2, 0, weights=weights)
            column_sig = np.sqrt(column_var)
            
            self.cmeDistribution[column+'_mu'] = column_mu
            self.cmeDistribution[column+'_sig'] = column_sig
        
        return

if __name__ == '__main__':
    # =========================================================================
    # THIS SHOULD ALL BE MOVED TO A NOTEBOOK WHEN WORKING!
    # =========================================================================
    import copy
    
    # %%========================================================================
    # Initialize an MSIR inputs object
    # =========================================================================
    start = datetime.datetime(2012, 6, 1)
    stop = datetime.datetime(2012, 7, 1)
    rmax = 10
    latmax = 30
    
    inputs = msir_inputs(start, stop, rmax=rmax, latmax=latmax)
    # test_sta = copy.deepcopy(test)
    # test_stb = copy.deepcopy(test)
    
    # %%=============================================================================
    # Search for available background SW and transient data
    # =============================================================================
    inputs.get_availableBackgroundData()
    inputs.filter_availableBackgroundData()
    # inputs.sort_availableSources('rad_HGI')
    
    # Get ICME/IPS data for all available source
    inputs.get_availableTransientData()
    
    # %%=============================================================================
    # Generate background and boundary distributions:
    #   - Remove ICMEs
    #   - GP interpolate 1D in-situ time series
    #   - Backmap to 21.5 RS
    #   - GP interpolate 3D (time, lon, lat) source model
    # =============================================================================
    nSamples = 50
    
    # inputs.remove_Transients()
    
    inputs.generate_backgroundDistributions()
    
    breakpoint()
    inputs.make_boundaryDistributions()
    
    # Either choose one boundary distribution, or do a 3D GP interpolation
    inputs.make_boundaryDistribution3D()
    inputs.sample_boundaryDistribution(where=None)
    
    
    breakpoint()
    
    # # Generate an input background solar wind boundary distribution
    # OMNI
    omni = test.get_omni()
    omni_icmes = queryDONKI.ICME(test.simstart, test.simstop, 
                                 location = 'Earth', duration = 2*u.day)
    test.generate_backgroundDistribution(insitu=omni, icmes=omni_icmes)
    test.generate_boundaryDistribution(nSamples, constant_sig=0)
    
    fig, ax = plt.subplots()
    ax.imshow(test.boundaryDistribution['U_mu_grid'])
    ax.set(title = 'OMNI-based, weighted forward-backward mapping')
    plt.show()
    
    # STEREOA
    sta = test.get_stereoa()
    sta_icmes = queryDONKI.ICME(test_sta.simstart, test_sta.simstop, 
                                location = 'STEREO%20A', duration = 2*u.day)
    test_sta.generate_backgroundDistribution(insitu=sta, icmes=sta_icmes)
    test_sta.generate_boundaryDistribution(nSamples, constant_sig=0)
    
    fig, ax = plt.subplots()
    ax.imshow(test_sta.boundaryDistribution['U_mu_grid'])
    ax.set(title = 'STEREO-A-based, weighted forward-backward mapping')
    plt.show()
    
    # STEREOB
    stb = test.get_stereob()
    stb_icmes = queryDONKI.ICME(test_stb.simstart, test_stb.simstop, 
                                location = 'STEREO%20B', duration = 2*u.day)
    test_stb.generate_backgroundDistribution(insitu=stb, icmes=stb_icmes)
    test_stb.generate_boundaryDistribution(nSamples, constant_sig=0)
    
    fig, ax = plt.subplots()
    ax.imshow(test_stb.boundaryDistribution['U_mu_grid'])
    ax.set(title = 'STEREO-B-based, weighted forward-backward mapping')
    plt.show()
    
    # for i_time in ...
    itime = 50
    fig, ax = plt.subplots()
    # ax.set(xlim = [0, 360], xlabel = 'Heliolongitude [HGI]', 
    #        ylim = [-20, 20], ylabel = 'Heliolatitude [HGI]')
    
    # Get longitude slice (at one time) and corresponding latitude
    all_t = test.boundaryDistribution['t_grid'][itime]
   
    omni_val = test.boundaryDistribution['U_mu_grid'][:, itime] # longitude slice, one time
    omni_lon = np.linspace(0, 360, *omni_val.shape)
    omni_lat = np.full_like(omni_lon, omni.query('mjd == @all_t')['lat_HGI'])
    
    ax.scatter(omni_lon, omni_lat, label = 'OMNI')
    
    # Get longitude slice (at one time) and corresponding latitude
    sta_val = test_sta.boundaryDistribution['U_mu_grid'][:, itime] # longitude slice, one time
    sta_lon = np.linspace(0, 360, *sta_val.shape)
    sta_lat = np.full_like(sta_lon, sta.query('mjd == @all_t')['lat_HGI'])
    
    
    ax.scatter(sta_lon, sta_lat, label = 'STEREO A')
    
    # Get longitude slice (at one time) and corresponding latitude
    stb_val = test_stb.boundaryDistribution['U_mu_grid'][:, itime] # longitude slice, one time
    stb_lon = np.linspace(0, 360, *stb_val.shape)
    stb_lat = np.full_like(stb_lon, stb.query('mjd == @all_t')['lat_HGI'])
    
    
    ax.scatter(stb_lon, stb_lat, label = 'STEREO B')
    
    ax.legend()
    plt.show()
    
    
    import gpflow
    import tensorflow as tf

    xlon = np.array([*omni_lon, *sta_lon, ])#*stb_lon])
    xlat = np.array([*omni_lat, *sta_lat, ])#*stb_lat])
    y = np.array([*omni_val, *sta_val, ])#*stb_val])
    
    X = np.array([[i,j] for i, j in zip(xlon, xlat)])
    Y = y[:, None]
    
    model = gpflow.models.GPR((X, Y),
                              kernel=gpflow.kernels.RBF(),
                              )
    
    n_grid = 100
    _, (ax_mean, ax_std) = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
    
    Xplot1, Xplot2 = np.meshgrid(np.linspace(0, 360, n_grid), np.linspace(-20, 20, n_grid))
    Xplot = np.stack([Xplot1, Xplot2], axis=-1)
    Xplot = Xplot.reshape([n_grid ** 2, 2])
    
    # iv = getattr(model, "inducing_variable", None)
    # # Do not optimize inducing variables, so that we can better see the impact their choice has. When solving
    # # a real problem you should generally optimise your inducing points.
    # if iv is not None:
    #     gpflow.set_trainable(iv, False)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    
    y_mean, y_var = model.predict_y(Xplot)
    y_mean = y_mean.numpy()
    y_std = tf.sqrt(y_var).numpy()
    
    mean_view = ax_mean.pcolor(Xplot1, Xplot2, y_mean.reshape(Xplot1.shape))
    std_view = ax_std.pcolor(Xplot1, Xplot2, y_std.reshape(Xplot1.shape))
    # ax_mean.scatter(X[:, 0], X[:, 1], marker='x', s=1, c="black", )
    # ax_std.scatter(X[:, 0], X[:, 1], marker='.', s=1, c="black",)
    ax_mean.scatter(omni_lon, omni_lat, 
                    marker='.', s=4, label='OMNI')
    ax_mean.scatter(sta_lon, sta_lat, 
                    marker='.', s=4, label='STEREO A')
    ax_mean.scatter(stb_lon, stb_lat, 
                    marker='.', s=4, label='STEREO B')
    
    ax_mean.legend(loc = 'upper right', markerscale=5)
    fig.colorbar(mean_view)
    fig.colorbar(std_view)
    
    # # Also plot the inducing variables if possible:
    # if iv is not None:
    #     ax_mean.scatter(iv.Z[:, 0], iv.Z[:, 1], marker="x", color="red")

    breakpoint()
    
    # # Generate an input CME distribution
    test.generate_cmeDistribution()
    
    # Read all the available assimilation data
    test.get_availableData()
    test.filter_availableData()
    test.sort_availableSources('rad_HGI')
    
    # Save copies of the original distributions, for comparisons
    original_boundaryDistribution = copy.deepcopy(test.boundaryDistribution)
    original_cmeDistribution = copy.deepcopy(test.cmeDistribution)
    
    # Initial weights
    weights = [1/nSamples]*nSamples
    
    # Propagate an ensemble to each source
    for source in test.availableSources:
        
        # Step 1: (re)sample
        boundarySamples, cmeSamples = test.sample(weights)
        
        # Step 2: forecast the solar wind
        # Propagate the sample inputs to this source
        ensemble = test.predict2(boundarySamples, cmeSamples, source)
        #!!!!
        
        # Step 3: get the new weights
        # Calculate the new weights (goodness-of-fit) for each sample to the source
        new_weights = test.update_weights(ensemble, weights, 10, test.availableData[source])
        
        # Step 4: estimate new background distributions from new weights
        test.update_distributions(new_weights, boundarySamples, cmeSamples)
        
        weights = new_weights

    
        