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
import pickle
import tqdm

import sys
sys.path.append('/Users/mrutala/projects/OHTransients/HUXt/code/')
sys.path.append('/Users/mrutala/projects/OHTransients/code/')
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
# import huxt_atObserver as hao
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
        return ((self.radmax / self.usw_minimum).to(u.day), 27 * u.day)
    
    @property 
    def simstart(self):
        return self.start - datetime.timedelta(days=self.simpadding[0].to(u.day).value)
    
    @property 
    def simstop(self):
        return self.stop + datetime.timedelta(days=self.simpadding[1].to(u.day).value)
    
    @property
    def simstarttime(self):
        return self.starttime - self.simpadding[0]
    
    @property
    def simstoptime(self):
        return self.stoptime + self.simpadding[1]
    
    @property
    def availableSources(self):
        availableSources = set(self.availableBackgroundData.columns.get_level_values(0))
        availableSources = set(availableSources) - {'mjd'}
        return sorted(availableSources)
        
    @property
    def boundarySources(self):
        boundarySources = set(self.availableTransientData['affiliated_source'])
        return sorted(boundarySources)
    
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
        
        # self.availableSources.extend(available_sources)
        self.availableBackgroundData = available_data_df
            
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
            out_of_range = (np.abs(self.availableBackgroundData[(source, 'lat_HGI')]) > np.abs(self.latmax)) &\
                           (self.availableBackgroundData[(source, 'rad_HGI')] > self.radmax)
            
            # Set these as NaNs
            self.availableBackgroundData.loc[out_of_range, source] = np.nan
            
            # If no data is in range, delete the source and columns entirely
            if out_of_range.all() == True:
                sources_to_remove.append(source)
                self.availableBackgroundData.drop(columns = source, level = 0, inplace = True)
                          
        # for source in sources_to_remove:
        #     self.availableSources.remove(source)
            
        return
    
    # def sort_availableSources(self, column='rad_HGI'):
        
    #     mean_vals = {}
    #     for source in self.availableSources:
    #         mean_vals[source] = np.mean(self.availableBackgroundData[(source, column)])
            
    #     order = np.argsort(list(mean_vals.values()))
    #     self.availableSources = np.array(list(mean_vals.keys()))[order]
        
    #     return
    
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
                                    duration = duration,
                                    ensureCME=False)
            icmes['affiliated_source'] = source
            
            availableTransientData_list.append(icmes)
        
        availableTransientData_df = pd.concat(availableTransientData_list, axis='rows')
        availableTransientData_df['mjd'] = Time(availableTransientData_df['eventTime'])
        availableTransientData_df.reset_index(inplace=True, drop=True)

        self.availableTransientData = availableTransientData_df 
        
        return
    
    def get_indexICME(self, source):
        
        icme_buffer = 0.5 * u.day
        interp_buffer = 1 * u.day
        
        # Get the insitu data + mjd at this source
        insitu = self.availableBackgroundData[source]
        insitu['mjd'] = self.availableBackgroundData['mjd']
        
        # Get the list of ICMEs at this source
        icmes = self.availableTransientData.query('affiliated_source == @source')
        icmes.reset_index(inplace=True, drop=True)
        
        # Remove ICMEs from OMNI data, leaving NaNs behind
        if 'eventTime' in icmes.columns: 
            icmes = icmes.rename(columns = {'eventTime': 'Shock_time'})
            icmes['ICME_end'] = [row['Shock_time'] + datetime.timedelta(days=(row['duration'])) for _, row in icmes.iterrows()]
        
        
        insitu_noicme = Hin.remove_ICMEs(insitu, icmes, 
                                         params=['U'], 
                                         interpolate = False, 
                                         icme_buffer = icme_buffer, 
                                         interp_buffer = interp_buffer, 
                                         fill_vals = np.nan)
        
        return insitu_noicme['U'].isna()
    
    def generate_backgroundDistributions(self, insitu=None, icmes=None):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        import gpflow
        from sklearn.cluster import KMeans
        
        max_mjd_scale = 100 # mjd will be rescaled between [0, max_mjd_scale]
        period = 27 * u.day
        # n_samples = 10
        average_cluster_span = 6 * u.hour # For K-means clustering to reduce data 
        
        # Calculate the span from stop - start
        span = self.simstop - self.simstart
        
        backgroundDistributions_dict = {}
        for source in self.boundarySources:
            
            # Where an ICME is present, set U, BR to NaN
            indexICME = self.get_indexICME(source)
            insitu_noICME = self.availableBackgroundData[source]
            insitu_noICME['mjd'] = self.availableBackgroundData['mjd']
            insitu_noICME.loc[indexICME, ['U', 'BR']] = np.nan
            
            # Get the mjd and U as column vectors for GPflow
            mjd = insitu_noICME.dropna(axis='index', how='any')['mjd'].to_numpy(float)[:, None]
            U = insitu_noICME.dropna(axis='index', how='any')['U'].to_numpy(float)[:, None]
    
            # Scale the MJD (abscissa) and U (ordinate) for GP imputation
            mjd_rescaler = MinMaxScaler((0, max_mjd_scale))
            mjd_rescaler.fit(mjd)
            X = mjd_rescaler.transform(mjd)
    
            U_rescaler = StandardScaler()
            U_rescaler.fit(U)
            Y = U_rescaler.transform(U)
    
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
            Xo = self.availableBackgroundData['mjd'].to_numpy(float)[:, None]
            Xoc = mjd_rescaler.transform(Xo)
            
            Yoc_mu, Yoc_var = model.predict_y(Xoc)
            Yoc_mu, Yoc_var = np.array(Yoc_mu), np.array(Yoc_var)
            Yoc_sig = np.sqrt(Yoc_var)
            
            # foc = model.predict_f_samples(Xoc, n_samples, full_cov=False)
            # foc = np.array(foc)
            
            Xo = mjd_rescaler.inverse_transform(Xoc)
            Yo_mu = U_rescaler.inverse_transform(Yoc_mu)
            Yo_sig = Yoc_sig * U_rescaler.scale_
            # fo_samples = np.array([speed_rescaler.inverse_transform(f) for f in foc])
            
            new_insitu = insitu_noICME.copy(deep=True)
            
            new_insitu['U_mu'] = Yo_mu.ravel()
            new_insitu['U_sig'] = Yo_sig.ravel()
            new_insitu.drop(columns='U', inplace=True)
            
            # self.backgroundDistribution = new_insitu
            backgroundDistributions_dict[source] = new_insitu
            
        backgroundDistributions = pd.concat(backgroundDistributions_dict,
                                            axis='columns')
        self.backgroundDistributions = backgroundDistributions
        self.backgroundDistributions['mjd'] = self.availableBackgroundData['mjd']
        
        # =============================================================================
        # Visualization
        # =============================================================================
        fig, axs = plt.subplots(nrows=len(self.boundarySources), sharex=True, sharey=True,
                                figsize=(6, 4.5))
        plt.subplots_adjust(bottom=(0.16), left=(0.12), top=(1-0.08), right=(1-0.06),
                            hspace=0)
        
        for ax, source in zip(axs, self.boundarySources):
            
            ax.scatter(self.availableBackgroundData['mjd'], self.availableBackgroundData[(source, 'U')],
                       color='black', marker='.', s=2, zorder=3,
                       label = 'Raw Data')
            # ax.scatter(self.availableBackgroundData.loc[indexICME, 'mjd'], 
            #            self.availableBackgroundData.loc[indexICME, (source, 'U')],
            #            edgecolor='xkcd:scarlet', marker='o', s=6, zorder=2, facecolor='None', lw=0.5,
            #            label = 'ICMEs from DONKI')
            
            indexICME = self.get_indexICME(source)
            onlyICMEs = self.availableBackgroundData.copy(deep=True)
            onlyICMEs.loc[~indexICME, :] = np.nan
            # ax.plot(onlyICMEs['mjd'], 
            #         onlyICMEs[(source, 'U')],
            #         color='xkcd:ruby', zorder=2, lw=2,
            #         label = 'DONKI ICMEs')
            ax.scatter(onlyICMEs['mjd'], 
                       onlyICMEs[(source, 'U')],
                       color='xkcd:bright blue', zorder=3, marker='x', s=4, lw=1,
                       label = 'DONKI ICMEs')
            
            #ax.scatter(Xc, Yc, label='Inducing Points', color='C1', marker='o', s=6, zorder=4)
        
            ax.plot(self.backgroundDistributions['mjd'], 
                    self.backgroundDistributions[(source, 'U_mu')], 
                    label="GP Prediction", color='xkcd:pumpkin', lw=1.5, zorder=3)
            ax.fill_between(
                    self.backgroundDistributions['mjd'],
                    (self.backgroundDistributions[(source, 'U_mu')] - 1.96 * self.backgroundDistributions[(source, 'U_sig')]),
                    (self.backgroundDistributions[(source, 'U_mu')] + 1.96 * self.backgroundDistributions[(source, 'U_sig')]),
                    alpha=0.33, color='xkcd:pumpkin',
                    label=r"95% CI", zorder=0)
            
            # for fo_sample in fo_samples:
            #     ax.plot(Xo.ravel(), fo_sample.ravel(), lw=1, color='C3', alpha=0.2, zorder=-1)
            # ax.plot(Xo.ravel()[0:1], fo_sample.ravel()[0:1], lw=1, color='C3', alpha=1, 
            #         label = 'Samples about Mean')
        
            # ax.legend(scatterpoints=3, loc='upper right')
            
            ax.grid(True, which='major', axis='x',
                    color='black', ls=':', alpha=0.5)
            ax.annotate(source, (0, 1), (1,-1), 
                        xycoords='axes fraction', textcoords='offset fontsize',
                        ha='left', va='top',
                        color='xkcd:black')
        
        axs[0].legend(loc='lower left', bbox_to_anchor=(0., 1.05, 1.0, 0.1),
                      ncols=4, mode="expand", borderaxespad=0.,
                      scatterpoints=3, markerscale=2)
        ax.set(xlim=[self.starttime.mjd, self.stoptime.mjd],
               ylim=[250, 850])
        ax.secondary_xaxis(-0.23, 
                           functions=(lambda x: x-self.starttime.mjd, lambda x: x+self.starttime.mjd))
        
        fig.supxlabel('Date [MJD]; Days from {}'.format(datetime.datetime.strftime(self.start, '%Y-%m-%d %H:%M')))
        fig.supylabel('Solar Wind Speed [km/s]')
        
        plt.show()
        
        return
    
    def generate_boundaryDistributions(self, nSamples=16, constant_sig=0):
        from tqdm import tqdm
        # from dask.distributed import Client, as_completed, LocalCluster
        import multiprocessing as mp
        import logging
        from joblib import Parallel, delayed
        from tqdm import tqdm
        
        nCores = int(0.75 * mp.cpu_count()) 

        rng = np.random.default_rng()
        
        methodOptions = ['forward', 'back', 'both']
        # methodOptions = ['both']
        
        # Define an inner function to be run in parallel
        def map_vBoundaryInwards(insitu_df, corot_type):
            
            # Generate the Carrington grids
            t, vcarr, bcarr = Hin.generate_vCarr_from_OMNI(self.simstart, self.simstop, 
                                                           omni_input=insitu_df, corot_type=corot_type)
            
            # Map to 210 solar radii, then to the inner boundary for the model
            vcarr_210 = vcarr.copy()
            vcarr_21p5 = vcarr.copy()
            for i, _ in enumerate(t):
                
                # vcarr_210[:,i] = Hin.map_v_boundary_inwards(vcarr[:,i]*u.km/u.s, 
                #                                             (insitu_df['rad_HGI'][i]*u.AU).to(u.solRad), 
                #                                             210 * u.solRad)
                
                # vcarr_21p5[:,i] = Hin.map_v_boundary_inwards(vcarr_210[:,i]*u.km/u.s,
                #                                              210 * u.solRad,
                #                                              self.innerbound)
                
                vcarr_21p5[:,i] = Hin.map_v_boundary_inwards(vcarr[:,i]*u.km/u.s,
                                                             (insitu_df['rad_HGI'][i]*u.AU).to(u.solRad),
                                                             self.innerbound)
                
                
            return vcarr_21p5
        
        
        boundaryDistributions_dict = {}
        for source in self.boundarySources:
        
        # Format the insitu df (backgroundDistribution) as HUXt expects it
            insitu_df = self.backgroundDistributions[source].copy(deep=True)
            insitu_df['BX_GSE'] =  -insitu_df['BR']
            insitu_df['V'] = insitu_df['U_mu']
            insitu_df['datetime'] = insitu_df.index
            insitu_df = insitu_df.reset_index()
        
            # Map inwards once to get the appropriate dimensions, etc.
            t, vcarr, bcarr = Hin.generate_vCarr_from_OMNI(self.simstart, self.simstop, omni_input=insitu_df)
        
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
        
            # Calculate boundary distributions by backmapping each sample
            def process_sample(U_sample, method_sample):
                insitu_df_copy = insitu_df.copy(deep=True)
                insitu_df_copy['V'] = U_sample
                return map_vBoundaryInwards(insitu_df_copy, method_sample)
            
            # Sample the velocity distribution and assign random mapping directions (method)
            # Randomly assigning these is equivalent to performing each mapping for each sample (for large numbers of samples)
            # Having a single random population should be better mathematically
            uSamples = [rng.normal(loc=insitu_df['U_mu'], 
                                   scale=insitu_df['U_sig']
                                   ) for _ in range(nSamples)]
            
            methodSamples = rng.choice(methodOptions, nSamples)
            
            vcarrGenerator = Parallel(return_as='generator', n_jobs=nCores)(
                delayed(process_sample)(u, method) 
                for u, method in zip(uSamples, methodSamples)
                )
            vcarrSamples = list(tqdm(vcarrGenerator, total=nSamples))
    
            # Characterize the resulting samples as one distribution
            vcarr_mu = np.mean(vcarrSamples, axis=0)
            vcarr_sig = np.sqrt(np.std(vcarrSamples, axis=0)**2 + constant_sig**2)
        
            boundaryDistributions_dict[source] = {'t_grid': t,
                                                  'U_mu_grid': vcarr_mu,
                                                  'U_sig_grid': vcarr_sig,
                                                  'B_grid': bcarr}
        
        self.boundaryDistributions = boundaryDistributions_dict
        
        # # =============================================================================
        # # Visualization 
        # # =============================================================================
        # fig, axs = plt.subplots(figsize=(6,4.5), ncols=2)
        
        # mu_img = axs[0].imshow(vcarr_mu, 
        #                        extent=[self.simstarttime.mjd, self.simstoptime.mjd, 0, 360], 
        #                        origin='lower', aspect=0.2)
        # axs[0].set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        # fig.colorbar(mu_img, ax=axs[0])
        
        # sig_img = axs[1].imshow(vcarr_sig, 
        #                         extent=[self.simstarttime.mjd, self.simstoptime.mjd, 0, 360], 
        #                         origin='lower', aspect=0.2)
        # axs[1].set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        # fig.colorbar(sig_img, ax=axs[1])
        
        # axs[0].set(ylabel='Heliolongitude [deg.]', xlabel='Date [MJD]')
        # axs[1].set(xlabel='Date [MJD]')
        
        # plt.show()
        
        return
    
    def generate_boundaryDistribution3D(self):
        import gpflow
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        # from scipy.cluster.vq import kmeans
        from sklearn.cluster import KMeans
        import multiprocessing as mp
        from joblib import Parallel, delayed
        
        # Get dimensions from OMNI boundary distribution, which *must* exist
        nLon, nTime = self.boundaryDistributions['omni']['U_mu_grid'].shape
        nLat = int(nLon/4)
        
        # Coordinates = (lon, time, lat)
        # Values = boundary speed, magnetic field* (*not implemented fully)
        lat_for3d = np.linspace(-self.latmax.value, self.latmax.value, nLat)
        lon_for3d = np.linspace(0, 360, nLon)
        mjd_for3d = self.boundaryDistributions['omni']['t_grid']
        # U_mu_3d = np.zeros((nLat, nLon, nTime))
        # U_sigma_3d = np.zeros((nLat, nLon, nTime))
        
        # Setup Normalizations ahead of time
        lat_scaler = MinMaxScaler((0,1))
        lat_scaler.fit(lat_for3d[:,None])
        
        lon_scaler = MinMaxScaler((0,1))
        lon_scaler.fit(lon_for3d[:,None])
        
        mjd_scaler = MinMaxScaler((0,1))
        mjd_scaler.fit(self.boundaryDistributions['omni']['t_grid'][:,None])

        # All of the backmapped values
        val_scaler = StandardScaler()
        val_scaler.fit(np.array([v['U_mu_grid'] 
                                 for _, v in self.boundaryDistributions.items()
                                 ]).flatten()[:,None])
        
        rng = np.random.default_rng()
        # for i in range(nTime):
        lat, lon, mjd, val_mu, val_sigma, = [], [], [], [], []
        # lat, lon, mjd, val, = [], [], [], []
        for source in self.boundarySources:
            
            # # lon does not vary with time
            # lon_1d = np.linspace(0, 360, nLon)
            # # lon_2d = np.tile(lon_1d, (n))
            
            # # lat does vary with time, so fill an array to match lon
            # lat_0d = np.interp(self.boundaryDistributions[source]['t_grid'][i],
            #                    self.backgroundDistributions[source]['mjd'],
            #                    self.backgroundDistributions[source]['lat_HGI'])
            # lat_1d = np.full_like(lon_1d, lat_0d)
            
            # mjd_0d = self.boundaryDistributions[source]['t_grid'][i]
            
            # lon.extend(lon_1d.flatten())
            # lat.extend(lat_1d.flatten())
            # val_1d = self.boundaryDistributions[source]['U_mu_grid'][:, i]
            # val.extend(val_1d)
            
            lon_1d = np.linspace(0, 360, nLon)
            mjd_1d = self.boundaryDistributions[source]['t_grid']
            lat_1d = np.interp(self.boundaryDistributions[source]['t_grid'],
                               self.backgroundDistributions[source]['mjd'],
                               self.backgroundDistributions[source]['lat_HGI'])
            
            mjd_2d, lon_2d, = np.meshgrid(mjd_1d, lon_1d)
            lat_2d = np.tile(lat_1d, (128, 1))
            
            val_mu_2d = self.boundaryDistributions[source]['U_mu_grid']
            val_sigma_2d = self.boundaryDistributions[source]['U_sig_grid']
            
            lat.extend(lat_2d.flatten())
            lon.extend(lon_2d.flatten())
            mjd.extend(mjd_2d.flatten())
            
            val_mu.extend(val_mu_2d.flatten())
            val_sigma.extend(val_sigma_2d.flatten())
            
            # sample from val sample, rather than val mu?
            # val.extend(rng.normal(loc=val_mu_2d.flatten(), 
            #                       scale=val_sigma_2d.flatten()))
            
        # Recast as arrays
        lon = np.array(lon)
        lat = np.array(lat)
        mjd = np.array(mjd)
        val_mu = np.array(val_mu)
        val_sigma = np.array(val_sigma)
        # val = np.array(val)
        
        # Normalizations & NaN removal
        xlat = lat_scaler.transform(lat[~np.isnan(val_mu),None])
        xlon = lon_scaler.transform(lon[~np.isnan(val_mu),None])
        xmjd = mjd_scaler.transform(mjd[~np.isnan(val_mu),None])
        
        yval_mu = val_scaler.transform(val_mu[~np.isnan(val_mu),None])
        yval_sigma = (1/val_scaler.scale_) * val_sigma[~np.isnan(val_sigma),None]
        
        # xlon = xlon[~np.isnan(yval)]
        # xlat = xlat[~np.isnan(yval)]
        # yval = yval[~np.isnan(yval)][:,None]
        
        # Find XY clusters to reduce number of points in GP
        n_clusters = int(0.05 * len(yval_mu))
        
        X = np.column_stack([xlat, xlon, xmjd])
        Y_mu = yval_mu
        Y_sigma = yval_sigma
        
        XY = np.column_stack([X, Y_mu, Y_sigma])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(XY)
        XYc = kmeans.cluster_centers_
        # Xc, Yc = XYc[:,0:3], XYc[:,3][:,None]
        Xc, Yc_mu, Yc_sigma = XYc[:,:3], XYc[:,3][:,None], XYc[:,4][:,None] # 2D Y
        
        # kernel = gpflow.kernels.RBF()
        
        lat_kernel = gpflow.kernels.RBF(active_dims=[0])
        lon_kernel = gpflow.kernels.Periodic(gpflow.kernels.RBF(active_dims=[1]), 
                                             period=gpflow.Parameter(1, trainable=False))
        mjd_kernel = gpflow.kernels.RBF(active_dims=[2])
        all_kernel = gpflow.kernels.RBF()
        kernel_mu = (lat_kernel + lon_kernel + mjd_kernel + 
                     lat_kernel*lon_kernel + lat_kernel*mjd_kernel + lon_kernel*mjd_kernel +
                     all_kernel)
        # import copy
        kernel_sigma = copy.deepcopy(kernel_mu)
        
        model_mu = gpflow.models.GPR((Xc, Yc_mu),
                                     kernel=kernel_mu,
                                     noise_variance=1.0)
        opt_mu = gpflow.optimizers.Scipy()
        opt_mu.minimize(model_mu.training_loss, model_mu.trainable_variables)
        
        model_sigma = gpflow.models.GPR((Xc, Yc_sigma),
                                        kernel=kernel_sigma,
                                        noise_variance=1.0)
        opt_sigma = gpflow.optimizers.Scipy()
        opt_sigma.minimize(model_sigma.training_loss, model_sigma.trainable_variables)
        
        # model = gpflow.models.GPR((Xc, Yc),
        #                           kernel=kernel,
        #                           likelihood=gpflow.likelihoods.Gaussian(scale=gpflow.functions.Polynomial(degree=2)),
        #                           )
        
        # model = gpflow.models.GPR((Xc, Yc),
        #                           kernel=kernel)
        
        
        
        Xlat, Xlon, Xmjd = np.meshgrid(lat_scaler.transform(lat_for3d[:,None]),
                                       lon_scaler.transform(lon_for3d[:,None]), 
                                       mjd_scaler.transform(mjd_for3d[:,None]),
                                       indexing='ij')
        X3d = np.column_stack([Xlat.flatten()[:,None],
                               Xlon.flatten()[:,None],
                               Xmjd.flatten()[:,None]])
        
        # def chunk(iterable, size):
        #     return [iterable[pos:pos + size] for pos in range(0, len(iterable), size)]
        # X3d_chunks = chunk(X3d, chunksize)
        # for X3d_chunk in X3d_chunks:
        #     Ymu, Ysigma2 = model.predict_y(X3d)
        
        # Ymu, Ysigma = np.zeros((len(X3d),1)), np.zeros((len(X3d),1))
        # for pos in tqdm.tqdm(range(0, len(X3d), chunksize)):
            
        #     X3d_chunk = X3d[pos:pos + chunksize]
            
        #     Ymu_chunk, Ysigma2_chunk = model.predict_y(X3d_chunk)
        #     Ymu[pos:pos + chunksize] = Ymu_chunk
        #     Ysigma[pos:pos + chunksize] = tf.sqrt(Ysigma2_chunk)
        
        # Parallel chunk processing 
        nCores = int(0.75 * mp.cpu_count()) 
        chunksize = 5000
        X3d_chunks = [X3d[pos:pos + chunksize] for pos in range(0, len(X3d), chunksize)]
        
        def process(model, X3d_chunk):
            Ymu_chunk, Ysigma2_chunk = model.predict_y(X3d_chunk)
            
            # Ymumu, Ymusigma2 = Ymu_chunk[:,0], Ymu_chunk[:,1]
            # Ymusigma = tf.sqrt(Ymusigma2)
            
            # Ysigmamu, Ysigmasigma2 = Ysigma_chunk[:,0], Ysigma_chunk[:,1]
            # Ysigmasigma = tf.sqrt(Ysigmasigma2)
            
            # return Ymumu, Ymusigma, Ysigmamu, Ysigmasigma
            
            return Ymu_chunk, tf.sqrt(Ysigma2_chunk)
        
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", UserWarning)
        import warnings
        warnings.filterwarnings('ignore') 
        
        mu_generator = Parallel(return_as='generator', n_jobs=nCores)(
            delayed(process)(model_mu, X3d_chunk) 
            for X3d_chunk in X3d_chunks
            )
        Ymu_chunklist = list(tqdm.tqdm(mu_generator, total=len(X3d_chunks)))
        
        sigma_generator = Parallel(return_as='generator', n_jobs=nCores)(
            delayed(process)(model_sigma, X3d_chunk) 
            for X3d_chunk in X3d_chunks
            )
        Ysigma_chunklist = list(tqdm.tqdm(sigma_generator, total=len(X3d_chunks)))
        
        Ymu = np.concatenate([elem[0] for elem in Ymu_chunklist], axis=0)
        Ymusigma = np.concatenate([elem[1] for elem in Ymu_chunklist], axis=0)
        Ysigma = np.concatenate([elem[0] for elem in Ysigma_chunklist], axis=0)
        
        # Ymumu = np.concatenate([elem[0] for elem in Y_chunklist], axis=0)
        # Ymusigma = np.concatenate([elem[1] for elem in Y_chunklist], axis=0)
        # Ysigmamu = np.concatenate([elem[2] for elem in Y_chunklist], axis=0)
        # Ysigmasigma = np.concatenate([elem[3] for elem in Y_chunklist], axis=0)
            
        U_mu = val_scaler.inverse_transform(Ymu)
        U_sigma = val_scaler.scale_ * np.sqrt(Ysigma**2 + Ymusigma**2)
        
        U_mu_3d = U_mu.reshape(nLat, nLon, nTime)
        U_sigma_3d = U_sigma.reshape(nLat, nLon, nTime)
        
        # Generate an OBVIOUSLY WRONG B
        B_grid_3d = np.tile(self.boundaryDistributions['omni']['B_grid'], (64, 1, 1))
        
        self.boundaryDistributions3D = {'t_grid': mjd_for3d,
                                        'lon_grid': lon_for3d,
                                        'lat_grid': lat_for3d,
                                        'U_mu_grid': U_mu_3d,
                                        'U_sig_grid': U_sigma_3d,
                                        'B_grid': B_grid_3d,
                                        }
        
        # =============================================================================
        # Visualization     
        # =============================================================================
    
        # for source in self.boundarySources:
            
        #     # Reconstruct the backmapped solar wind view at each source
        #     fig, axs = plt.subplots(nrows=2)
            
        #     # Get the latitude at each time step
        #     sampleLat = np.interp(self.boundaryDistributions3D['t_grid'], 
        #                           self.backgroundDistributions[(source, 'mjd')],
        #                           self.backgroundDistributions[(source, 'lat_HGI')])
            
        #     # Get the closest index for each of those latitudes
        #     sampleLat_indx = np.interp(sampleLat,
        #                                self.boundaryDistributions3D['lat_grid'],
        #                                np.arange(len(self.boundaryDistributions3D['lat_grid'])))
        #     sampleLat_indx = np.round(sampleLat_indx).astype(int)
            
        #     # Extract the correct latitude indices from the 3D distribution
        #     U_mu_2d = np.zeros(self.boundaryDistributions3D['U_mu_grid'].shape[1:])
        #     for i, t in zip(sampleLat_indx, np.arange(0, len(self.boundaryDistributions3D['t_grid']))):
        #         U_mu_2d[:,t] = self.boundaryDistributions3D['U_mu_grid'][i,:,t]
                
        #     axs[0].imshow(self.boundaryDistributions[source]['U_mu_grid'],
        #                   vmin=200, vmax=600)
            
        #     _ = axs[1].imshow(U_mu_2d,
        #                       vmin=200, vmax=600)
            
        #     fig.suptitle(source)
        #     plt.colorbar(_)
            
        #     plt.show()
                
            
        # breakpoint()
    
        return
    
    def sample_boundaryDistribution3D(self, at=None):
        
        # !!!! Catch exceptions better...
        if at not in self.availableSources:
            breakpoint()
            
        shape3D = self.boundaryDistributions3D['U_mu_grid'].shape
            
        # Get the HGI latitude of the target at times matching the boundary grid
        target_lats = np.interp(self.boundaryDistributions3D['t_grid'],
                                self.availableBackgroundData['mjd'],
                                self.availableBackgroundData[(at, 'lat_HGI')])
        
        # Construct a 2D boundary 
        U_mu_2d = np.zeros(shape3D[1:])
        U_sigma_2d = np.zeros(shape3D[1:])
        
        for i, target_lat in enumerate(target_lats):
            # Locate the nearest index to the target_lat
            # target_lat_indx = np.interp(target_lat,
            #                             self.boundaryDistributions3D['lat_grid'],
            #                             np.arange(shape3D[0]))
            # target_lat_indx = np.round(target_lat_indx).astype(int)
            
            # Sample the 3D distribution
            # U_mu_2d[:,i] = self.boundaryDistributions3D['U_mu_grid'][target_lat_indx,:,i]
            # U_sigma_2d[:,i] = self.boundaryDistributions3D['U_sig_grid'][target_lat_indx,:,i]
            
            # Sample the 3D distribution by interpolation
            # This prevents edge effects in the boundary
            for j in range(shape3D[1]):
                U_mu_2d[j,i] = np.interp(target_lat, 
                                         self.boundaryDistributions3D['lat_grid'],
                                         self.boundaryDistributions3D['U_mu_grid'][:,j,i])
                U_sigma_2d[j,i] = np.interp(target_lat, 
                                            self.boundaryDistributions3D['lat_grid'],
                                            self.boundaryDistributions3D['U_sig_grid'][:,j,i])
        
        # AGAIN, CLEARLY WRONG B!!!!
        B_grid = self.boundaryDistributions3D['B_grid'][0,:,:]
        
        return {'t_grid': self.boundaryDistributions3D['t_grid'],
                'U_mu_grid': U_mu_2d, 
                'U_sig_grid': U_sigma_2d,
                'B_grid': B_grid}
    
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
    
    def sample3D(self, weights, at='omni'):
        
        n_samples = len(weights)
        
        rng = np.random.default_rng()
        
        # Plain normal samples
        # backgroundSamples = rng.normal(loc=self.backgroundDistribution['u_mu'],
        #                                scale=self.backgroundDistribution['u_sig'])
        
        # Offset normal samples
        boundaryDist = self.sample_boundaryDistribution3D(at)
        # boundaryDist = self.boundaryDistributions[at]
        boundarySamples_U = []
        offsets = rng.normal(loc=0, scale=1, size=n_samples)
        offsets_ratio = 0.1
        for offset in offsets:
            boundarySamples_U.append(rng.normal(loc=boundaryDist['U_mu_grid'] + offsets_ratio*offset*boundaryDist['U_sig_grid'],
                                              scale=(1-offsets_ratio)*boundaryDist['U_sig_grid'],
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
            interp_result = pd.DataFrame(index=self.availableBackgroundData.index,
                                         columns=result.columns)
            for col in interp_result.columns:
                interp_result[col] = np.interp(self.availableBackgroundData['mjd'], result['mjd'], result[col])
                
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
        observer = H.Observer(observer_name, Time(self.boundaryDistributions3D['t_grid'], format='mjd'))
        
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
                                         self.boundaryDistributions3D['t_grid'], 
                                         boundarySample_U,
                                         self.boundaryDistributions3D['B_grid'][0,:,:], 
                                         observer_name, observer,
                                         dpadding = dpadding, 
                                         cme_list = cme_list,
                                         r_min=self.innerbound)
            
            
            futureInterpolated = pd.DataFrame(index=self.availableBackgroundData.index,
                                              columns=future.columns)
            for col in futureInterpolated.columns:
                futureInterpolated[col] = np.interp(self.availableBackgroundData['mjd'], future['mjd'], future[col])
            
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
        
        # Save ensemble
        self.current_ensemble = ensemble
        
        return ensemble
    
    # def score_propagatedInputs(self, ensemble, source):
    #     import numpy as np
    #     breakpoint()
        
    #     fig, ax = plt.subplots()
    #     ax.scatter(self.availableBackgroundData['mjd'], self.availableBackgroundData[(source, 'U')],
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
    #         # w = score_function(self.availableBackgroundData[(source, 'U')], member_interp)
                                                  
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
        metamodel = pd.DataFrame(index = ensemble[0].index)
        ensemble_columns = ensemble[0].columns
        
        for col in ensemble_columns:
            for index in metamodel.index:
                vals = [m.loc[index, col] for m in ensemble]
                valsort_indx = np.argsort(vals)
                cumsum_weights = np.cumsum(np.array(weights)[valsort_indx])
                
                weighted_median = vals[valsort_indx[np.searchsorted(cumsum_weights, 0.5 * cumsum_weights[-1])]]
                weighted_upper95 = vals[valsort_indx[np.searchsorted(cumsum_weights, 0.975 * cumsum_weights[-1])]]
                weighted_lower95 = vals[valsort_indx[np.searchsorted(cumsum_weights, 0.025 * cumsum_weights[-1])]]
                
                if col in ['U', 'BX']:
                    metamodel.loc[index, col+"_median"] = weighted_median
                    metamodel.loc[index, col+"_upper95"] = weighted_upper95
                    metamodel.loc[index, col+"_lower95"] = weighted_lower95
                else:
                    metamodel.loc[index, col] = weighted_median
        
        return metamodel
    
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

# %%
if __name__ == '__main__':
    import generate_external_input
    # =========================================================================
    # THIS SHOULD ALL BE MOVED TO A NOTEBOOK WHEN WORKING!
    # =========================================================================
    import copy
    
    # Checkpoint for testing: skip things you know work already
    skip_to_checkpoint = True

    if not skip_to_checkpoint:
        # ========================================================================
        # Initialize an MSIR inputs object
        # =========================================================================
        start = datetime.datetime(2012, 6, 1)
        stop = datetime.datetime(2012, 7, 1)
        rmax = 10 # AU
        latmax = 15
        
        inputs = msir_inputs(start, stop, rmax=rmax, latmax=latmax)
        # test_sta = copy.deepcopy(test)
        # test_stb = copy.deepcopy(test)
        
        # =============================================================================
        # Search for available background SW and transient data
        # =============================================================================
        inputs.get_availableBackgroundData()
        inputs.filter_availableBackgroundData()
        # inputs.sort_availableSources('rad_HGI')
        
        # Get ICME/IPS data for all available source
        inputs.get_availableTransientData()
        
        # =============================================================================
        # Generate background and boundary distributions:
        #   - Remove ICMEs
        #   - GP interpolate 1D in-situ time series
        #   - Backmap to 21.5 RS
        #   - GP interpolate 3D (time, lon, lat) source model
        # =============================================================================
        
        inputs.generate_backgroundDistributions()
        
        inputs.generate_boundaryDistributions(nSamples=16, constant_sig=0)
        
        # Either choose one boundary distribution, or do a 3D GP interpolation
        inputs.generate_boundaryDistribution3D()
        
        # Generate an input CME distribution
        inputs.generate_cmeDistribution()
        
        # Add Saturn SKR Data
        saturn_df = generate_external_input.Cassini_SKR(inputs.availableBackgroundData.index)
        inputs.availableBackgroundData = pd.merge(inputs.availableBackgroundData, 
                                                  saturn_df,
                                                  left_index=True, right_index=True)
        
        nSamples = 64
        weights = [1/nSamples]*nSamples
        
        # for source in ...
        source = 'saturn'
        
        boundarySamples, cmeSamples = inputs.sample3D(weights, at=source)
        
        ensemble = inputs.predict2(boundarySamples, cmeSamples, source)
        
        # Save as checkpoint
        with open('/Users/mrutala/projects/OHTransients/inputs_checkpoint.pkl', 'wb') as f:
            pickle.dump(inputs, f)
         
    with open('/Users/mrutala/projects/OHTransients/inputs_checkpoint.pkl', 'rb') as f:
        inputs = pickle.load(f)
    
    # CIME interaction time @ Saturn (Palmerio+ 2021)
    interaction_time = datetime.datetime(2012, 6, 12, 00, 00)
    
    # %%=============================================================================
    # PRE X: Plot locations of sources (top-down, heliolatitude)
    # =============================================================================
    fig, axs = plt.subplots(nrows=2, height_ratios=[2.5, 1], figsize=[4, 4.5])
    plt.subplots_adjust(bottom=(0.15), left=(0.16), top=(1-0.135), right=(1-0.04),
                        hspace=0.02)
    
    target_subset = inputs.availableBackgroundData.query("@inputs.start <= index < @inputs.stop")
    
    colors = {'omni': 'xkcd:forest green',
              'stereo a': 'xkcd:ruby',
              'stereo b': 'xkcd:sapphire',
              'saturn': 'xkcd:beige'}
    
    axs[0].scatter([0], [0], marker='o', s=64, color='gold') # The Sun
    for source in inputs.availableSources:
        
        x_polar = target_subset[(source, 'rad_HGI')] * \
            np.cos(np.deg2rad(target_subset[(source, 'lon_HGI')] + 60))
        y_polar = target_subset[(source, 'rad_HGI')] * \
            np.sin(np.deg2rad(target_subset[(source, 'lon_HGI')] + 60))
            
        axs[0].plot(x_polar, y_polar,
                    color=colors[source], lw=1.5, ls=':')
        
        axs[0].scatter(x_polar[-1], y_polar[-1],
                       color=colors[source], marker='o', s=16, label=source)
        
        axs[1].plot(target_subset['mjd'], target_subset[(source, 'lat_HGI')],
                    color=colors[source], lw=1.5)

        
    axs[0].set(xlim=[-10,2], xlabel=r'$X_{HGI}$ [AU]', 
               ylim=[-6,3], ylabel=r'$Y_{HGI}$ [AU]')
    axs[0].tick_params(which='both', top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0].xaxis.set_label_position('top')
    axs[0].annotate("Date: {}".format(target_subset.index[-1].strftime("%Y-%m-%d %H:%M")), (0,1), (1,-1),
                    'axes fraction', 'offset fontsize')
    axs[0].annotate("MJD: {}".format(inputs.stoptime.mjd), (0,1), (1,-2),
                    'axes fraction', 'offset fontsize')
    axs[0].legend(bbox_to_anchor=(0., 1.155, 1., .102), loc='lower left',
                  ncols=4, mode="expand", borderaxespad=0.)   
    
    axs[1].set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd],
               ylabel='Heliolatitude [deg.]')
    axs12 = axs[1].secondary_xaxis(-0.23, 
                functions=(lambda x: x-inputs.starttime.mjd, lambda x: x+inputs.starttime.mjd))
    axs12.set(xlabel = 'MJD; Days from {}'.format(inputs.start.strftime('%Y-%m-%d %H:%M')))
    
    axs[1].axvline(Time(interaction_time).mjd, color='xkcd:fuchsia', ls='--', lw=1.5)
    plt.show()
    
    # %%=============================================================================
    # PRE X: SKR data during region of interest
    # =============================================================================
    fig, axs = plt.subplots(nrows=2, figsize=[4, 4.5], sharex=True, sharey=True)
    plt.subplots_adjust(bottom=(0.15), left=(0.16), top=(1-0.045), right=(1-0.04),
                        hspace=0.1)
    
    axs[0].set(ylim=[0,8e8])
    for source in ['saturn']:
        axs[0].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData[(source, 'P_RH_core')],
                color='red', lw=0.5, zorder=-10, alpha=0.10,  
                label='1-hour')
        axs[0].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData.rolling('10h', center=True).mean()[(source, 'P_RH_core')],
                color='red', lw=1.0, zorder=-10, alpha=1.0,  
                label='10-hour')
        axs[0].legend(loc='upper right')
        
        axs[1].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData[(source, 'P_LH_core')],
                color='black', lw=0.5, zorder=-10, alpha=0.10, 
                label='1-hour')
        
        axs[1].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData.rolling('10h', center=True).mean()[(source, 'P_LH_core')],
                color='black', lw=1.0, zorder=-10, alpha=1.0, 
                label='10-hour')
        axs[1].legend(loc='upper right')
        
    
    axs[0].set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd],
               ylabel = 'Right-hand Polarized')
    axs[1].set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd],
               ylabel = 'Left-hand Polarized')
    
    axs12 = axs[1].secondary_xaxis(-0.115, 
                functions=(lambda x: x-inputs.starttime.mjd, lambda x: x+inputs.starttime.mjd))
    axs12.set(xlabel = 'MJD; Days from {}'.format(inputs.start.strftime('%Y-%m-%d %H:%M')))
    
    fig.supylabel('SKR Power, Integrated (100-400 kHz) [W/sr]', size='medium')
    
    for ax in axs: 
        ax.axvline(Time(interaction_time).mjd, color='xkcd:fuchsia', ls='--', lw=1.5)
    plt.show()
    
    # %% =============================================================================
    # PRE X: Plot normal, non-GP backmapped boundaries
    # =============================================================================
    # Setup stuff
    # Pick a fixed time by index and find the associated MJD
    fixed_time_mjd = Time(interaction_time).mjd
    fixed_time_indx = np.interp(fixed_time_mjd, 
                                inputs.boundaryDistributions3D['t_grid'], 
                                np.arange(len(inputs.boundaryDistributions3D['t_grid'])))
    fixed_time_indx = np.round(fixed_time_indx).astype(int)
   
    
    # What latitude is the source at at this time? In descending order
    fixed_time_lats = {}
    for source in inputs.boundarySources:
        fixed_time_lat = np.interp(fixed_time_mjd,
                                   inputs.availableBackgroundData['mjd'],
                                   inputs.availableBackgroundData[(source, 'lat_HGI')])
        fixed_time_lats[source] = fixed_time_lat
    fixed_time_lats = {k: v for k, v in sorted(fixed_time_lats.items(), key=lambda x: x[1], reverse=True)}
    
    # Actual plot
    fig, axs = plt.subplots(nrows=len(inputs.boundarySources), sharex=True, sharey=True,
                            figsize=(4, 4.5))
    plt.subplots_adjust(bottom=(0.09), left=(0.16), top=(1-0.135), right=(1-0.04),
                        hspace=0)
    
    lon_grid = np.linspace(0, 360, inputs.boundaryDistributions['omni']['U_mu_grid'].shape[0])
    
    for ax, source in zip(axs, fixed_time_lats.keys()):
        
        U_mu = inputs.boundaryDistributions[source]['U_mu_grid'][:,fixed_time_indx]
        U_upper = U_mu + inputs.boundaryDistributions[source]['U_sig_grid'][:,fixed_time_indx]
        U_lower = U_mu - inputs.boundaryDistributions[source]['U_sig_grid'][:,fixed_time_indx]
        
        ax.plot(lon_grid, U_mu,
                color='xkcd:pumpkin', lw=1.5,
                label = 'Backmapped Data')
        ax.fill_between(lon_grid, U_upper, U_lower,
                        color='xkcd:pumpkin', alpha=0.33,
                        label='95% CI')
        bbox = dict(boxstyle="round", fc='white', ec='black',
                    pad=0.2, alpha=0.66)
        ax.annotate(source, (0,1), (1,-1), 
                    'axes fraction', 'offset fontsize',
                    color='black', ha='left', va='top',
                    bbox=bbox)
    
    axs[0].legend(bbox_to_anchor=(0., 1.05, 1., .102), loc='lower left',
                  ncols=1, mode="expand", borderaxespad=0.)   
    axs[0].set(xlim=[0,360], ylim=[250, 700])
    
    fig.supxlabel('Heliolongitude [deg.]')
    fig.supylabel('Solar Wind Speed [km/s]')
        
    plt.show()

    # %% =============================================================================
    # PRE X: Compare 3D boundary distribution to individal backmapped results
    # =============================================================================
    # Remake the last plot, with comparison to 3D GP results
    fig, axs = plt.subplots(nrows=len(inputs.boundarySources), sharex=True, sharey=True,
                            figsize=(4, 4.5))
    plt.subplots_adjust(bottom=(0.09), left=(0.16), top=(1-0.135), right=(1-0.04),
                        hspace=0)
    
    lon_grid = np.linspace(0, 360, inputs.boundaryDistributions['omni']['U_mu_grid'].shape[0])
    
    for ax, source in zip(axs, fixed_time_lats.keys()):
        
        U_mu = inputs.boundaryDistributions[source]['U_mu_grid'][:,fixed_time_indx]
        U_upper = U_mu + inputs.boundaryDistributions[source]['U_sig_grid'][:,fixed_time_indx]
        U_lower = U_mu - inputs.boundaryDistributions[source]['U_sig_grid'][:,fixed_time_indx]
        
        ax.plot(lon_grid, U_mu,
                color='xkcd:pumpkin', lw=1.5,
                label = 'Backmapped Data')
        ax.fill_between(lon_grid, U_upper, U_lower,
                        color='xkcd:pumpkin', alpha=0.33,
                        label='95% CI')
        bbox = dict(boxstyle="round", fc='white', ec='black',
                    pad=0.2, alpha=0.66)
        ax.annotate(source, (0,1), (1,-1), 
                    'axes fraction', 'offset fontsize',
                    color='black', ha='left', va='top',
                    bbox=bbox)
               
        boundary_sample = inputs.sample_boundaryDistribution3D(source)
        U_mu = boundary_sample['U_mu_grid'][:,fixed_time_indx]
        U_upper = U_mu + boundary_sample['U_sig_grid'][:,fixed_time_indx]
        U_lower = U_mu - boundary_sample['U_sig_grid'][:,fixed_time_indx]
        ax.plot(lon_grid, U_mu,
                color='xkcd:cerulean', lw=1.5,
                label='3D GP Prediction')
        ax.fill_between(lon_grid, U_upper, U_lower,
                        color='xkcd:cerulean', alpha=0.33,
                        label='95% CI')
        
    axs[0].legend(bbox_to_anchor=(0., 1.05, 1., .102), loc='lower left',
                  ncols=2, mode="expand", borderaxespad=0.)   
    axs[0].set(xlim=[0,360], ylim=[250, 700])
    
    fig.supxlabel('Heliolongitude [deg.]')
    fig.supylabel('Solar Wind Speed [km/s]')
    plt.show()
    
    # %% =============================================================================
    #     # lon-lat plot
    # =============================================================================
    # transparent-opaque white colorbar
    import matplotlib.pylab as pl
    from matplotlib.colors import ListedColormap
    # Choose colormap
    cmap = pl.cm.binary
    # Get the colormap colors
    my_cmap = cmap(np.zeros(cmap.N)) # np.arange(cmap.N))
    # Set alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    
    
    fig, ax = plt.subplots(figsize=(4,4.5))
    plt.subplots_adjust(bottom=(0.09), left=(0.16), top=(1-0.135), right=(1-0.04),
                        hspace=0)
    
    lons, lats = np.meshgrid(inputs.boundaryDistributions3D['lon_grid'],
                             inputs.boundaryDistributions3D['lat_grid'])
    pcm = ax.pcolormesh(lons, lats, 
                        inputs.boundaryDistributions3D['U_mu_grid'][:,:,fixed_time_indx],
                        vmin=250, vmax=600, cmap='magma')
    ax.pcolormesh(lons, lats, 
                  inputs.boundaryDistributions3D['U_sig_grid'][:,:,fixed_time_indx],
                  vmin=50, vmax=150, cmap=my_cmap)
    
    cax = fig.add_axes((0.16, 1-0.125, 0.80, 0.045))
    fig.colorbar(pcm, cax=cax, pad=0.01, location='top', label='Solar Wind Speed [km/s]')

    for source, lat, ax_key in zip(fixed_time_lats.keys(), fixed_time_lats.values(), ['b', 'c', 'd']):
    # for source, ax_key in zip(inputs.availableSources, ['b', 'c', 'd']):
        
        ax.axhline(lat, label = source, 
                   color='xkcd:cerulean', lw=1.5)
        bbox = dict(boxstyle="round", fc='xkcd:cerulean', ec='xkcd:cerulean',
                    pad=0.2, alpha=0.66)
        ax.annotate(source, (0, lat), (1,0.5), 
                    'data', 'offset fontsize',
                    color='white', ha='left', va='bottom', 
                    bbox=bbox)
    
    ax.set(xlim=[0,360], 
           ylim=[-inputs.latmax.value,inputs.latmax.value])
    fig.supxlabel('Heliolongitude [deg.]')
    fig.supylabel('Heliolatitude [deg.]')

    plt.show()
    
    
    # %% =============================================================================
    #     # lon-lat plot 2
    # =============================================================================
    # What latitude is the source at at this time? Now with Saturn
    fixed_time_lats = {}
    for source in [*inputs.boundarySources, 'saturn']:
        fixed_time_lat = np.interp(fixed_time_mjd,
                                   inputs.availableBackgroundData['mjd'],
                                   inputs.availableBackgroundData[(source, 'lat_HGI')])
        fixed_time_lats[source] = fixed_time_lat
    fixed_time_lats = {k: v for k, v in sorted(fixed_time_lats.items(), key=lambda x: x[1], reverse=True)}
    
    
    fig, ax = plt.subplots(figsize=(4,4.5))
    plt.subplots_adjust(bottom=(0.09), left=(0.16), top=(1-0.135), right=(1-0.04),
                        hspace=0)
    
    lons, lats = np.meshgrid(inputs.boundaryDistributions3D['lon_grid'],
                             inputs.boundaryDistributions3D['lat_grid'])
    pcm = ax.pcolormesh(lons, lats, 
                        inputs.boundaryDistributions3D['U_mu_grid'][:,:,fixed_time_indx],
                        vmin=250, vmax=600, cmap='magma')
    
    ax.pcolormesh(lons, lats, 
                  inputs.boundaryDistributions3D['U_sig_grid'][:,:,fixed_time_indx],
                  vmin=50, vmax=150, cmap=my_cmap)
    
    
    cax = fig.add_axes((0.16, 1-0.125, 0.80, 0.045))
    fig.colorbar(pcm, cax=cax, pad=0.01, location='top', label='Solar Wind Speed [km/s]')

    for source, lat in zip(fixed_time_lats.keys(), fixed_time_lats.values()):
    # for source, ax_key in zip(inputs.availableSources, ['b', 'c', 'd']):
        
        ax.axhline(lat, 
                   color='xkcd:cerulean', lw=1.5)
        bbox = dict(boxstyle="round", fc='xkcd:cerulean', ec='xkcd:cerulean',
                    pad=0.2, alpha=0.66)
        ax.annotate(source, (0, lat), (1,0.5), 
                    'data', 'offset fontsize',
                    color='white', ha='left', va='bottom', 
                    bbox=bbox)
    
    # Add CMEs
    cme_mjds = inputs.cmeDistribution['t_mu']/(24*3600) + inputs.simstarttime.mjd
    cme_delta = 5
    interaction_cme_indx = (cme_mjds > Time(interaction_time).mjd-cme_delta) & (cme_mjds < Time(interaction_time).mjd+cme_delta)
    interaction_cmes = inputs.cmeDistribution.loc[interaction_cme_indx, :]
    
    ax.scatter(interaction_cmes['lon_mu']%360, interaction_cmes['lat_mu'], 
               color = 'xkcd:kelly green', marker='o', fc='None', s=512, lw=2,
               label = 'CME')
        
    ax.legend(loc='upper right', markerscale=1/10)
    
    ax.set(xlim=[0,360], 
           ylim=[-inputs.latmax.value,inputs.latmax.value])
    fig.supxlabel('Heliolongitude [deg.]')
    fig.supylabel('Heliolatitude [deg.]')

    plt.show()
    
    # %% =============================================================================
    # Saturn pre-SIR
    # =============================================================================
    import scipy
    fig, axs = plt.subplots(nrows=2, figsize=[6, 4.5], sharex=True, sharey=True)
    plt.subplots_adjust(bottom=(0.15), left=(0.12), top=(1-0.045), right=(1-0.12),
                        hspace=0.1)
    
    axs[0].set(ylim=[0,8e8])
    for source in ['saturn']:
        axs[0].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData[(source, 'P_RH_core')],
                color='red', lw=0.5, zorder=-10, alpha=0.10,  
                label='1-hour')
        axs[0].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData.rolling('10h', center=True).mean()[(source, 'P_RH_core')],
                color='red', lw=1.0, zorder=-10, alpha=1.0,  
                label='10-hour')
        axs[0].legend(loc='upper right')
        
        axs[1].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData[(source, 'P_LH_core')],
                color='black', lw=0.5, zorder=-10, alpha=0.10, 
                label='1-hour')
        
        axs[1].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData.rolling('10h', center=True).mean()[(source, 'P_LH_core')],
                color='black', lw=1.0, zorder=-10, alpha=1.0, 
                label='10-hour')
        axs[1].legend(loc='upper right')
        
    
    axs[0].set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd],
               ylabel = 'Right-hand Polarized')
    axs[1].set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd],
               ylabel = 'Left-hand Polarized')
    
    axs12 = axs[1].secondary_xaxis(-0.115, 
                functions=(lambda x: x-inputs.starttime.mjd, lambda x: x+inputs.starttime.mjd))
    axs12.set(xlabel = 'MJD; Days from {}'.format(inputs.start.strftime('%Y-%m-%d %H:%M')))
    
    fig.supylabel('SKR Power, Integrated (100-400 kHz) [W/sr]', size='medium')
    
    for ax in axs: 
        ax.axvline(Time(interaction_time).mjd, color='xkcd:fuchsia', ls='--', lw=1.5)
    
    ensemble = inputs.current_ensemble
    source = 'saturn'
    
    weights = [1/len(ensemble)] * len(ensemble)
    weights = np.array(weights)
    metamodel = inputs.estimate(ensemble, weights)
    for ax in axs:
        twin_ax = ax.twinx()
        
        for member in ensemble:
            twin_ax.plot(member['mjd'], member['U'], color='xkcd:cerulean', alpha=0.20, zorder=2, lw=1)
        
        twin_ax.plot(metamodel['mjd'], metamodel['U_median'], color='xkcd:royal blue', alpha=1, lw=1.5, zorder=10,
                     label = 'Ensemble')
        twin_ax.fill_between(metamodel['mjd'], metamodel['U_upper95'], metamodel['U_lower95'],
                             color='xkcd:royal blue', alpha=0.33, lw=1.5, zorder=6,
                             label = '95% CI')
        twin_ax.set(ylim=[250,600], ylabel = 'Solar Wind Speed [km/s]')
        twin_ax.legend(loc='upper left')
    
    ax.set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd])
    
    plt.show()
    
    
    # %% =============================================================================
    # Saturn post-SIR
    # =============================================================================
    import scipy
    r = []
    for member in ensemble:
        ax.plot(member['mjd'], member['U'], color='xkcd:cerulean', alpha=0.10, zorder=4)
        
        r_pershift = []
        for iTime in range(1, 15):
            r_pershift.append(scipy.stats.pearsonr(
                inputs.availableBackgroundData.rolling('10h', center=True).mean()[(source,'P_LH_core')].iloc[:-iTime],
                member['U'].iloc[iTime:])[0])
        
        r.append(np.max(r_pershift))
        
    r = np.array(r)
    new_weights = (r)/(np.sum(r))
    
    
    fig, axs = plt.subplots(nrows=2, figsize=[6, 4.5], sharex=True, sharey=True)
    plt.subplots_adjust(bottom=(0.15), left=(0.12), top=(1-0.045), right=(1-0.12),
                        hspace=0.1)
    
    axs[0].set(ylim=[0,8e8])
    for source in ['saturn']:
        axs[0].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData[(source, 'P_RH_core')],
                color='red', lw=0.5, zorder=-10, alpha=0.10,  
                label='1-hour')
        axs[0].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData.rolling('10h', center=True).mean()[(source, 'P_RH_core')],
                color='red', lw=1.0, zorder=-10, alpha=1.0,  
                label='10-hour')
        axs[0].legend(loc='upper right')
        
        axs[1].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData[(source, 'P_LH_core')],
                color='black', lw=0.5, zorder=-10, alpha=0.10, 
                label='1-hour')
        
        axs[1].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData.rolling('10h', center=True).mean()[(source, 'P_LH_core')],
                color='black', lw=1.0, zorder=-10, alpha=1.0, 
                label='10-hour')
        axs[1].legend(loc='upper right')
        
    
    axs[0].set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd],
               ylabel = 'Right-hand Polarized')
    axs[1].set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd],
               ylabel = 'Left-hand Polarized')
    
    axs12 = axs[1].secondary_xaxis(-0.115, 
                functions=(lambda x: x-inputs.starttime.mjd, lambda x: x+inputs.starttime.mjd))
    axs12.set(xlabel = 'MJD; Days from {}'.format(inputs.start.strftime('%Y-%m-%d %H:%M')))
    
    fig.supylabel('SKR Power, Integrated (100-400 kHz) [W/sr]', size='medium')
    
    for ax in axs: 
        ax.axvline(Time(interaction_time).mjd, color='xkcd:fuchsia', ls='--', lw=1.5)
    
    ensemble = inputs.current_ensemble
    source = 'saturn'
    
    metamodel = inputs.estimate(ensemble, new_weights)
    for ax in axs:
        twin_ax = ax.twinx()
        
        for member in ensemble:
            twin_ax.plot(member['mjd'], member['U'], color='xkcd:cerulean', alpha=0.20, zorder=2, lw=1)
        
        twin_ax.plot(metamodel['mjd'], metamodel['U_median'], color='xkcd:royal blue', alpha=1, lw=1.5, zorder=10,
                     label = 'Ensemble')
        twin_ax.fill_between(metamodel['mjd'], metamodel['U_upper95'], metamodel['U_lower95'],
                             color='xkcd:royal blue', alpha=0.33, lw=1.5, zorder=6,
                             label = '95% CI')
        twin_ax.set(ylim=[250,600], ylabel = 'Solar Wind Speed [km/s]')
        twin_ax.legend(loc='upper left')
    
    ax.set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd])
    
    plt.show()

    
    # %% =============================================================================
    # Saturn post-SIR Zoom out
    # =============================================================================
    import scipy
    r = []
    for member in ensemble:
        ax.plot(member['mjd'], member['U'], color='xkcd:cerulean', alpha=0.10, zorder=4)
        
        r_pershift = []
        for iTime in range(1, 15):
            r_pershift.append(scipy.stats.pearsonr(
                inputs.availableBackgroundData.rolling('10h', center=True).mean()[(source,'P_LH_core')].iloc[:-iTime],
                member['U'].iloc[iTime:])[0])
        
        r.append(np.max(r_pershift))
        
    r = np.array(r)
    new_weights = (r)/(np.sum(r))
    
    
    fig, axs = plt.subplots(nrows=2, figsize=[6, 4.5], sharex=True, sharey=True)
    plt.subplots_adjust(bottom=(0.15), left=(0.12), top=(1-0.045), right=(1-0.12),
                        hspace=0.1)
    
    axs[0].set(ylim=[0,8e8])
    for source in ['saturn']:
        axs[0].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData[(source, 'P_RH_core')],
                color='red', lw=0.5, zorder=-10, alpha=0.10,  
                label='1-hour')
        axs[0].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData.rolling('10h', center=True).mean()[(source, 'P_RH_core')],
                color='red', lw=1.0, zorder=-10, alpha=1.0,  
                label='10-hour')
        axs[0].legend(loc='upper right')
        
        axs[1].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData[(source, 'P_LH_core')],
                color='black', lw=0.5, zorder=-10, alpha=0.10, 
                label='1-hour')
        
        axs[1].plot(inputs.availableBackgroundData['mjd'], 
                inputs.availableBackgroundData.rolling('10h', center=True).mean()[(source, 'P_LH_core')],
                color='black', lw=1.0, zorder=-10, alpha=1.0, 
                label='10-hour')
        axs[1].legend(loc='upper right')
        
    
    axs[0].set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd],
               ylabel = 'Right-hand Polarized')
    axs[1].set(xlim = [inputs.starttime.mjd, inputs.stoptime.mjd],
               ylabel = 'Left-hand Polarized')
    
    axs12 = axs[1].secondary_xaxis(-0.115, 
                functions=(lambda x: x-inputs.starttime.mjd, lambda x: x+inputs.starttime.mjd))
    axs12.set(xlabel = 'MJD; Days from {}'.format(inputs.start.strftime('%Y-%m-%d %H:%M')))
    
    fig.supylabel('SKR Power, Integrated (100-400 kHz) [W/sr]', size='medium')
    
    for ax in axs: 
        ax.axvline(Time(interaction_time).mjd, color='xkcd:fuchsia', ls='--', lw=1.5)
    
    ensemble = inputs.current_ensemble
    source = 'saturn'
    
    metamodel = inputs.estimate(ensemble, new_weights)
    for ax in axs:
        twin_ax = ax.twinx()
        
        for member in ensemble:
            twin_ax.plot(member['mjd'], member['U'], color='xkcd:cerulean', alpha=0.20, zorder=2, lw=1)
        
        twin_ax.plot(metamodel['mjd'], metamodel['U_median'], color='xkcd:royal blue', alpha=1, lw=1.5, zorder=10,
                     label = 'Ensemble')
        twin_ax.fill_between(metamodel['mjd'], metamodel['U_upper95'], metamodel['U_lower95'],
                             color='xkcd:royal blue', alpha=0.33, lw=1.5, zorder=6,
                             label = '95% CI')
        twin_ax.set(ylim=[250,600], ylabel = 'Solar Wind Speed [km/s]')
        twin_ax.legend(loc='upper left')
    
    ax.set(xlim = [inputs.simstarttime.mjd, inputs.stoptime.mjd])
    
    plt.show()

    

    # %%
    breakpoint()

    
    # Read all the available assimilation data
    # test.get_availableData()
    # test.filter_availableData()
    # test.sort_availableSources('rad_HGI')
    
    # Save copies of the original distributions, for comparisons
    original_boundaryDistribution = copy.deepcopy(test.boundaryDistribution)
    original_cmeDistribution = copy.deepcopy(test.cmeDistribution)
    
    # Initial weights
    
    
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
        # !!!! Update the individual 2D distributions,
        # !!!! then update the 3D distribution
        test.update_distributions(new_weights, boundarySamples, cmeSamples)
        
        weights = new_weights

    
        