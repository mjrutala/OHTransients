#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:09:53 2025

@author: mrutala
"""
import astropy.units as u
from astropy.time import Time
import datetime
# import datetime
# import os
# import astropy.units as u
# import glob
# import re
import numpy as np
import time
from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries
# import requests
import matplotlib.pyplot as plt
import pandas as pd
from astroquery.jplhorizons import Horizons
# import dask
import pickle
import tqdm
import copy
import tensorflow as tf

import sys
sys.path.append('/Users/mrutala/projects/HUXt/code/')
sys.path.append('/Users/mrutala/projects/OHTransients/code/')
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
import huxt_atObserver as hao
# from scipy import ndimage
# from scipy import stats
# from sklearn.metrics import root_mean_squared_error as rmse
# from astroquery.jplhorizons import Horizons

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

class multihuxt_inputs:
    def __init__(self, start, stop, rmax=1, latmax=10):
        self.start = start
        self.stop = stop
        self.radmax = rmax * u.AU
        self.latmax = latmax * u.deg
        self.innerbound= 21.5 * u.solRad
        
        self.usw_minimum = 200 * u.km/u.s
        self.SiderealCarringtonRotation = 27.28 * u.day
        self.SynodicCarringtonRotation = 25.38 * u.day
        
        # Required initializations
        # Other methods check that these are None (or have value) before 
        # continuing, so they must be intialized here
        self._boundarySources = None
        self._ephemeris = {}
        
        # Input data initialization
        cols = ['t_mu', 't_sig', 'lon_mu', 'lon_sig', 'lat_mu', 'lat_sig',
                'width_mu', 'width_sig', 'speed_mu', 'speed_sig', 
                'thickness_mu', 'thickness_sig', 'innerbound']
        self.cmeDistribution = pd.DataFrame(columns = cols)
        
        
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
        n = np.ceil((self.radmax / self.usw_minimum).to(u.day) / (27*u.day))
        return (n * 27 * u.day, 27 * u.day)
    
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
        
    # @property
    # def boundarySources(self):
    #     # # If the time span is short enough that no ICMEs are measured
    #     # boundarySources = set(self.availableTransientData['affiliated_source'])
    #     # Temporary fix: hardcoding
    #     boundarySources = {'omni', 'stereo a', 'stereo b'}
    #     return sorted(boundarySources)
    @property
    def boundarySources(self):
        if self._boundarySources is None:
            self._boundarySources = ['omni', 'stereo a', 'stereo b']
        return self._boundarySources
    
    @boundarySources.setter
    def boundarySources(self, boundarySources):
        self._boundarySources = boundarySources
    
    def copy(self):
        return copy.deepcopy(self)
    
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
            print(source)
            print('----------------------------')
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
        try:
            dataset = attrs.cdaweb.Dataset(fetch_datasetID)
            result = Fido.search(timerange, dataset)
            downloaded_files = Fido.fetch(result, path = self.path_cohodata + source.upper() + '/{file}')
        except:
            breakpoint()
        
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
    
    def get_availableTransientData(self, sources=None, duration=2):
        
        duration = duration * u.day
        
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
                                    ensureCME = True)
            icmes['affiliated_source'] = source
            
            availableTransientData_list.append(icmes)
        
        availableTransientData_df = pd.concat(availableTransientData_list, axis='rows')
        availableTransientData_df.reset_index(inplace=True, drop=True)
        if len(availableTransientData_df) > 0:
            availableTransientData_df['mjd'] = Time(availableTransientData_df['eventTime'])

        self.availableTransientData = availableTransientData_df 
        
        return
    
    def get_indexICME(self, source, 
                      icme_buffer=0.5, 
                      interp_buffer=1.0):
        
        icme_buffer *= u.day
        interp_buffer *= u.day
        
        # Get the insitu data + mjd at this source
        insitu = self.availableBackgroundData[source]
        insitu['mjd'] = self.availableBackgroundData['mjd']
        
        # Interpolate over existing data gaps (NaNs), so they aren't caught as ICMEs
        insitu.interpolate(method='linear', axis='columns', limit_direction='both', inplace=True)
        
        # Get ICMEs
        icmes = self.availableTransientData.query('affiliated_source == @source')
        icmes.reset_index(inplace=True, drop=True)
        
        # Remove ICMEs from OMNI data, leaving NaNs behind
        if 'eventTime' in icmes.columns: 
            icmes = icmes.rename(columns = {'eventTime': 'Shock_time'})
            icmes['ICME_end'] = [row['Shock_time'] + datetime.timedelta(days=(row['duration'])) for _, row in icmes.iterrows()]
        
        if len(icmes) > 0:
            insitu_noicme = Hin.remove_ICMEs(insitu, icmes, 
                                             params=['U'], 
                                             interpolate = False, 
                                             icme_buffer = icme_buffer, 
                                             interp_buffer = interp_buffer, 
                                             fill_vals = np.nan)
        else:
            insitu_noicme = insitu
            
        return insitu_noicme['U'].isna()
    
    @property
    def ephemeris(self):
        from astropy.time import Time
        # If this hasn't been run before, run for all 
        if len(self._ephemeris) == 0:
            print("No ephemeris loaded. Now generating...")
            for source in self.availableSources:
                eph = H.Observer(source, Time(self.availableBackgroundData.index))
                self._ephemeris[source] = eph
                    
        return self._ephemeris
    
    
    def get_carringtonPeriod(self, distance):
                   
        # source speed, approximated as circular
        kepler_const = ((1 * u.year).to(u.day))/((1 * u.au)**(3/2))
        source_period = kepler_const * distance**(3/2)
        source_speed = (2 * np.pi * u.rad) / (source_period.to(u.day))
        
        # sun speed
        sun_speed = (2 * np.pi * u.rad)/(25.38 * u.day)
        
        synodic_period = 1/(sun_speed - source_speed) * (2 * np.pi * u.rad)
        
        return synodic_period

    def generate_backgroundDistributions(self, insitu=None, ICME_df=None, 
                                         chunking=(60,30), average_cluster_span=6,
                                         inducing_variable=True,
                                         simple=False):
        
        # Calculate the span from stop - start
        span = (self.stop - self.start).total_seconds() * u.s
        simspan = (self.simstop - self.simstart).total_seconds() * u.s
        
        # Parse optional keywords
        chunk_starts, chunk_stops = [], []
        if chunking is None:
            chunk_starts.append(self.simstart)
            chunk_stops.append(self.simstop)
        else:
            # Slap some correct units on
            chunking *= u.day
            
            n_chunks_approx = np.ceil((simspan/chunking[1]).decompose()).value - 1
            
            for i in range(int(n_chunks_approx)):
                chunk_starts.append(self.simstart + datetime.timedelta(days=chunking[1].value*i))
                chunk_stops.append(chunk_starts[-1] + datetime.timedelta(days=chunking[0].value))
            chunk_stops[-1] = self.simstop
            # Merge the last two chunks if the last chunk is too small
            if (chunk_stops[-1] - chunk_starts[-1]).total_seconds() * u.s < 0.5 * chunking[1]:
                chunk_starts.pop()
                chunk_stops.pop()
                chunk_stops[-1] = self.simstop
        n_chunks = len(chunk_starts)
        
        # More correct unit-slapping
        average_cluster_span *= u.hour
        
        target_variables = ['U'] # , 'BR']
        
        n_data = len(self.availableBackgroundData)
        
        ambient_solar_wind_d = {k: {v: {'mean': np.full([n_data, n_chunks], np.nan), 
                                        'std': np.full([n_data, n_chunks], np.nan), 
                                        'cov': np.full([n_data, n_data, n_chunks], np.nan)} 
                                    for v in target_variables} 
                                for k in self.boundarySources}
        
        for source in self.boundarySources:
            
            # If indexICME is not supplied, look it up
            if ICME_df is None:
                indexICME = self.get_indexICME(source)
            else:
                indexICME = ICME_df[source]
            
            # Where an ICME is present, set U, BR to NaN
            insitu_noICME = self.availableBackgroundData[source]
            insitu_noICME['mjd'] = self.availableBackgroundData['mjd']
            insitu_noICME.loc[indexICME, ['U', 'BR']] = np.nan
            
            new_insitu_list = []
            covariance_list = []
            # sample_func_df = pd.DataFrame(columns=['start_mjd', 'stop_mjd', 'func'])
            
            # Break the resulting df into chunks of size break_at, then loop
            insitu_noICME_chunks = []
            for chunk_start, chunk_stop in zip(chunk_starts, chunk_stops):
                
                chunk = insitu_noICME.query("@chunk_start <= index < @chunk_stop")
                
                insitu_noICME_chunks.append(chunk)
            
            for i, chunk in enumerate(insitu_noICME_chunks):
                    
                # Option #1: Simple linear interpolation
                if simple:
                    print("Skipping GPR!")
                    new_insitu, sample_func = self.ambient_solar_wind_LI(chunk)
                
                # Options #2: Gaussian Process Regression
                else:
                    # Perform the GPR
                    if chunk['U'].isna().all():
                        print("ERROR: No valid variable for GPR (all NaN!)")
                        breakpoint()
                        
                    # Calculate Carrington Period for this source
                    chunk_eph_indx = (self.ephemeris[source].time.mjd >= chunk['mjd'].iloc[0]) & \
                                     (self.ephemeris[source].time.mjd < chunk['mjd'].iloc[-1])
                    chunk_mean_dist = self.ephemeris[source].r[chunk_eph_indx].mean()
                    
                    carrington_period = self.get_carringtonPeriod(chunk_mean_dist)
                    
                    ambient_solar_wind = self.ambient_solar_wind_GP(chunk, 
                                                                    average_cluster_span, 
                                                                    carrington_period,
                                                                    inducing_variable=inducing_variable,
                                                                    target_variables=target_variables)
                    
                c0indx = np.where(insitu_noICME['mjd'].to_numpy() == chunk['mjd'].to_numpy()[0])[0][0]
                c1indx = c0indx + len(chunk)
                for var in target_variables:
                    ambient_solar_wind_d[source][var]['mean'][c0indx:c1indx, i] = ambient_solar_wind[var]['mean']
                    ambient_solar_wind_d[source][var]['std'][c0indx:c1indx, i] = ambient_solar_wind[var]['std']
                    ambient_solar_wind_d[source][var]['cov'][c0indx:c1indx, c0indx:c1indx, i] = ambient_solar_wind[var]['cov']
                
        # Recombine the chunked mean, standard deviation and add them to a copy of the original data
        backgroundDistributions = self.availableBackgroundData.copy(deep=True)
        for source in set(self.availableSources) - set(self.boundarySources):
            backgroundDistributions.drop(columns=[source], inplace=True)
            
        for source in self.boundarySources:
            for var in target_variables:
                backgroundDistributions.drop(columns=[(source, var)], inplace=True)
                
                # This handles the overlapping portions of each chunk
                mean_mean = np.nanmean(ambient_solar_wind_d[source][var]['mean'], axis=1)
                mean_std = np.nanmean(ambient_solar_wind_d[source][var]['std'], axis=1)
                
                backgroundDistributions[(source, var+'_mu')] = mean_mean
                backgroundDistributions[(source, var+'_sig')] = mean_std
        
        # Recombine the chunked covariance matrix, then store it with means as lists
        # This is for easier sampling later
        backgroundCovariances = {s: {v: {'time': [], 'mean': [], 'std': [], 'cov': []} 
                                     for v in target_variables} 
                                 for s in self.boundarySources}
        
        for source in self.boundarySources:
            for var in target_variables:
                
                # This handles the overlapping portions of each chunk
                mean_mean = np.nanmean(ambient_solar_wind_d[source][var]['mean'], axis=1)
                mean_std = np.nanmean(ambient_solar_wind_d[source][var]['std'], axis=1)
                mean_cov = np.nanmean(ambient_solar_wind_d[source][var]['cov'], axis=2)
                
                # Now that there aren't edge effects, split it back up
                for chunk_start, chunk_stop in zip(chunk_starts, chunk_stops):
                    c0indx = np.where(backgroundDistributions.index == chunk_start)[0][0]
                    if chunk_stop != chunk_stops[-1]:
                        c1indx = np.where(backgroundDistributions.index == chunk_stop)[0][0]
                    else:
                        c1indx = len(backgroundDistributions)
                        
                    backgroundCovariances[source][var]['time'].append(backgroundDistributions.index[c0indx:c1indx])
                    backgroundCovariances[source][var]['mean'].append(mean_mean[c0indx:c1indx])
                    backgroundCovariances[source][var]['std'].append(mean_std[c0indx:c1indx])
                    backgroundCovariances[source][var]['cov'].append(mean_cov[c0indx:c1indx, c0indx:c1indx])
        
        self.backgroundDistributions = backgroundDistributions
        self.backgroundCovariances = backgroundCovariances
        
        # =============================================================================
        # Visualization
        # =============================================================================
        fig, axs = plt.subplots(nrows=len(self.boundarySources), sharex=True, sharey=True,
                                figsize=(6, 4.5))
        plt.subplots_adjust(bottom=(0.16), left=(0.12), top=(1-0.08), right=(1-0.06),
                            hspace=0)
        if len(self.boundarySources) == 1:
            axs = [axs]
        for ax, source in zip(axs, self.boundarySources):
            
            ax.scatter(self.availableBackgroundData['mjd'], self.availableBackgroundData[(source, 'U')],
                       color='black', marker='.', s=2, zorder=3,
                       label = 'Raw Data')
            # ax.scatter(self.availableBackgroundData.loc[indexICME, 'mjd'], 
            #            self.availableBackgroundData.loc[indexICME, (source, 'U')],
            #            edgecolor='xkcd:scarlet', marker='o', s=6, zorder=2, facecolor='None', lw=0.5,
            #            label = 'ICMEs from DONKI')
            
            # If indexICME is not supplied, look it up
            if ICME_df is None:
                indexICME = self.get_indexICME(source)
            else:
                indexICME = ICME_df[source]
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
    
    def ambient_solar_wind_GP(self, df, average_cluster_span, carrington_period,
                              inducing_variable=False, target_variables=['U']):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        import gpflow
        from sklearn.cluster import KMeans
        
        # Check that all target_variables are present
        for target_variable in target_variables:
            if df[target_variable].isna().all() == True:
                print("All variables are NaNs; cannot proceed with GPR")
                return None, None
        
        # Set up dictionaries 
        gp_variables = {k: {} for k in target_variables}
        
        for target_variable in target_variables:
                
            # Get the mjd and U as column vectors for GPR
            df_nonan = df.dropna(axis='index', how='any', subset=['U'])
            mjd = df_nonan['mjd'].to_numpy(float)[:, None]
            var = df_nonan[target_variable].to_numpy(float)[:, None]
    
            # MJD scaled to 1-day increments
            mjd_rescaler = MinMaxScaler((0, mjd[-1]-mjd[0]))
            mjd_rescaler.fit(mjd)
            X = mjd_rescaler.transform(mjd)

            # Variable rescaled to z-score
            var_rescaler = StandardScaler()
            var_rescaler.fit(var)
            Y = var_rescaler.transform(var)
    
            # K-means cluster the data (fewer data = faster processing)
            # And calculate the variance within each cluster
            XY = np.array(list(zip(X.flatten(), Y.flatten())))
            chunk_span = (df_nonan.index[-1] - df_nonan.index[0]).total_seconds() * u.s
            n_clusters = int((chunk_span / average_cluster_span).decompose())
            
            if n_clusters < len(XY):
                kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(XY)
                XYc = kmeans.cluster_centers_
                # # This requires each cluster to have multiple points, which is not guaranteed
                # # Instead, use the length scale as a guess at uncertaintys
                Yc_var = np.array([np.var(XY[kmeans.labels_ == i, 1]) 
                                   for i in range(kmeans.n_clusters)])
                approx_noise = np.percentile(Yc_var, 90)
            else:
                n_clusters = len(XY)
                XYc = XY
                # 1/10 the overall standard deviation of Y
                # Yc_var = np.array([0.1]*len(XY))
                approx_noise = 0.1
            print("{} clusters to be fit".format(len(XYc)))

            
            # Arrange clusters to be strictly increasing in time (X)
            Xc, Yc = XYc.T[0], XYc.T[1]
            cluster_sort = np.argsort(Xc)
            Xc = Xc[cluster_sort][:, None]
            Yc = Yc[cluster_sort][:, None]
            Yc_var = Yc_var[cluster_sort][:, None]
    
            # Construct the signal kernel for GP
            # Again, these lengthscales could probable be calculated?
            # small_scale_kernel = gpflow.kernels.SquaredExponential(variance=2**2, lengthscales=1.0)
            # large_scale_kernel = gpflow.kernels.SquaredExponential(variance=2**2, lengthscales=10.0)
            # irregularities_kernel = gpflow.kernels.SquaredExponential(variance=1**2, lengthscales=1.0)
            # # Fixed period Carrington rotation kernel
            # period_rescaled = 100 / (simspan / carrington_period).decompose().value
            # carrington_kernel = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), 
            #                                             period=gpflow.Parameter(period_rescaled, trainable=False))
    
            # signal_kernel = small_scale_kernel + large_scale_kernel + irregularities_kernel + carrington_kernel
            
            # New kernel
            period_rescaled = carrington_period.to(u.day).value
            signal_kernel = gpflow.kernels.RationalQuadratic() + \
                            gpflow.kernels.RationalQuadratic() * \
                            gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(),
                                                    period=gpflow.Parameter(period_rescaled, trainable=False)) 
            
            # This model *could* be solved with the exact noise for each point...
            # But this would require defining a custom likelihood class...
            if inducing_variable == False:
                model = gpflow.models.GPR((Xc, Yc), 
                                          kernel=copy.deepcopy(signal_kernel), 
                                          noise_variance=approx_noise
                                          )
            else:
                model = gpflow.models.SGPR((X, Y), 
                                           kernel=copy.deepcopy(signal_kernel), 
                                           noise_variance=approx_noise,
                                           inducing_variable=Xc)
                
            opt = gpflow.optimizers.Scipy()
            opt.minimize(model.training_loss, model.trainable_variables)
            
            # gpflow.utilities.print_summary(model)
            
            Xo = mjd_rescaler.transform(df['mjd'].to_numpy()[:, None])
            
            Yo_mu, Yo_var = model.predict_y(Xo)
            Yo_mu, Yoc_var = np.array(Yo_mu), np.array(Yo_var)
            Yo_sig = np.sqrt(Yo_var)
            _, cov = model.predict_f(Xo, full_cov=True)
            
            new_mjd = mjd_rescaler.inverse_transform(Xo)
            new_var_mu = var_rescaler.inverse_transform(Yo_mu)
            new_var_sig= Yo_sig * var_rescaler.scale_
            new_var_cov = cov * var_rescaler.scale_
            
            # Assign to dict
            gp_variables[target_variable]['mean'] = new_var_mu.ravel()
            gp_variables[target_variable]['std'] = new_var_sig.ravel()
            gp_variables[target_variable]['cov'] = new_var_cov.numpy().squeeze()
            
        return gp_variables
    
    def ambient_solar_wind_LI(self, df):
        
        
        new_insitu = df.copy(deep=True)
        new_insitu.drop(columns='U', inplace=True)
        
        for var_str in ['U']: # ['U', 'BR']:
            
            new_insitu['U_mu'] = df[var_str].interpolate(limit_direction=None)
            new_insitu['U_sig'] = new_insitu['U_mu'].rolling('1d').std()

            # # Replace non-ICME regions with real data
            # noNaN_bool = ~df[var_str].isna()
            # new_insitu.loc[noNaN_bool, 'U_mu'] = df.loc[noNaN_bool, 'U']
            # new_insitu.loc[noNaN_bool, 'U_sig'] *= 1/10.
            
            # # Save a function to generate samples of f with full covariance
            def func(mjd, num_samples):
                fo_samples = []
                for _ in range(num_samples):
                    sample = new_insitu.query("@mjd[0] <= mjd <= @mjd[-1]")[var_str+'_mu']
                    fo_samples.append(sample)
                fo_samples = np.array(fo_samples)
                return fo_samples
            
            breakpoint()
            
        return new_insitu, func
                 
    def generate_backgroundSamples(self, num_samples):
        import gpflow
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        # Check that the sampler functions have been defined
        try: 
            keys = self.backgroundCovariances.keys()
        except:
            print("generate_backgroundDistributions() must be run before background samples can be generated")
        
        # Check if we've already generated some samples?
        # breakpoint()
        
        # Set up a list of sample dataframes ahead of time
        backgroundSamples = [self.backgroundDistributions.copy(deep=True) for _ in range(num_samples)]            
        
        # Try my hand at sampling the covariance matrix directly
        for source in self.boundarySources:
            for var, vals in self.backgroundCovariances[source].items():
                
                time_list = vals['time']
                mu_list = vals['mean']
                std_list = vals['std']
                cov_list = vals['cov']
                
                rng = np.random.default_rng()
                
                for time, mu, std, cov in zip(time_list, mu_list, std_list, cov_list):
                    
                    # Get a rescaler to satisfy the matrix sampler
                    var_rescaler = StandardScaler()
                    var_rescaler.fit(mu[:, None])
                    
                    # Columnize the mean
                    mu_scaled = var_rescaler.transform(mu[:, None])
                    mu_for_sample = tf.linalg.adjoint(mu_scaled)
                    
                    # Columnize the standard deviation
                    sigma_scaled = std[:, None]/var_rescaler.scale_
                    
                    # Get randomly sampled mean + standard deviation (as backup)
                    samples_rng = [rng.normal(loc = mu_scaled, scale = sigma_scaled) 
                                   for _ in range(num_samples)]
                    samples_rng = np.array(samples_rng).squeeze()
                    
                    # "Columnize" the covariance matrix
                    cov_scaled = cov / var_rescaler.scale_
                    cov_for_sample = cov_scaled[None, :, :]
                    
                    # Get samples
                    samples_cov = gpflow.conditionals.util.sample_mvn(
                        mu_for_sample, cov_for_sample, True, num_samples=num_samples)
                    samples_cov = samples_cov.numpy().squeeze()
                    
                    # Each successive chunk will overwrite the previous overlapping bit
                    # Since we've ensured continuity, this *seems* to be okay
                    for i, (sample_cov, sample_rng) in enumerate(zip(samples_cov, samples_rng)):
                        
                        sample_cov_unscaled = var_rescaler.inverse_transform(sample_cov[:, None])
                        sample_rng_unscaled = var_rescaler.inverse_transform(sample_rng[:, None])
                        
                        # Check if the Cholesky decomposition failed (all NaN?)
                        if np.isnan(sample_cov).any() == False:
                            
                            # If not, add the sample
                            backgroundSamples[i].loc[time, (source, var+'_mu')] = sample_cov_unscaled.flatten()
                            backgroundSamples[i].loc[time, (source, var+'_sig')] = 0
                            
                        else: 
                            
                            backgroundSamples[i].loc[time, (source, var+'_mu')] = sample_rng_unscaled.flatten()
                            # Sigma remains unchanged for this case
                    
        self.backgroundSamples = backgroundSamples
  
        return
        
    def generate_boundaryDistributions(self, constant_sig=0):
        from tqdm import tqdm
        # from dask.distributed import Client, as_completed, LocalCluster
        import multiprocessing as mp
        import logging
        from joblib import Parallel, delayed
        from tqdm import tqdm
        
        nCores = int(0.75 * mp.cpu_count()) 

        rng = np.random.default_rng()
        
        # methodOptions = ['forward', 'back', 'both']
        methodOptions = ['both']
        
        boundaryDistributions_dict = {}
        for source in self.boundarySources:
        
            # Format the insitu df (backgroundDistribution) as HUXt expects it
            insitu_df = self.backgroundDistributions[source].copy(deep=True)
            insitu_df['BX_GSE'] =  -insitu_df['BR']
            insitu_df['V'] = insitu_df['U_mu']
            insitu_df['datetime'] = insitu_df.index
            insitu_df = insitu_df.reset_index()
        
            # Map inwards once to get the appropriate dimensions, etc.
            # t, vcarr, bcarr = Hin.generate_vCarr_from_OMNI(self.simstart, self.simstop, omni_input=insitu_df)
            t, vcarr, bcarr = Hin.generate_vCarr_from_insitu(self.simstart, self.simstop, 
                                                             insitu_source=source, insitu_input=insitu_df)
            
            # Sample the velocity distribution and assign random mapping directions (method)
            # Randomly assigning these is equivalent to performing each mapping for each sample (for large numbers of samples)
            # Having a single random population should be better mathematically
            
            dfSamples = [df[source] for df in self.backgroundSamples]
            methodSamples = rng.choice(methodOptions, len(dfSamples))
            
            func = _map_vBoundaryInwards
            funcGenerator = Parallel(return_as='generator', n_jobs=nCores)(
                delayed(func)(self.simstart, self.simstop, source, df_sample, method_sample, self.innerbound)
                for df_sample, method_sample in zip(dfSamples, methodSamples))
            
            results = list(tqdm(funcGenerator, total=len(dfSamples)))
            
            # uSamples = [rng.normal(loc=insitu_df['U_mu'], 
            #                        scale=insitu_df['U_sig']
            #                        ) for _ in range(nSamples)]
            
            # methodSamples = rng.choice(methodOptions, nSamples)
            
            # vcarrGenerator = Parallel(return_as='generator', n_jobs=nCores, background='threading')(
            #     delayed(process_sample)(u, method) 
            #     for u, method in zip(uSamples, methodSamples)
            #     )
            # vcarrSamples = list(tqdm(vcarrGenerator, total=nSamples))
    
            # Characterize the resulting samples as one distribution
            vcarr_mu = np.nanmean(results, axis=0)
            vcarr_sig = np.sqrt(np.nanstd(results, axis=0)**2 + constant_sig**2)
            
            
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
    
    def generate_boundaryDistribution3D(self, nLat=32, extend=None, GP=True):
        
        # Get dimensions from OMNI boundary distribution, which *must* exist
        nLon, nTime = self.boundaryDistributions['omni']['U_mu_grid'].shape
        
        # Coordinates = (lat, lon, time)
        # Values = boundary speed, magnetic field* (*not implemented fully)
        lat_for3d = np.linspace(-self.latmax.value, self.latmax.value, nLat)
        lon_for3d = np.linspace(0, 360, nLon)
        mjd_for3d = self.boundaryDistributions['omni']['t_grid']
        
        if (type(extend) == str) & (GP == True):
            print("Cannot have extend=str and GP=True!")
            return
        if type(extend) == str:
            U_mu_3d, U_sigma_3d, B_3d = self._extend_boundaryDistributions(nLat, extend)
        elif GP is True:
            U_mu_3d, U_sigma_3d, B_3d = self._impute_boundaryDistributions(lat_for3d, lon_for3d, mjd_for3d)
            
        
        self.boundaryDistributions3D = {'t_grid': mjd_for3d,
                                        'lon_grid': lon_for3d,
                                        'lat_grid': lat_for3d,
                                        'U_mu_grid': U_mu_3d,
                                        'U_sig_grid': U_sigma_3d,
                                        'B_grid': B_3d,
                                        }
        
        return
        
    def _extend_boundaryDistributions(self, nLat, name):
        
        U_mu_3d = np.tile(self.boundaryDistributions[name]['U_mu_grid'], 
                          (nLat, 1, 1))
        U_sigma_3d = np.tile(self.boundaryDistributions[name]['U_sig_grid'], 
                          (nLat, 1, 1))
        B_3d = np.tile(self.boundaryDistributions[name]['B_grid'], 
                          (nLat, 1, 1))
        
        return U_mu_3d, U_sigma_3d, B_3d
    
    def _interpolate_boundaryDistributions(self, lat_for3d, lon_for3d, mjd_for3d):
        
        breakpoint()
        
        return
        
    def _impute_boundaryDistributions(self, lat_for3d, lon_for3d, mjd_for3d):
        import gpflow
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        # from scipy.cluster.vq import kmeans
        from sklearn.cluster import KMeans
        import multiprocessing as mp
        from joblib import Parallel, delayed
        
        # Get dimensions from OMNI boundary distribution, which *must* exist
        nLat = len(lat_for3d)
        nLon = len(lon_for3d)
        nMjd = len(mjd_for3d)
        
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
        
        # rng = np.random.default_rng()
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
                               self.backgroundDistributions['mjd'],
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
        
        # =============================================================================
        #         # Find XY clusters to reduce number of points in GP
        # =============================================================================
        # print("OPTIMIZE DOWNSAMPLING BY Z-SCORE RMSE")
        # breakpoint()
        
        # n_clusters = int(0.01 * len(yval_mu))
        n_clusters = 2000
        
        X = np.column_stack([xlat, xlon, xmjd])
        Y_mu = yval_mu
        Y_sigma = yval_sigma
        
        # Cluster for means
        XY_mu = np.column_stack([X, Y_mu])
        kmeans_mu = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(XY_mu)
        XYc_mu = kmeans_mu.cluster_centers_
        Xc_mu, Yc_mu = XYc_mu[:,:3], XYc_mu[:,3][:,None]
        Yc_mu_sigma = np.array([np.std(XY_mu[kmeans_mu.labels_ == i, 1]) 
                                for i in range(kmeans_mu.n_clusters)])
        Yc_mu_noise = np.percentile(Yc_mu_sigma, 50)**2
        
        # Cluster for standard deviations
        XY_sigma = np.column_stack([X, Y_sigma])
        kmeans_sigma = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(XY_sigma)
        XYc_sigma = kmeans_sigma.cluster_centers_
        Xc_sigma, Yc_sigma = XYc_sigma[:,:3], XYc_sigma[:,3][:,None]
        Yc_sigma_sigma = np.array([np.std(XY_sigma[kmeans_sigma.labels_ == i, 1]) 
                                for i in range(kmeans_sigma.n_clusters)])
        Yc_sigma_noise = np.percentile(Yc_mu_sigma, 50)**2

        # =============================================================================
        #         # Define kernel for each dimension separately, then altogether
        # =============================================================================
        lat_kernel = gpflow.kernels.RationalQuadratic(active_dims=[0])
        
        lon_kernel = gpflow.kernels.SquaredExponential(active_dims=[1]) + \
                     gpflow.kernels.SquaredExponential(active_dims=[1]) * \
                     gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims=[1]), 
                                             period=gpflow.Parameter(1, trainable=False))
                     
        mjd_kernel = gpflow.kernels.RationalQuadratic(active_dims=[2])
        
        all_kernel = gpflow.kernels.RationalQuadratic()
        kernel_mu = (lat_kernel + lon_kernel + mjd_kernel + 
                     lat_kernel*lon_kernel + lat_kernel*mjd_kernel + lon_kernel*mjd_kernel +
                     all_kernel)
        
        kernel_sigma = copy.deepcopy(kernel_mu)
        
        # =============================================================================
        #         Optimize
        # =============================================================================
        print("Optimizing GP Model... This may take some time.")
        print("Trying to optimize for {} point".format(len(Yc_mu)))
        
        model_mu = gpflow.models.GPR((Xc_mu, Yc_mu),
                                     kernel=kernel_mu,
                                     noise_variance=Yc_mu_noise)
        opt_mu = gpflow.optimizers.Scipy()
        opt_mu.minimize(model_mu.training_loss, model_mu.trainable_variables)
        
        model_sigma = gpflow.models.GPR((Xc_sigma, Yc_sigma),
                                        kernel=kernel_sigma,
                                        noise_variance=Yc_sigma_noise)
        opt_sigma = gpflow.optimizers.Scipy()
        opt_sigma.minimize(model_sigma.training_loss, model_sigma.trainable_variables)
        
        # =============================================================================
        # Predict values for the full grid...     
        # =============================================================================
        Xlat, Xlon, Xmjd = np.meshgrid(lat_scaler.transform(lat_for3d[:,None]),
                                       lon_scaler.transform(lon_for3d[:,None]), 
                                       mjd_scaler.transform(mjd_for3d[:,None]),
                                       indexing='ij')
        X3d = np.column_stack([Xlat.flatten()[:,None],
                               Xlon.flatten()[:,None],
                               Xmjd.flatten()[:,None]])
        
        breakpoint()
        
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
        
        U_mu_3d = U_mu.reshape(nLat, nLon, nMjd)
        U_sigma_3d = U_sigma.reshape(nLat, nLon, nMjd)
        
        # Generate an OBVIOUSLY WRONG B
        B_3d = np.tile(self.boundaryDistributions['omni']['B_grid'], (64, 1, 1))
        
        return U_mu_3d, U_sigma_3d, B_3d
        
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
    
    def generate_cmeDistribution(self, search=True):
        
        # 
        t_sig_init = 36000 # seconds
        lon_sig_init = 15 # degrees
        lat_sig_init = 15 # degrees
        width_sig_init = 30 # degrees
        thick_mu_init = 4 # solar radii
        thick_sig_init = 1 # solar radii
        speed_sig_init = 400 # km/s
        
        # Get the CMEs
        if search == True:
            cmes = queryDONKI.CME(self.simstart, self.simstop)
        else:
            return
        
        for index, row in cmes.iterrows():
            # Extract CME Analysis info
            info = row['cmeAnalyses']
            
            # Setup a dict to hold CME params
            cmeDistribution_dict = {}
            
            t = (datetime.datetime.strptime(info['time21_5'], "%Y-%m-%dT%H:%MZ") - self.simstart).total_seconds()
            cmeDistribution_dict['t_mu'] = t
            cmeDistribution_dict['t_sig'] = t_sig_init
            
            cmeDistribution_dict['lon_mu'] = info['longitude']
            cmeDistribution_dict['lon_sig'] = lon_sig_init
            
            cmeDistribution_dict['lat_mu'] = info['latitude']
            cmeDistribution_dict['lat_sig'] = lat_sig_init
            
            cmeDistribution_dict['width_mu'] = 2*info['halfAngle']
            cmeDistribution_dict['width_sig'] = width_sig_init
            
            cmeDistribution_dict['speed_mu'] = info['speed']
            cmeDistribution_dict['speed_sig'] = speed_sig_init
            
            cmeDistribution_dict['thickness_mu'] = thick_mu_init
            cmeDistribution_dict['thickness_sig'] = thick_sig_init
            
            cmeDistribution_dict['innerbound'] = 21.5
            
            self.cmeDistribution.loc[index, :] = cmeDistribution_dict
         
        # cmeDistribution = pd.DataFrame(cmeDistribution_dict)
        
        # Drop CMEs at high lat
        lat_cutoff = np.abs(self.cmeDistribution['lat_mu']) > 2.0*self.latmax
        self.cmeDistribution.loc[lat_cutoff, 'lat_mu'] = np.nan
        
        # Drop NaNs
        self.cmeDistribution.dropna(how='any', axis='index', inplace = True)
        
        # self.cmeDistribution = cmeDistribution
        
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
    
    def predict_withDask(self, boundarySamples_U, cmeSamples, observer_name, dpadding=0.03):
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
    
    def predict(self, boundarySamples_U, cmeSamples, observer_name, dpadding=0.03):
        import multiprocessing as mp
        from tqdm import tqdm
        from joblib import Parallel, delayed
        
        t0 = time.time()
        nSamples = len(boundarySamples_U)
        
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
            for boundarySample_U, cmeSample in zip(boundarySamples_U, cmeSamples)
            )
        
        ensemble = list(tqdm(futureGenerator, total=nSamples))
        
        print("{} HUXt forecasts completed in {}s".format(len(ensemble), time.time()-t0))
        
        # =============================================================================
        # Visualize    
        # =============================================================================
        # fig, ax = plt.subplots(figsize=(6,4.5))
        
        # for member in ensemble:
        #     ax.plot(member['mjd'], member['U'], color='C3', lw=1, alpha=0.2)
        # ax.plot(member['mjd'][0:1], member['U'][0:1], lw=1, color='C3', alpha=1, 
        #         label = 'Ensemble Members')
        
        
        # ax.legend(scatterpoints=3, loc='upper right')
        
        # ax.set(xlim=[self.starttime.mjd, self.stoptime.mjd])
        # ax.set(xlabel='Date [MJD], from {}'.format(datetime.datetime.strftime(self.start, '%Y-%m-%d %H:%M')), 
        #        ylabel='Solar Wind Speed [km/s]', 
        #        title='HUXt Ensemble @ {}'.format(observer_name))
        
        # plt.show()
        
        # Save ensemble
        self.current_ensemble = ensemble
        
        return ensemble
    
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
    

# Define an inner function to be run in parallel
def _map_vBoundaryInwards(simstart, simstop, source, insitu_df, corot_type, innerbound):
    
    # Reformat for HUXt inputs expectation
    insitu_df['BX_GSE'] =  -insitu_df['BR']
    insitu_df['V'] = insitu_df['U_mu']
    insitu_df['datetime'] = insitu_df.index
    insitu_df = insitu_df.reset_index()
    
    # Generate the Carrington grids
    t, vcarr, bcarr = Hin.generate_vCarr_from_insitu(simstart, simstop, 
                                                     insitu_source=source, insitu_input=insitu_df, 
                                                     corot_type=corot_type)
    
    # Map to 210 solar radii, then to the inner boundary for the model
    vcarr_inner = vcarr.copy()
    for i, _ in enumerate(t):
        vcarr_inner[:,i] = Hin.map_v_boundary_inwards(vcarr[:,i]*u.km/u.s,
                                                     (insitu_df['rad_HGI'][i]*u.AU).to(u.solRad),
                                                     innerbound)
    return vcarr_inner

# def _process_sample(df_sample, method_sample):
#     sf_sample_copy = df_sample.copy(deep=True)
#     insitu_df_copy['V'] = U_sample
#     return map_vBoundaryInwards(source, insitu_df_copy, method_sample)

# # %%
# if __name__ == '__main__':
#     import generate_external_input
#     # =========================================================================
#     # THIS SHOULD ALL BE MOVED TO A NOTEBOOK WHEN WORKING!
#     # =========================================================================
    
#     # ========================================================================
#     # Initialize an MSIR inputs object
#     # =========================================================================
#     start = datetime.datetime(2012, 1, 1)
#     stop = datetime.datetime(2012, 7, 1)
#     rmax = 10 # AU
#     latmax = 15
    
#     inputs = multihuxt_inputs(start, stop, rmax=rmax, latmax=latmax)
#     # =============================================================================
#     # Search for available background SW and transient data
#     # =============================================================================
#     inputs.get_availableBackgroundData()
#     inputs.filter_availableBackgroundData()
#     # inputs.sort_availableSources('rad_HGI')
    
#     # Get ICME/IPS data for all available source
#     inputs.get_availableTransientData()
    
#     # =============================================================================
#     # Generate background and boundary distributions:
#     #   - Remove ICMEs
#     #   - GP interpolate 1D in-situ time series
#     #   - Backmap to 21.5 RS
#     #   - GP interpolate 3D (time, lon, lat) source model
#     # =============================================================================
    
#     # Generate an input CME distribution
#     inputs.generate_cmeDistribution()
    
#     inputs.generate_backgroundDistributions()
    
#     inputs.generate_boundaryDistributions(nSamples=16, constant_sig=0)
    
#     # Either choose one boundary distribution, or do a 3D GP interpolation
#     # inputs.generate_boundaryDistribution3D(nLat=32, extend='omni', GP=False)
#     inputs.generate_boundaryDistribution3D(nLat=32, GP=True)
    

    
#     breakpoint()

#     # Add Saturn SKR Data
#     saturn_df = generate_external_input.Cassini_SKR(inputs.availableBackgroundData.index)
#     inputs.availableBackgroundData = pd.merge(inputs.availableBackgroundData, 
#                                               saturn_df,
#                                               left_index=True, right_index=True)
    
#     nSamples = 16
#     weights = [1/nSamples]*nSamples
    
#     # for source in ...
#     source = 'saturn'
    
#     boundarySamples, cmeSamples = inputs.sample3D(weights, at=source)
    
#     ensemble = inputs.predict2(boundarySamples, cmeSamples, source)
    
#     # Save as checkpoint
#     with open('/Users/mrutala/projects/OHTransients/inputs_checkpoint.pkl', 'wb') as f:
#         pickle.dump(inputs, f)
         
#     with open('/Users/mrutala/projects/OHTransients/inputs_checkpoint.pkl', 'rb') as f:
#         inputs = pickle.load(f)
    
#     # CIME interaction time @ Saturn (Palmerio+ 2021)
#     interaction_time = datetime.datetime(2012, 6, 12, 00, 00)
    