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
import multihuxt_readers as mr
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
    
# %%
import tensorflow_probability  as     tfp
class multihuxt_inputs:
    def __init__(self, start, stop, 
                 rmax=1, latmax=10):
        self.start = start
        self.stop = stop
        self.radmax = rmax * u.AU
        self.latmax = latmax * u.deg
        self.innerbound= 21.5 * u.solRad
        
        self.usw_minimum = 200 * u.km/u.s
        self.SiderealCarringtonRotation = 27.28 * u.day
        self.SynodicCarringtonRotation = 25.38 * u.day
        
        # These keywords can only be set AFTER object initialization
        
        # ICME parameters
        self._icme_duration = 4.0 * u.day # conservative duration (Richardson & Cane 2010)
        self._icme_duration_buffer = 1.0 * u.day # conservative buffer (Richardson & Cane 2010)
        self._icme_interp_buffer = 1.0 * u.day
        
        # Required initializations
        # Other methods check that these are None (or have value) before 
        # continuing, so they must be intialized here
        self._availableSources = None
        self._boundarySources = None
        self._ephemeris = {}
        
        # Input data initialization
        cols = ['t_mu', 't_sig', 'lon_mu', 'lon_sig', 'lat_mu', 'lat_sig',
                'width_mu', 'width_sig', 'speed_mu', 'speed_sig', 
                'thickness_mu', 'thickness_sig', 'innerbound']
        self.cmeDistribution = pd.DataFrame(columns = cols)
        
        
        
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
        if self._availableSources is None:
            availableSources = set(self.availableBackgroundData.columns.get_level_values(0))
            availableSources = set(availableSources) - {'mjd'}
            self._availableSources = sorted(availableSources)
        return self._availableSources
    
    @availableSources.setter
    def availableSources(self, addedSources):
        self._availableSources.extend(addedSources)
        self._availableSources = sorted(self._availableSources)
        
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
    
    # def _identify_source(self, source):  
    #     source_aliases = {'omni': ['omni'],
    #                       'parker solar probe': ['parkersolarprobe', 'psp', 'parker solar probe'],
    #                       'stereo a': ['stereoa', 'stereo a', 'sta'],
    #                       'stereo b': ['stereob', 'stereo b', 'stb'],
    #                       # 'helios1': ['helios1', 'helios 1'],
    #                       # 'helios2': ['helios2', 'helios 2'],
    #                       'ulysses': ['ulysses', 'uy'],
    #                       # 'maven': ['maven'],
    #                       'voyager 1': ['voyager1', 'voyager 1'],
    #                       'voyager 2': ['voyager2', 'voyager 2']}
        
    def get_availableBackgroundData(self, sources=None):
        
        all_sources = ['omni', 'parker solar probe', 'stereo a', 'stereo b',
                       'ulysses', 'voyager 1', 'voyager 2']
        
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
            data_df = mr.SolarWindData(source, self.simstart, self.simstop).data
            if not data_df.isna().all().all(): 
                available_sources.append(source)
                available_data_dict[source] = data_df
                
        available_data_df = pd.concat(available_data_dict, axis='columns')
        available_data_df['mjd'] = Time(available_data_df.index).mjd
        
        # self.availableSources.extend(available_sources)
        self.availableBackgroundData = available_data_df
        
        return
    
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
    
    def get_availableTransientData(self, sources=None):
        
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
                                    duration = self._icme_duration,
                                    ensureCME = True) 

            icmes['affiliated_source'] = source
            
            availableTransientData_list.append(icmes)
        
        availableTransientData_list = [df for df in availableTransientData_list if not df.empty]
        availableTransientData_df = pd.concat(availableTransientData_list, axis='rows')
        availableTransientData_df.reset_index(inplace=True, drop=True)
        if len(availableTransientData_df) > 0:
            availableTransientData_df['mjd'] = Time(availableTransientData_df['eventTime'])

        self.availableTransientData = availableTransientData_df 
        
        return
    
    def get_indexICME(self, source):
        
        # Get the insitu data + mjd at this source
        insitu = self.availableBackgroundData[source].copy()
        insitu.loc[:, 'mjd'] = self.availableBackgroundData.loc[:, 'mjd']
        
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
                                             icme_buffer = self._icme_duration_buffer, 
                                             interp_buffer = self._icme_interp_buffer, 
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
                                         inducing_variable=True,
                                         GP = False, extend = False,
                                         target_noise = 1e-2,
                                         max_chunk_length = 1024,
                                         num_samples = 0):
        target_variables = ['U']
        
        # # Calculate the span from stop - start
        # span = (self.stop - self.start).total_seconds() * u.s
        # simspan = (self.simstop - self.simstart).total_seconds() * u.s
        
        # n_data = len(self.availableBackgroundData)
        
        # summary holds summary statistics (mean, standard deviation)
        all_summary = {}
        # samples holds individual samples drawn from the full covariance
        all_samples = {}
        
        for source in self.boundarySources:
            
            # If indexICME is not supplied, look it up
            if ICME_df is None:
                indexICME = self.get_indexICME(source)
            else:
                indexICME = ICME_df[source]
            
            # Where an ICME is present, set U, Br to NaN
            insitu_noICME = self.availableBackgroundData[source].copy()
            insitu_noICME['mjd'] = self.availableBackgroundData['mjd']
            insitu_noICME.loc[indexICME, ['U', 'Br']] = np.nan
            
            # new_insitu_list = []
            # covariance_list = []
            # sample_func_df = pd.DataFrame(columns=['start_mjd', 'stop_mjd', 'func'])
            
            # Pass the insitu DataFrame to the correct parser
            if (type(extend) == str) & (GP == True):
                print("Cannot have extend=str and GP=True!")
                return
            if type(extend) == str:
                print("Skipping GPR!")
                summary, samples = self._extend_backgroundDistributions(insitu_noICME,
                    target_variables=['U'],
                    num_samples=num_samples)
            elif GP is True:
                carrington_period = self.get_carringtonPeriod(self.ephemeris[source].r.mean())
                
                summary, samples = self._impute_backgroundDistributions(insitu_noICME, 
                    carrington_period,
                    target_variables=target_variables, 
                    target_noise=target_noise,
                    max_chunk_length=max_chunk_length,
                    num_samples=num_samples)
            else:
                print("You must set either GP or extend!")
                return
            
            # Cast res and samples into full dfs
            full_summary = insitu_noICME.copy(deep=True)
            full_summary.drop(columns=target_variables, inplace=True)
            for target_var in target_variables:
                full_summary[target_var+'_mu'] = summary[target_var+'_mu']
                full_summary[target_var+'_sigma'] = summary[target_var+'_sigma']
            
            full_samples = []
            for i in range(num_samples):
                full_sample_df = insitu_noICME.copy(deep=True)
                for target_var in target_variables:
                    full_sample_df[target_var] = samples[i][target_var]
                    
                full_samples.append(full_sample_df)
            
            # Add to dictionary with source
            all_summary[source] = full_summary
            all_samples[source] = full_samples
            
        bD = pd.concat(all_summary, axis=1)
        bD['mjd'] = self.availableBackgroundData['mjd']
        self.backgroundDistributions = bD
        
        bS = [pd.concat(dict(zip(all_samples.keys(), vals_by_sample)), axis=1) 
              for vals_by_sample in zip(*all_samples.values())]
        for i in range(len(bS)):
            bS[i]['mjd'] = self.availableBackgroundData['mjd']
        self.backgroundSamples = bS
        # self.backgroundCovariances = backgroundCovariances
        
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
                    (self.backgroundDistributions[(source, 'U_mu')] - 1.96 * self.backgroundDistributions[(source, 'U_sigma')]),
                    (self.backgroundDistributions[(source, 'U_mu')] + 1.96 * self.backgroundDistributions[(source, 'U_sigma')]),
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
    
    def _extend_backgroundDistributions(self, df,
                                        target_variables = ['U', 'Br'],
                                        noise_constant = 0.0,
                                        num_samples = 0):
        
        print("_extend_backgroundDistributions has not yet been implemented!")
        breakpoint()
        
        return backgroundDistribution_df, backgroundSamples
    
    def _impute_backgroundDistributions(self, df, carrington_period,
                                        target_variables = ['U', 'Br'],
                                        target_noise = 1e-2,
                                        max_chunk_length = 1024,
                                        num_samples = 0):
        import gpflow
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
        from sklearn.pipeline import Pipeline
        # from scipy.cluster.vq import kmeans
        from sklearn.cluster import KMeans
        import multiprocessing as mp
        from joblib import Parallel, delayed
        
        # Cap this at 0.01: we don't see much improvement beyond this, and it causes numerical uissues
        # if cluster_distance_limit < 1e-4:
        #     print("Cluster Distances smaller than ~0.0001 may cause numerical issues. Adjusting to 0.0001 now.")
        #     cluster_distance_limit = 1e-4
        
        backgroundDistribution_df = pd.DataFrame(index=df.index)
        backgroundSamples = [backgroundDistribution_df.copy() for _ in range(num_samples)]
        for target_var in target_variables:
        
            # Set up value scalers
            mjd_scaler = MinMaxScaler()
            mjd_scaler.fit(df['mjd'].to_numpy()[:,None])
            
            val_scaler = StandardScaler()
            val_scaler.fit(df[target_var].to_numpy()[:,None])
            
            # Normalizations & NaN removal
            def_indx = ~df['U'].isna().to_numpy()
            xmjd = mjd_scaler.transform(df['mjd'].to_numpy()[def_indx,None])
            yval = val_scaler.transform(df[target_var].to_numpy()[def_indx,None])
            
            # =================================================================
            # Define kernel for each dimension separately, then altogether
            # =================================================================
            period_rescaled = carrington_period.to(u.day).value * mjd_scaler.scale_[0]
            period_gp = gpflow.Parameter(period_rescaled, trainable=False)
            
            # Only predict 1 Carrington Rotation forward
            min_x = 0
            mid_x = period_rescaled / 2
            max_x = period_rescaled
            
            lengthscale_gp = gpflow.Parameter(mid_x, 
                transform = tfp.bijectors.SoftClip(min_x, max_x))
            
            base_kernel = gpflow.kernels.RationalQuadratic(lengthscales = lengthscale_gp)
            amplitude_kernel = gpflow.kernels.RationalQuadratic(lengthscales = lengthscale_gp)
            period_kernel = gpflow.kernels.Periodic(
                gpflow.kernels.SquaredExponential(lengthscales=period_gp),
                period=period_gp)
            
            kernel = base_kernel + amplitude_kernel * period_kernel
            
            # =============================================================================
            # ~Fancy~ Chunking  
            # =============================================================================
            X = xmjd
            Y = yval
            
            Xc, Yc, optimized_noise = self._optimize_clustering(X, Y, target_noise=target_noise, inX=True)
            XYc = np.column_stack([Xc, Yc])
            
            n_chunks = int(np.ceil(len(Xc)/max_chunk_length))
            
            sort = np.argsort(XYc[:,0]) # sort by MJD
            XYc_chunks = np.array_split(XYc[sort,:], n_chunks)
            Xc_chunks = [chunk[:,0][:,None] for chunk in XYc_chunks]
            Yc_chunks = [chunk[:,1][:,None] for chunk in XYc_chunks]
            
            # =============================================================================
            # Plug into the ensemble GP model
            # =============================================================================
            model = GPFlowEnsemble(kernel, Xc_chunks, Yc_chunks, optimized_noise)
            
            # =================================================================
            # Get predictions for all MJD (filling in gaps)
            # and inverse transform
            # =================================================================
            Xo = mjd_scaler.transform(df['mjd'].to_numpy()[:, None])
            fo_mu, fo_var = model.predict_f(Xo, chunk_size=2048, cpu_fraction=0.6)
            fo_sig = np.sqrt(fo_var)
            # yo_mu, yo_var = model.predict_y(Xo)
            # yo_sig = np.sqrt(yo_var)
            
            fo_samples = model.predict_f_samples(Xo, num_samples, chunk_size=2048, cpu_fraction=0.6)
            
            mjd = df['mjd'].to_numpy()
            val_f_mu = val_scaler.inverse_transform(fo_mu).flatten()
            val_f_sigma = (val_scaler.scale_ * fo_sig).flatten()
            val_f_samples = [val_scaler.inverse_transform(fo).flatten() for fo in fo_samples]
            # val_y_mu = val_scaler.inverse_transform(yo_mu).flatten()
            # val_y_sigma = (val_scaler.scale_ * yo_sig).flatten()
            
            backgroundDistribution_df['mjd'] = mjd
            backgroundDistribution_df[target_var+'_mu'] = val_f_mu
            backgroundDistribution_df[target_var+'_sigma'] = val_f_sigma
            
            for i in range(num_samples):
                backgroundSamples[i]['mjd'] = mjd
                backgroundSamples[i][target_var] = val_f_samples[i]
                
        return backgroundDistribution_df, backgroundSamples
            
        
    # def ambient_solar_wind_GP(self, df, average_cluster_span, carrington_period,
    #                           inducing_variable=False, target_variables=['U']):
    #     from sklearn.preprocessing import StandardScaler, MinMaxScaler
    #     import gpflow
    #     from sklearn.cluster import KMeans
        
    #     # Check that all target_variables are present
    #     for target_variable in target_variables:
    #         if df[target_variable].isna().all() == True:
    #             print("All variables are NaNs; cannot proceed with GPR")
    #             return None, None
        
    #     # Set up dictionaries 
    #     gp_variables = {k: {} for k in target_variables}
        
    #     for target_variable in target_variables:
                
    #         # Get the mjd and U as column vectors for GPR
    #         df_nonan = df.dropna(axis='index', how='any', subset=['U'])
    #         mjd = df_nonan['mjd'].to_numpy(float)[:, None]
    #         var = df_nonan[target_variable].to_numpy(float)[:, None]
    
    #         # MJD scaled to 1-day increments
    #         mjd_rescaler = MinMaxScaler((0, mjd[-1]-mjd[0]))
    #         mjd_rescaler.fit(mjd)
    #         X = mjd_rescaler.transform(mjd)

    #         # Variable rescaled to z-score
    #         var_rescaler = StandardScaler()
    #         var_rescaler.fit(var)
    #         Y = var_rescaler.transform(var)
    
    #         # K-means cluster the data (fewer data = faster processing)
    #         # And calculate the variance within each cluster
    #         XY = np.array(list(zip(X.flatten(), Y.flatten())))
    #         chunk_span = (df_nonan.index[-1] - df_nonan.index[0]).total_seconds() * u.s
    #         n_clusters = int((chunk_span / average_cluster_span).decompose())
            
    #         if n_clusters < len(XY):
    #             kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(XY)
    #             XYc = kmeans.cluster_centers_
    #             # # This requires each cluster to have multiple points, which is not guaranteed
    #             # # Instead, use the length scale as a guess at uncertaintys
    #             Yc_var = np.array([np.var(XY[kmeans.labels_ == i, 1]) 
    #                                for i in range(kmeans.n_clusters)])
    #             approx_noise = np.percentile(Yc_var, 90)
    #         else:
    #             n_clusters = len(XY)
    #             XYc = XY
    #             # 1/10 the overall standard deviation of Y
    #             # Yc_var = np.array([0.1]*len(XY))
    #             approx_noise = 0.1
    #         print("{} clusters to be fit".format(len(XYc)))
    #
    #        
    #         # Arrange clusters to be strictly increasing in time (X)
    #         Xc, Yc = XYc.T[0], XYc.T[1]
    #         cluster_sort = np.argsort(Xc)
    #         Xc = Xc[cluster_sort][:, None]
    #         Yc = Yc[cluster_sort][:, None]
    #         Yc_var = Yc_var[cluster_sort][:, None]
    #
    #         # Construct the signal kernel for GP
    #         # Again, these lengthscales could probable be calculated?
    #         # small_scale_kernel = gpflow.kernels.SquaredExponential(variance=2**2, lengthscales=1.0)
    #         # large_scale_kernel = gpflow.kernels.SquaredExponential(variance=2**2, lengthscales=10.0)
    #         # irregularities_kernel = gpflow.kernels.SquaredExponential(variance=1**2, lengthscales=1.0)
    #         # # Fixed period Carrington rotation kernel
    #         # period_rescaled = 100 / (simspan / carrington_period).decompose().value
    #         # carrington_kernel = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), 
    #         #                                             period=gpflow.Parameter(period_rescaled, trainable=False))
    #
    #         # signal_kernel = small_scale_kernel + large_scale_kernel + irregularities_kernel + carrington_kernel
    #        
    #         # New kernel
    #         period_rescaled = carrington_period.to(u.day).value
    #         signal_kernel = gpflow.kernels.RationalQuadratic() + \
    #                         gpflow.kernels.RationalQuadratic() * \
    #                         gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(),
    #                                                 period=gpflow.Parameter(period_rescaled, trainable=False)) 
    #        
    #         # This model *could* be solved with the exact noise for each point...
    #         # But this would require defining a custom likelihood class...
    #         if inducing_variable == False:
    #             model = gpflow.models.GPR((Xc, Yc), 
    #                                       kernel=copy.deepcopy(signal_kernel), 
    #                                       noise_variance=approx_noise
    #                                       )
    #         else:
    #             model = gpflow.models.SGPR((X, Y), 
    #                                        kernel=copy.deepcopy(signal_kernel), 
    #                                        noise_variance=approx_noise,
    #                                        inducing_variable=Xc)
    #            
    #         opt = gpflow.optimizers.Scipy()
    #         opt.minimize(model.training_loss, model.trainable_variables)
    #        
    #         # gpflow.utilities.print_summary(model)
    #        
    #         Xo = mjd_rescaler.transform(df['mjd'].to_numpy()[:, None])
    #        
    #         Yo_mu, Yo_var = model.predict_y(Xo)
    #         Yo_mu, Yoc_var = np.array(Yo_mu), np.array(Yo_var)
    #         Yo_sig = np.sqrt(Yo_var)
    #         _, cov = model.predict_f(Xo, full_cov=True)
    #        
    #         new_mjd = mjd_rescaler.inverse_transform(Xo)
    #         new_var_mu = var_rescaler.inverse_transform(Yo_mu)
    #         new_var_sig= Yo_sig * var_rescaler.scale_
    #         new_var_cov = cov * var_rescaler.scale_
    #        
    #         # Assign to dict
    #         gp_variables[target_variable]['mean'] = new_var_mu.ravel()
    #         gp_variables[target_variable]['std'] = new_var_sig.ravel()
    #         gp_variables[target_variable]['cov'] = new_var_cov.numpy().squeeze()
    #        
    #     return gp_variables
    
    def ambient_solar_wind_LI(self, df, target_variables=['U']):
        
        
        new_insitu = df.copy(deep=True)
        new_insitu.drop(columns='U', inplace=True)
        
        # Set up dictionaries 
        gp_variables = {k: {} for k in target_variables}
        
        for target_var in target_variables: # ['U', 'Br']:
            
            new_insitu[target_var+'_mu'] = df[target_var].interpolate(limit_direction=None)
            new_insitu[target_var+'_sig'] = new_insitu[target_var+'_mu'].rolling('1d').std()
            
            new_var_mu = df[target_var].interpolate(limit_direction=None)
            new_var_sig = new_var_mu.rolling('1d').std()
            new_var_cov = np.full((len(new_var_mu), len(new_var_mu)), np.nan)
            
            new_var_mu = new_var_mu.to_numpy()
            new_var_sig = new_var_sig.to_numpy()

            # # Replace non-ICME regions with real data
            # noNaN_bool = ~df[var_str].isna()
            # new_insitu.loc[noNaN_bool, 'U_mu'] = df.loc[noNaN_bool, 'U']
            # new_insitu.loc[noNaN_bool, 'U_sig'] *= 1/10.
            
            # # # Save a function to generate samples of f with full covariance
            # def func(mjd, num_samples):
            #     fo_samples = []
            #     for _ in range(num_samples):
            #         sample = new_insitu.query("@mjd[0] <= mjd <= @mjd[-1]")[var_str+'_mu']
            #         fo_samples.append(sample)
            #     fo_samples = np.array(fo_samples)
            #     return fo_samples
            
            # Assign to dict
            gp_variables[target_var]['mean'] = new_var_mu
            gp_variables[target_var]['std'] = new_var_sig
            gp_variables[target_var]['cov'] = new_var_cov
            
        return gp_variables
                 
    # def generate_backgroundSamples(self, num_samples):
    #     import gpflow
    #     from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
    #     # Check that the sampler functions have been defined
    #     try: 
    #         keys = self.backgroundCovariances.keys()
    #     except:
    #         print("generate_backgroundDistributions() must be run before background samples can be generated")
        
    #     # Check if we've already generated some samples?
    #     # breakpoint()
        
    #     # Set up a list of sample dataframes ahead of time
    #     backgroundSamples = [self.backgroundDistributions.copy(deep=True) for _ in range(num_samples)]            
        
    #     # Try my hand at sampling the covariance matrix directly
    #     for source in self.boundarySources:
    #         for var, vals in self.backgroundCovariances[source].items():
                
    #             time_list = vals['time']
    #             mu_list = vals['mean']
    #             std_list = vals['std']
    #             cov_list = vals['cov']
                
    #             rng = np.random.default_rng()
                
    #             for time, mu, std, cov in zip(time_list, mu_list, std_list, cov_list):
                    
    #                 # Get a rescaler to satisfy the matrix sampler
    #                 var_rescaler = StandardScaler()
    #                 var_rescaler.fit(mu[:, None])
                    
    #                 # Columnize the mean
    #                 mu_scaled = var_rescaler.transform(mu[:, None])
    #                 mu_for_sample = tf.linalg.adjoint(mu_scaled)
                    
    #                 # Columnize the standard deviation
    #                 sigma_scaled = std[:, None]/var_rescaler.scale_
                    
    #                 # Get randomly sampled mean + standard deviation (as backup)
    #                 samples_rng = [rng.normal(loc = mu_scaled, scale = sigma_scaled) 
    #                                for _ in range(num_samples)]
    #                 samples_rng = np.array(samples_rng).squeeze()
                    
    #                 # "Columnize" the covariance matrix
    #                 cov_scaled = cov / var_rescaler.scale_
    #                 cov_for_sample = cov_scaled[None, :, :]
                    
    #                 # Get samples
    #                 samples_cov = gpflow.conditionals.util.sample_mvn(
    #                     mu_for_sample, cov_for_sample, True, num_samples=num_samples)
    #                 samples_cov = samples_cov.numpy().squeeze()
                    
    #                 # Each successive chunk will overwrite the previous overlapping bit
    #                 # Since we've ensured continuity, this *seems* to be okay
    #                 for i, (sample_cov, sample_rng) in enumerate(zip(samples_cov, samples_rng)):
                        
    #                     sample_cov_unscaled = var_rescaler.inverse_transform(sample_cov[:, None])
    #                     sample_rng_unscaled = var_rescaler.inverse_transform(sample_rng[:, None])
                        
    #                     # Check if the Cholesky decomposition failed (all NaN?)
    #                     if np.isnan(sample_cov).any() == False:
                            
    #                         # If not, add the sample
    #                         backgroundSamples[i].loc[time, (source, var+'_mu')] = sample_cov_unscaled.flatten()
    #                         backgroundSamples[i].loc[time, (source, var+'_sig')] = 0
                            
    #                     else: 
                            
    #                         backgroundSamples[i].loc[time, (source, var+'_mu')] = sample_rng_unscaled.flatten()
    #                         # Sigma remains unchanged for this case
                    
    #     self.backgroundSamples = backgroundSamples
  
    #     return
        
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
        
        boundaryDistributions_d = {}
        boundarySamples_d = {}
        for source in self.boundarySources:
        
            # Format the insitu df (backgroundDistribution) as HUXt expects it
            insitu_df = self.backgroundDistributions[source].copy(deep=True)
            insitu_df['BX_GSE'] =  -insitu_df['Br']
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
                delayed(func)(self.simstart, self.simstop, source, df_sample, method_sample, self.ephemeris[source], self.innerbound)
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
            
            # Get the left edges of longitude bins
            lons = np.linspace(0, 360, vcarr_mu.shape[0]+1)[:-1]
            
            boundaryDistributions_d[source] = {'t_grid': t,
                                               'lon_grid': lons, 
                                               'U_mu_grid': vcarr_mu,
                                               'U_sig_grid': vcarr_sig,
                                               'B_grid': bcarr}
            
            # For completeness, add boundarySamples here
            boundarySamples_d[source] = []
            for result in results:
                boundarySamples_d[source].append({'t_grid': t,
                                                  'lon_grid': lons, 
                                                  'U_grid': result,
                                                  'B_grid': bcarr})
        
        self.boundaryDistributions = boundaryDistributions_d
        self.boundarySamples = boundarySamples_d
        
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
    
    def generate_boundaryDistribution3D(self, nLat=16, extend=None, GP=True, 
                                        num_samples=0, 
                                        **kwargs):
                                        # max_chunk_length=1024,
                                        # target_reduction = None, target_noise = None,
                                        # SGPR=0.1):
        
        # Get dimensions from OMNI boundary distribution, which *must* exist
        nLon, nTime = self.boundaryDistributions['omni']['U_mu_grid'].shape
        
        # Coordinates = (lat, lon, time)
        # Values = boundary speed, magnetic field* (*not implemented fully)
        lat_for3d = np.linspace(-self.latmax.value, self.latmax.value, nLat)
        lon_for3d = np.linspace(0, 360, nLon+1)[:-1]
        mjd_for3d = self.boundaryDistributions['omni']['t_grid']
        
        if (type(extend) == str) & (GP == True):
            print("Cannot have extend=str and GP=True!")
            return
        if type(extend) == str:
            U_mu_3d, U_sigma_3d, B_3d = self._extend_boundaryDistributions(nLat, extend)
        elif GP is True:
            U_mu_3d, U_sigma_3d, B_3d = self._impute_boundaryDistributions(lat_for3d, lon_for3d, mjd_for3d, num_samples=num_samples, **kwargs)
            
        self._assign_boundaryDistributions3D(mjd_for3d, lon_for3d, lat_for3d,
                                             U_mu_3d, U_sigma_3d, B_3d)
        
        return
    
    def _assign_boundaryDistributions3D(self, t_grid, lon_grid, lat_grid, U_mu_grid, U_sig_grid, B_grid):
        """
        This method is independent of generate_boundaryDistributions3D to allow
        assignment to attribute within the _extend and _impute methods, and 
        thus to allow easier testing

        Parameters
        ----------
        t_grid : TYPE
            DESCRIPTION.
        lon_grid : TYPE
            DESCRIPTION.
        lat_grid : TYPE
            DESCRIPTION.
        U_mu_grid : TYPE
            DESCRIPTION.
        U_sig_grid : TYPE
            DESCRIPTION.
        B_grid : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.boundaryDistributions3D = {'t_grid': t_grid,
                                        'lon_grid': lon_grid,
                                        'lat_grid': lat_grid,
                                        'U_mu_grid': U_mu_grid,
                                        'U_sig_grid': U_sig_grid,
                                        'B_grid': B_grid,
                                        }
        
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
        
    def _impute_boundaryDistributions(self, lat_for3d, lon_for3d, mjd_for3d,
                                      num_samples=0, **kwargs):
        import gpflow
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
        from sklearn.pipeline import Pipeline
        # from scipy.cluster.vq import kmeans
        from sklearn.cluster import KMeans
        import multiprocessing as mp
        from joblib import Parallel, delayed
        from sklearn.cluster import MiniBatchKMeans
        
        # Get dimensions from OMNI boundary distribution, which *must* exist
        nLat = len(lat_for3d)
        nLon = len(lon_for3d)
        nMjd = len(mjd_for3d)
        
        # Setup Normalizations ahead of time
        lat_scaler = MinMaxScaler((-1,1))
        lat_scaler.fit(lat_for3d[:,None])
        
        lon_scaler = MinMaxScaler((-1,1))
        lon_scaler.fit(lon_for3d[:,None])
        
        mjd_scaler = MinMaxScaler((-1,1))
        mjd_scaler.fit(mjd_for3d[:,None])

        # Initialize value scalers for mean (mu) and standard deviation (sigma)
        val_mu_scaler = StandardScaler()

        val_sigma_scaler = Pipeline([
            ('log_transform', FunctionTransformer(np.log1p, inverse_func=np.expm1, check_inverse=False)),
            ('scaler', StandardScaler()),
            ])
        
        #
        lat, lon, mjd, val_mu, val_sigma, = [], [], [], [], []
        for source in self.boundarySources:
            
            bound = self._rescale_2DBoundary(self.boundaryDistributions[source],
                                             target_reduction = kwargs.get('target_reduction'),
                                             target_size = kwargs.get('target_size'))
            
            lon_1d = bound['lon_grid']
            mjd_1d = bound['t_grid']
            lat_1d = np.interp(mjd_1d, 
                               self.ephemeris[source].time.mjd, 
                               self.ephemeris[source].lat_c.to(u.deg).value)
            
            mjd_2d, lon_2d, = np.meshgrid(mjd_1d, lon_1d)
            lat_2d, lon_2d, = np.meshgrid(lat_1d, lon_1d)
            
            val_mu_2d = bound['U_mu_grid']
            val_sigma_2d = bound['U_sig_grid']
            
            
            # We're going to transpose all of these 2D matrices
            # So, when flattened, lon is the second (faster changing) dim
            # And mjd/lat is the first (slower changing) dim
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
        # log_val_sigma = np.log10(val_sigma)
        
        # Normalizations & NaN removal
        xlat = lat_scaler.transform(lat[~np.isnan(val_mu),None])
        xlon = lon_scaler.transform(lon[~np.isnan(val_mu),None])
        xmjd = mjd_scaler.transform(mjd[~np.isnan(val_mu),None])
        
        val_mu_scaler.fit(val_mu[:,None])
        yval_mu = val_mu_scaler.transform(val_mu[~np.isnan(val_mu),None])
        
        val_sigma_scaler.fit(val_sigma[:,None])
        yval_sigma = val_sigma_scaler.transform(val_sigma[~np.isnan(val_mu),None])

        # %% ==================================================================
        # GP Kernel Definitions
        # =====================================================================
        
        # lat_scale_min = 0 * lat_scaler.scale_
        lat_scale_mid = 1 * lat_scaler.scale_
        # lat_scale_max = 3 * lat_scaler.scale_
        # lat_lengthscale = gpflow.Parameter(lat_scale_mid, 
        #    transform = tfp.bijectors.SoftClip(lat_scale_min, lat_scale_max))
        lat_lengthscale = gpflow.Parameter(lat_scale_mid)
        
        # mjd_scale_min = 0.0
        mjd_scale_mid = 3 * 25.38 * mjd_scaler.scale_
        # mjd_scale_max = 6 * 25.38 * mjd_scaler.scale_
        # if mjd_scale_mid > 0.9: mjd_scale_mid[0] = 0.9
        # if mjd_scale_max > 1.0: mjd_scale_max[0] = 1.0
        # mjd_lengthscale = gpflow.Parameter(mjd_scale_mid, 
        #    transform = tfp.bijectors.SoftClip(mjd_scale_min, mjd_scale_max))
        mjd_lengthscale = gpflow.Parameter(mjd_scale_mid)
        
        # lon_scale_min = np.float64(0.0)
        lon_scale_mid = np.float64(0.5)
        # lon_scale_max = np.float64(1.0)
        # lon_lengthscale = gpflow.Parameter(lon_scale_mid, 
        #    transform = tfp.bijectors.SoftClip(lon_scale_min, lon_scale_max))
        lon_lengthscale = gpflow.Parameter(lon_scale_mid)
        
        lat_kernel = gpflow.kernels.RationalQuadratic(active_dims=[0], lengthscales=lat_lengthscale)
        
        period_gp = gpflow.Parameter(1, trainable=False)
        base_kernel = gpflow.kernels.RationalQuadratic(active_dims=[1], lengthscales=lon_lengthscale)
        amplitude_kernel = gpflow.kernels.RationalQuadratic(active_dims=[1], lengthscales=lon_lengthscale)
        period_kernel = gpflow.kernels.Periodic(
            gpflow.kernels.SquaredExponential(active_dims=[1], lengthscales=period_gp), 
            period=period_gp)
        lon_kernel = base_kernel + amplitude_kernel * period_kernel
                     
        mjd_kernel = gpflow.kernels.RationalQuadratic(active_dims=[2], lengthscales=mjd_lengthscale)
        
        all_kernel = gpflow.kernels.RationalQuadratic()
        kernel_mu = (lat_kernel + lon_kernel + mjd_kernel + 
                     lat_kernel*lon_kernel + lat_kernel*mjd_kernel + lon_kernel*mjd_kernel +
                     lat_kernel*lon_kernel*mjd_kernel + 
                     all_kernel)
        
        kernel_sigma = copy.deepcopy(kernel_mu)
        
        # %% ==================================================================
        # Optimize Clustering & Cluster
        # =====================================================================
        X = np.column_stack([xlat, xlon, xmjd])
        Y_mu = yval_mu
        Y_sigma = yval_sigma
        
        # Xc_mu, Yc_mu, opt_noise_mu = self._optimize_clustering(X, Y_mu, **kwargs)
        # # Xc_sigma, Yc_sigma, opt_noise_sigma = self._optimize_clustering(X, Y_sigma, 
        # #     target_reduction=target_reduction, target_noise=target_noise, inX=True)
        # Xc_sigma, Yc_sigma, opt_noise_sigma = self._optimize_clustering(X, Y_sigma, **kwargs)

        # XYc_mu = np.column_stack([Xc_mu, Yc_mu])
        # XYc_sigma = np.column_stack([Xc_sigma, Yc_sigma])
        
        opt_noise_mu = 0.05
        opt_noise_sigma = 0.05

        # %% ==================================================================
        # Chunk Data for Processing
        # =====================================================================
        
        Xc_mu_chunks, Yc_mu_chunks = self._optimize_chunking(X, Y_mu, **kwargs)
        
        Xc_sigma_chunks, Yc_sigma_chunks = self._optimize_chunking(X, Y_sigma, **kwargs)
        
        
        fig, axs = plt.subplots(ncols=len(Xc_mu_chunks), figsize=[10,5], subplot_kw={'projection': '3d'})
        for ax, Xchunk, Ychunk in zip(axs, Xc_mu_chunks, Yc_mu_chunks):
            ax.scatter(Xchunk[:,2], Xchunk[:,1], Xchunk[:,0], c=Ychunk[:,0], 
                       alpha=0.5, marker='.', s=36, vmin=-2, vmax=2)
            ax.set(xlabel='Time', ylabel='Longitude', zlabel='Latitude')
            ax.view_init(elev=30., azim=80)
        plt.show()

        # %% ==================================================================
        # Run the GP Regression
        # =====================================================================
        SGPR = kwargs.get('SGPR', 0.1)
        model_mu = GPFlowEnsemble(kernel_mu, Xc_mu_chunks, Yc_mu_chunks, opt_noise_mu, SGPR=SGPR)
        model_sigma = GPFlowEnsemble(kernel_sigma, Xc_sigma_chunks, Yc_sigma_chunks, opt_noise_sigma, SGPR=SGPR)
        # =============================================================================
        # Verify performance against input data    
        # =============================================================================
        model_mu_results = model_mu.predict_f(X, chunk_size=4096, cpu_fraction=0.75)
        model_sigma_results = model_sigma.predict_f(X, chunk_size=4096, cpu_fraction=0.75)
        diff_mu = model_mu_results[0] - Y_mu
        diff_sigma = model_sigma_results[0] - Y_sigma
        if (diff_mu.std() > 1) | (diff_sigma.std() > 1):
            breakpoint()

        fig, axs = plt.subplots(ncols=2, figsize=[10,5], subplot_kw={'projection': '3d'})
        temp_result_mu, temp_result_var = model_mu.predict_f(Xc_mu_chunks[1])
        for ax, Ychunk in zip(axs, [Yc_mu_chunks[1], temp_result_mu]):
            ax.scatter(Xc_mu_chunks[1][:,2], Xc_mu_chunks[1][:,1], Xc_mu_chunks[1][:,0], c=Ychunk[:,0], 
                       alpha=0.5, marker='.', s=36, vmin=-2, vmax=2)
            ax.set(xlabel='Time', ylabel='Longitude', zlabel='Latitude')
            ax.view_init(elev=30., azim=130)
        plt.show()
        
        fig, ax = plt.subplots(figsize=[5,5], subplot_kw={'projection': '3d'})

        ax.scatter(Xc_mu_chunks[1][:,2], Xc_mu_chunks[1][:,1], Xc_mu_chunks[1][:,0], c=temp_result_mu[:,0] - Yc_mu_chunks[1][:,0], 
                       alpha=0.5, marker='.', s=36, vmin=-1, vmax=1)
        
        iv0 = model_mu.model_list[1].inducing_variable.Z
        ax.scatter(iv0[:,2], iv0[:,1], iv0[:,0], color='black', marker='x', s=36)

        ax.set(xlabel='Time', ylabel='Longitude', zlabel='Latitude')
        ax.view_init(elev=30., azim=70)
        plt.show()
        
        
        # %% ==================================================================
        # Predict values for the full grid...     
        # =====================================================================
        Xlat, Xlon, Xmjd = np.meshgrid(lat_scaler.transform(lat_for3d[:,None]),
                                       lon_scaler.transform(lon_for3d[:,None]), 
                                       mjd_scaler.transform(mjd_for3d[:,None]),
                                       indexing='ij')
        X3d = np.column_stack([Xlat.flatten()[:,None],
                               Xlon.flatten()[:,None],
                               Xmjd.flatten()[:,None]])
        
        # Parallel chunk processing 
        fmu3d_mu, fmu3d_var = model_mu.predict_f(X3d, chunk_size=4096, cpu_fraction=0.75)
        # fmu_samples = model_mu.predict_f_samples(X3d, num_samples, chunk_size=4096, cpu_fraction=0.75)
        
        val_mu_mu = val_mu_scaler.inverse_transform(fmu3d_mu).reshape(nLat, nLon, nMjd)
        val_mu_sig = val_mu_scaler.scale_ * tf.sqrt(fmu3d_var).numpy().reshape(nLat, nLon, nMjd)
        

        
        # For the standard deviation
        fsig3d_mu, fsig3d_var = model_sigma.predict_f(X3d, chunk_size=4096, cpu_fraction=0.75)
        # fsig_samples = model_sigma.predict_f_samples(X3d, num_samples, chunk_size=4096, cpu_fraction=0.75)
        
        val_sig_mu = val_sigma_scaler.inverse_transform(fsig3d_mu).reshape(nLat, nLon, nMjd)
        
        # The uncertainty on the uncertainty is difficult to quantify
        
        # val_sig_sig = val_sigma_scaler.scale_ * tf.sqrt(fsig3d_var).reshape(nLat, nLon, nMjd)
        # test0 = val_sigma_scaler.inverse_transform(fsig3d_mu + tf.sqrt(fsig3d_var)).reshape(nLat, nLon, nMjd) - val_sig_mu
        # test1 = val_sig_mu - val_sigma_scaler.inverse_transform(fsig3d_mu - tf.sqrt(fsig3d_var)).reshape(nLat, nLon, nMjd)

        # !!!! Eventually, val will apply to both U and B...
        U_mu_3d = val_mu_mu
        U_sigma_3d = np.sqrt(val_mu_sig**2 + val_sig_mu**2)
        
        # Generate an OBVIOUSLY WRONG B
        B_3d = np.tile(self.boundaryDistributions['omni']['B_grid'], (64, 1, 1))
        
        # # %% ==================================================================
        # # TESTING PLOTS
        # # =====================================================================
        # print("Check for prediction quality!")
        # self._assign_boundaryDistributions3D(mjd_for3d, lon_for3d, lat_for3d, U_mu_3d, U_sigma_3d, B_3d)
        # test_atOMNI = self.sample_boundaryDistribution3D(at='omni')
        # test_atSTA = self.sample_boundaryDistribution3D(at='stereo a')
        
        # fig, axs = plt.subplots(nrows=2)
        # img = axs[0].pcolormesh(test_atOMNI['t_grid'], test_atOMNI['lon_grid'], 
        #                         test_atOMNI['U_mu_grid'] - self.boundaryDistributions['omni']['U_mu_grid'], 
        #                         vmin=-100, vmax=100)
        
        
        # img = axs[1].pcolormesh(test_atSTA['t_grid'], test_atSTA['lon_grid'], 
        #                         test_atSTA['U_mu_grid'] - self.boundaryDistributions['stereo a']['U_mu_grid'], 
        #                         vmin=-100, vmax=100)
        
        # fig.colorbar(img, ax=axs)
        # plt.show()
        
        # %% Return
        
        # Assign to self
        # Sample at OMNI/STA
        
        return U_mu_3d, U_sigma_3d, B_3d
        
        # =============================================================================
        # Visualization     
        # =============================================================================
        self._assign_boundaryDistributions3D(mjd_for3d, lon_for3d, lat_for3d, U_mu_3d, U_sigma_3d, B_3d)
        
        for source in self.boundarySources:
            
            # Reconstruct the backmapped solar wind view at each source
            fig, axs = plt.subplots(nrows=2)
            
            axs[0].imshow(self.boundaryDistributions[source]['U_mu_grid'],
                          vmin=200, vmax=600)
            
            boundary = self.sample_boundaryDistribution3D(source)
            _ = axs[1].imshow(boundary['U_mu_grid'],
                              vmin=200, vmax=600)
            
            fig.suptitle(source)
            # plt.colorbar(_, cax = ax)
            
            plt.show()
                
            
        # breakpoint()
    
        return
    
    def sample_boundaryDistribution3D(self, at=None):
        from scipy.interpolate import RegularGridInterpolator
        
        # !!!! Catch exceptions better...
        if at not in self.availableSources:
            breakpoint()
            
        shape3D = self.boundaryDistributions3D['U_mu_grid'].shape
            
        # Get the HGI latitude of the target at times matching the boundary grid
        target_lats = np.interp(self.boundaryDistributions3D['t_grid'],
                                self.availableBackgroundData['mjd'],
                                self.ephemeris[at].lat_c.to(u.deg).value)
        
        
        lon2d, t2d = np.meshgrid(self.boundaryDistributions3D['lon_grid'], 
                                 self.boundaryDistributions3D['t_grid'],
                                 indexing='ij')
        lon2d, lat2d = np.meshgrid(self.boundaryDistributions3D['lon_grid'],
                                   target_lats,
                                   indexing='ij')
        
        interp_mu = RegularGridInterpolator((self.boundaryDistributions3D['lat_grid'], 
                                             self.boundaryDistributions3D['lon_grid'], 
                                             self.boundaryDistributions3D['t_grid']), 
                                            self.boundaryDistributions3D['U_mu_grid'])
        
        U_mu_2d = interp_mu(np.column_stack((lat2d.flatten(), lon2d.flatten(), t2d.flatten()))).reshape(lon2d.shape)
        
        interp_sigma = RegularGridInterpolator((self.boundaryDistributions3D['lat_grid'], 
                                             self.boundaryDistributions3D['lon_grid'], 
                                             self.boundaryDistributions3D['t_grid']), 
                                            self.boundaryDistributions3D['U_sig_grid'])
        
        U_sigma_2d = interp_sigma(np.column_stack((lat2d.flatten(), lon2d.flatten(), t2d.flatten()))).reshape(lon2d.shape)
        
        # AGAIN, CLEARLY WRONG B!!!!
        B_grid = self.boundaryDistributions3D['B_grid'][0,:,:]
        
        # breakpoint()
        # # Visualization
        # for i in range(0,len(self.boundaryDistributions3D['t_grid']),10):
        #     fig, ax = plt.subplots(figsize=(6,4))
            
        #     ax.pcolormesh(self.boundaryDistributions3D['lon_grid'],
        #                   self.boundaryDistributions3D['lat_grid'],
        #                   self.boundaryDistributions3D['U_sig_grid'][:,:,i]/self.boundaryDistributions3D['U_mu_grid'][:,:,i],
        #                   vmin=0, vmax=0.5)
        #     for source in self.ephemeris.keys():
        #         index = self.ephemeris[source].time.mjd == self.boundaryDistributions3D['t_grid'][i]
        #         xy_coord = (self.ephemeris[source].lon[index].to(u.deg).value,
        #                     self.ephemeris[source].lat[index].to(u.deg).value)
        #         ax.scatter(*xy_coord, marker='o', s=64, color='black', lw=2, fc='None')
        #         ax.annotate(source, xy_coord, (1,-1), 'data', 'offset fontsize')
                
        #     ax.set(xlim=[0,360], xlabel='Heliolongitude', 
        #            ylim=[-self.latmax.value, self.latmax.value], ylabel='Heliolatitude')
        #     plt.show()
        
        
        return {'t_grid': self.boundaryDistributions3D['t_grid'],
                'lon_grid': self.boundaryDistributions3D['lon_grid'],
                'U_mu_grid': U_mu_2d, 
                'U_sig_grid': U_sigma_2d,
                'B_grid': B_grid}
    
    def generate_cmeDistribution(self, search=True):
        
        # 
        t_sig_init = 3*3600 # seconds
        lon_sig_init = 10 # degrees
        lat_sig_init = 10 # degrees
        width_sig_init = 10 # degrees
        thick_mu_init = 4 # solar radii
        thick_sig_init = 1 # solar radii
        speed_sig_init = 200 # km/s
        
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
            
            # Do a bit of reformatting
            future.drop(columns=['r', 'lon'], inplace=True)
            future.rename(columns={'U': 'U', 'BX': 'Br'}, inplace=True)
            
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
        # !!!! ditch ephemeris info in these files
        
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
    
    def estimate(self, ensemble, weights, columns=None): # in loop
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
        
        if columns is None:
            columns = ['U', 'Br']
        
        for col in ensemble_columns:
            for index in metamodel.index:
                vals = [m.loc[index, col] for m in ensemble]
                valsort_indx = np.argsort(vals)
                cumsum_weights = np.cumsum(np.array(weights)[valsort_indx])
                
                weighted_median = vals[valsort_indx[np.searchsorted(cumsum_weights, 0.5 * cumsum_weights[-1])]]
                weighted_upper95 = vals[valsort_indx[np.searchsorted(cumsum_weights, 0.975 * cumsum_weights[-1])]]
                weighted_lower95 = vals[valsort_indx[np.searchsorted(cumsum_weights, 0.025 * cumsum_weights[-1])]]

                if col in columns:
                    metamodel.loc[index, col+"_median"] = weighted_median
                    metamodel.loc[index, col+"_upper95"] = weighted_upper95
                    metamodel.loc[index, col+"_lower95"] = weighted_lower95
                else:
                    metamodel.loc[index, col] = weighted_median
                    
                # breakpoint()
        
        return metamodel
    
    
    def _rescale_2DBoundary(self, bound, target_reduction=None, target_size=None):
        from scipy import ndimage
        from skimage.transform import rescale
        from skimage.measure import block_reduce
        
        data_shape = bound['U_mu_grid'].shape
        
        if target_reduction is None and target_size is None:
            target_reduction = 0.25
        elif target_reduction is not None:
            zoom_scale = np.sqrt(target_reduction)
        else:
            zoom_scale = np.sqrt(target_size/np.product(data_shape))
        
        new_bound = {}
        for key, val in bound.items():
            
            # Create a mask for valid (non-NaN) pixels
            mask = ~np.isnan(val)
            val_clean = np.where(mask, val, 0.0)
            
            # Resize both image and mask
            val_rescaled = rescale(val_clean, zoom_scale, 
                                  anti_aliasing=True, preserve_range=True)
            mask_rescaled = rescale(mask.astype(float), zoom_scale, 
                                  anti_aliasing=True, preserve_range=True)
            
            new_val = val_rescaled/mask_rescaled
            new_val[~mask_rescaled.astype(bool)] = np.nan
                
            new_bound[key] = new_val
            
        return new_bound
                                         
    # =========================================================================
    # Utility Functions 
    # (that could be separated from this file with no loss of generalization 
    # or context)
    # =========================================================================
    def _optimize_clustering(self, X, Y, **kwargs):
                             #target_reduction=None, target_noise=None, inX=None, inXY=None):
        from sklearn.cluster import MiniBatchKMeans
        from scipy.optimize import curve_fit
        
        target_reduction = kwargs.get('target_reduction')
        target_noise = kwargs.get('target_noise')
        inX = kwargs.get('inX')
        inXY = kwargs.get('inXY')
        
        # Cluster points by similar independent and dependent values
        XY = np.column_stack([X, Y])
        
        if inXY == True:
            to_fit = XY
        else:
            to_fit = X
            
        # Test values of n_clusters varying from clusters of size 5 to 40
        n_points, n_Xdim = X.shape
    
        # Handle target keywords
        if (target_reduction is None) & (target_noise is None):
            print("One of target_reduction or target_noise must be set.")
            print("Assuming target_noise of 1% (0.01).")
            target_noise = 0.01
        if target_reduction is not None:
            optimized_n_clusters = np.round(target_reduction * XY.shape[0]).astype(int)
        elif target_noise is not None:
            
            potential_n_clusters_bounds = [np.log10(n_points), 1000] # Handy lower bound
            potential_n_clusters = (n_points / np.linspace(*potential_n_clusters_bounds, 100)).astype(int)
            
            kmeans_stats = pd.DataFrame(columns=['mean variance', 'max variance'],
                                        index=np.unique(potential_n_clusters)[::-1])
            for n_clusters in tqdm.tqdm(kmeans_stats.index, desc='Optimizing Clustering for GP'):
                mbkmeans = MiniBatchKMeans(init="k-means++", n_clusters=n_clusters, batch_size=2048,
                                           n_init="auto", max_no_improvement=10, verbose=0)
                kmeans = mbkmeans.fit(to_fit)
                
                # tvpc = total variance per cluster
                tvpc = [(XY[kmeans.labels_ == c].std(axis=0)**2).sum() 
                        for c in range(kmeans.n_clusters)]
                
                kmeans_stats.loc[n_clusters, 'mean variance'] = np.mean(tvpc)
                kmeans_stats.loc[n_clusters, 'max variance'] = np.max(tvpc)
            
            # Which statistic do we care about?
            target_var = 'mean variance'
            kmeans_stats = kmeans_stats.dropna(axis = 'index')
            
            # Fit the trend to get rid of minibatch 'jaggedness'
            # def trend(x, a, b, c): 
            #     return a * 1/np.log10(x + b) + c
            def loglog_trend(x, a, b):
                return a * x + b
            coeffs, cov = curve_fit(loglog_trend, 
                                    np.log10(kmeans_stats.index), 
                                    np.log10(kmeans_stats[target_var].to_numpy('float64')))
                
            # # Where does the mean variance equal our target?
            # if target < kmeans_stats[target_var].min():
            #     target = kmeans_stats[target_var].min()
            # if target > kmeans_stats[target_var].max():
            #     target = kmeans_stats[target_var].max()
            
            trend_x = (n_points / np.linspace(2, X.shape[0], 2000))
            trend_y = 10**loglog_trend(np.log10(trend_x), *coeffs)
            
            # By definition, this curve goes to 0 (well, NaN...) at n_points
            trend_x = np.insert(trend_x, 0, n_points)
            trend_y = np.insert(trend_y, 0, 0)
            
            fig, ax = plt.subplots()
            ax.scatter(kmeans_stats.index, kmeans_stats[target_var])
            ax.plot(trend_x, trend_y, color='black')
            ax.set(yscale='log', xscale='log')
            
            optimized_n_clusters = np.interp(target_noise, trend_y, trend_x).astype(int)
            # optimized_n_clusters = 10**np.round(optimized_n_clusters).astype(int)
        
        # Cluster data, then create chunks of clusters for optimization
        kmeans = KMeans(n_clusters=optimized_n_clusters,
                        random_state=0,
                        n_init="auto").fit(to_fit)
        
        # Calculate a final noise
        tvpc = [(XY[kmeans.labels_ == c].std(axis=0)**2).sum() 
                for c in range(kmeans.n_clusters)]
        variance = np.percentile(tvpc, 90) if np.percentile(tvpc, 90) > 0.0 else np.max(tvpc)
        
        if kmeans.cluster_centers_.shape[1] == X.shape[1]:
            Xc = kmeans.cluster_centers_
            Yc_list = [Y[kmeans.labels_ == c].mean(axis=0) 
                       for c in range(kmeans.n_clusters)]
            Yc = np.array(Yc_list)
        elif kmeans.cluster_centers_.shape[1] == XY.shape[1]:
            XYc = kmeans.cluster_centers_
            Xc = XYc[:,:n_Xdim]
            Yc = XYc[:,n_Xdim:]
        
        return Xc, Yc, variance
    
    def _optimize_chunking(self, X, Y, **kwargs):
        # Keywords
        max_chunk_length    = kwargs.get('max_chunk_length', 2048)
        byDimension         = kwargs.get('byDimension')
        byCluster           = kwargs.get('byCluster')
            
        #    
        XY = np.column_stack([X, Y])
        
        if (byDimension is None) & (byCluster is None):
            print("By default, chunking linearly in current order.")
            sort_indx = np.arange(0, XY.shape[0])
        elif byDimension is not None:
            sort_indx = np.argsort(XY[:,byDimension])
        elif byCluster is not None:
            sort_indx = np.arange(0, XY.shape[0])
        
        # Sort XY
        XY_sorted = XY[sort_indx,:]
        
        # Number of chunks
        nChunks = np.ceil(XY_sorted.shape[0] / max_chunk_length).astype(int)
        
        if byCluster is None:
            XY_chunks = np.array_split(XY_sorted, nChunks)
        else:
            if byCluster == 'X':
                kmeans = KMeans(n_clusters=nChunks).fit(X)
            else: # byCluster == 'XY':
                kmeans = KMeans(n_clusters=nChunks).fit(XY)
            XY_chunks = [XY[kmeans.labels_ == i, :] for i in range(kmeans.n_clusters)]
        
        X_chunks = [XY_chunk[:,:X.shape[1]] for XY_chunk in XY_chunks]
        Y_chunks = [XY_chunk[:,X.shape[1]:] for XY_chunk in XY_chunks]
        
        return X_chunks, Y_chunks
        
        # fig, axs = plt.subplots(ncols=n_chunks_mu, figsize=[10,5], subplot_kw={'projection': '3d'})
        # for ax, chunk in zip(axs, XYc_mu_chunks):
        #     ax.scatter(chunk[:,2], chunk[:,1], chunk[:,0], c=chunk[:,3], alpha=0.5, marker='.', s=36, vmin=-2, vmax=2)
        #     ax.set(xlabel='Time', ylabel='Longitude', zlabel='Latitude')
        #     ax.view_init(elev=30., azim=80)
        # plt.show()
        
        return
    

# Define an inner function to be run in parallel
def _map_vBoundaryInwards(simstart, simstop, source, insitu_df, corot_type, ephemeris, innerbound):
    
    # Reformat for HUXt inputs expectation
    insitu_df['BX_GSE'] =  -insitu_df['Br']
    insitu_df['V'] = insitu_df['U']
    insitu_df['datetime'] = insitu_df.index
    insitu_df = insitu_df.reset_index()
    
    # Generate the Carrington grids
    t, vcarr, bcarr = Hin.generate_vCarr_from_insitu(simstart, simstop, 
                                                     insitu_source=source, insitu_input=insitu_df, 
                                                     corot_type=corot_type)
    
    # Map to 210 solar radii, then to the inner boundary for the model
    vcarr_inner = vcarr.copy()
    for i, _ in enumerate(t):
        current_r = np.interp(t[i], ephemeris.time.mjd, ephemeris.r)
        vcarr_inner[:,i] = Hin.map_v_boundary_inwards(vcarr[:,i]*u.km/u.s,
                                                     current_r.to(u.solRad),
                                                     innerbound)
    return vcarr_inner







# %%
import gpflow
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
# from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
import multiprocessing as mp
from joblib import Parallel, delayed
class GPFlowEnsemble:
    def __init__(self, kernel, X_list, Y_list, noise_variance, weight_scaling=20, SGPR=1.0):
    
        if SGPR==1:
            self.type = 'GPR'
        else:
            self.type = 'SGPR'
            self.inducing_point_fraction = SGPR
            
        print(self.type)
    
        # Given variables
        self.kernel = kernel
        self.X_list = X_list
        self.Y_list = Y_list
        self.noise_variance = noise_variance
        self.weight_scaling = weight_scaling
        
        # Derived variables
        self.nChunks = len(X_list)
        
        self.model_list = []
        self.optimize_models()
    
    
    def optimize_models(self):
        import gpflow

        print("Optimizing {} GP models".format(self.nChunks))
        print("Current time: {}".format(datetime.datetime.now().strftime("%H:%M:%S")))
        t0 = time.time()
    
        for i, (X, Y) in enumerate(zip(self.X_list, self.Y_list)):
            print("Optimizing GP model #{} with {} points".format(i+1, len(X)))
            t1 = time.time()
            
            # Copy kernel so the model is freshly solved each loop
            kernel = copy.deepcopy(self.kernel)
            # kernel = self.kernel
            
            if self.type == 'GPR':
                model = gpflow.models.GPR((X, Y),
                                          kernel=kernel,
                                          noise_variance=self.noise_variance)
            elif self.type == 'SGPR':
                # aim for 20 points
                stepsize = int(np.round(1/self.inducing_point_fraction))
                print("Step size for SGPR: {}".format(stepsize))
                model = gpflow.models.SGPR((X, Y),
                                           kernel=kernel,
                                           noise_variance=self.noise_variance,
                                           inducing_variable=X[::stepsize,:],
                                           )
            else:
                breakpoint()
    
            opt = gpflow.optimizers.Scipy()
            opt.minimize(model.training_loss, model.trainable_variables)
            
            self.model_list.append(model)
            
            if i == 0: first_kernel_iter = copy.deepcopy(self.kernel)
            
            print("Completed in {:.1f} s".format(time.time() - t1))
            
        print("All GP models optimized in {:.1f} s".format(time.time() - t0))
        return
        
    #     weights = self.calculate_weights(X_new)
        
    #     result_mu = np.full((len(X_new), 1), 0, dtype='float64')
    #     result_sigma2 = np.full((len(X_new), 1), 0, dtype='float64')
    #     for w, model in zip(weights, self.model_list):
    #         f_mu, f_sigma2 = model.predict_f(X_new)
            
    #         result_mu += w[:,None] * f_mu.numpy()
    #         result_sigma2 += w[:,None] * f_sigma2.numpy()
            
    #     return result_mu, result_sigma2
    
    def predict_f(self, X_new, chunk_size=None, cpu_fraction=None):
        """
        Predict the values of f, the underlying function of GP regression, 
        without measurement errors.
        If chunksize is supplied, do the prediction in parallel.

        Parameters
        ----------
        X_new : TYPE
            DESCRIPTION.
        nCores : TYPE, optional
            DESCRIPTION. The default is None.
        chunksize : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        if chunk_size is None:
            chunk_size = len(X_new)
        if cpu_fraction is None:
            cpu_fraction = 0.50
        
        n_jobs = int(cpu_fraction * mp.cpu_count())
        
        X_new_chunked = [X_new[pos:pos + chunk_size] for pos in range(0, len(X_new), chunk_size)]
        
        def _predict_f(GPFlowEnsemble, _X):
            weights = GPFlowEnsemble.calculate_weights(_X)
            result_mu = np.full((len(_X), 1), 0, dtype='float64')
            result_sigma2 = np.full((len(_X), 1), 0, dtype='float64')
            for w, model in zip(weights, GPFlowEnsemble.model_list):
                f_mu, f_sigma2 = model.predict_f(_X)
                
                result_mu += w[:,None] * f_mu.numpy()
                result_sigma2 += w[:,None] * f_sigma2.numpy()
                
            return result_mu, result_sigma2
        
        # Avoid the parallelization overhead if chunk_size == len(X_new)
        if len(X_new_chunked) > 1:
            generator = Parallel(return_as='generator', n_jobs=n_jobs)(
                delayed(_predict_f)(self, X_chunk) for X_chunk in X_new_chunked)
        
            results = list(tqdm.tqdm(generator, total=len(X_new_chunked)))
        else:
            results = [_predict_f(self, X_new)]
        
        results_mu = np.vstack([r[0] for r in results])
        results_sigma2 = np.vstack([r[1] for r in results])
        
        return results_mu, results_sigma2
    
    # def predict_f_samples(self, X_new, num_samples=1):
        
    #     weights = self.calculate_weights(X_new)
        
    #     result = np.full((num_samples, len(X_new), 1), 0, dtype='float64')
    #     for w, model in zip(weights, self.model_list):
    #         f_samples = model.predict_f_samples(X_new, num_samples)
            
    #         result += np.tile(w[:,None], (num_samples, 1, 1)) * f_samples.numpy()
        
    #     return result
    
    def predict_f_samples(self, X_new, num_samples=1, chunk_size=None, cpu_fraction=None):
        """
        Predict the values of f, the underlying function of GP regression, 
        without measurement errors.
        If chunksize is supplied, do the prediction in parallel.

        Parameters
        ----------
        X_new : TYPE
            DESCRIPTION.
        nCores : TYPE, optional
            DESCRIPTION. The default is None.
        chunksize : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        if chunk_size is None:
            chunk_size = len(X_new)
        if cpu_fraction is None:
            cpu_fraction = 0.50
        
        n_jobs = int(cpu_fraction * mp.cpu_count())
        
        X_new_chunked = [X_new[pos:pos + chunk_size] for pos in range(0, len(X_new), chunk_size)]
        
        def _predict_f_samples(GPFlowEnsemble, _X):
            weights = GPFlowEnsemble.calculate_weights(_X)
            result = np.full((num_samples, len(_X), 1), 0, dtype='float64')
            for w, model in zip(weights, GPFlowEnsemble.model_list):
                f_samples = model.predict_f_samples(_X, num_samples)
                result += np.tile(w[:,None], (num_samples, 1, 1)) * f_samples.numpy()
                
            return result
        
        # Avoid the parallelization overhead if chunk_size == len(X_new)
        if len(X_new_chunked) > 1:
            generator = Parallel(return_as='generator', n_jobs=n_jobs)(
                delayed(_predict_f_samples)(self, X_chunk) for X_chunk in X_new_chunked)
        
            results = list(tqdm.tqdm(generator, total=len(X_new_chunked)))
        else:
            results = [_predict_f_samples(self, X_new)]
        
        results = np.concatenate(results, axis=1)
        
        return results
    
    # def predict_y(self, X_new):
        
    #     weights = self.calculate_weights(X_new)
        
    #     result_mu = np.full((len(X_new), 1), 0, dtype='float64')
    #     result_sigma2 = np.full((len(X_new), 1), 0, dtype='float64')
    #     for w, model in zip(weights, self.model_list):
    #         y_mu, y_sigma2 = model.predict_y(X_new)
            
    #         result_mu += w[:,None] * y_mu.numpy()
    #         result_sigma2 += w[:,None] * y_sigma2.numpy()
            
    #     return result_mu, result_sigma2
    
    def predict_y(self, X_new, chunk_size=None, cpu_fraction=None):
        if chunk_size is None:
            chunk_size = len(X_new)
        if cpu_fraction is None:
            cpu_fraction = 0.50
        
        n_jobs = int(cpu_fraction * mp.cpu_count())
        
        X_new_chunked = [X_new[pos:pos + chunk_size] for pos in range(0, len(X_new), chunk_size)]
        
        def _predict_y(GPFlowEnsemble, _X):
            weights = GPFlowEnsemble.calculate_weights(_X)
            result_mu = np.full((len(_X), 1), 0, dtype='float64')
            result_sigma2 = np.full((len(_X), 1), 0, dtype='float64')
            for w, model in zip(weights, GPFlowEnsemble.model_list):
                y_mu, y_sigma2 = model.predict_y(_X)
                
                result_mu += w[:,None] * y_mu.numpy()
                result_sigma2 += w[:,None] * y_sigma2.numpy()
                
            return result_mu, result_sigma2
        
        # Avoid the parallelization overhead if chunk_size == len(X_new)
        if len(X_new_chunked) > 1:
            generator = Parallel(return_as='generator', n_jobs=n_jobs)(
                delayed(_predict_y)(self, X_chunk) for X_chunk in X_new_chunked)
        
            results = list(tqdm.tqdm(generator, total=len(X_new_chunked)))
        else:
            results = [_predict_y(self, X_new)]
        
        results_mu = np.vstack([r[0] for r in results])
        results_sigma2 = np.vstack([r[1] for r in results])

        return results_mu, results_sigma2
    
    def calculate_weights(self, X_new):
        import scipy
        from scipy.spatial.distance import cdist
        
        # X_centers = [np.mean(X, axis=0) for X in self.X_list]
        
        # # Distances are n_chunks by n_X_new
        # distances = [np.linalg.norm(X_new - X_center, axis=1) for X_center in X_centers]
        # distances = np.stack(distances) 
        
        # weights = scipy.special.softmax(-distances, axis=0)
        
        min_distances = []
        for model in self.model_list:
            # Get only the X dimensions of the model data
            data = model.data[0]
            
            dist_matrix = cdist(data, X_new)
            min_dists = np.min(dist_matrix, axis=0)
            
            min_distances.append(min_dists)
            
        min_distances = np.array(min_distances)
        
        # Normalize min_distances to the distance expected after 
        norm_min_distances = min_distances / (1/len(self.X_list))
        norm_min_distances[norm_min_distances > 1] = 1
        
        weights = scipy.special.softmax(self.weight_scaling*(1-norm_min_distances), axis=0)
        
        return weights
    
    def print_summary(self):
        import gpflow
        for model in self.model_list:
            gpflow.utilities.print_summary(model, 'simple')
            
        df = pd.DataFrame()
        for i, model in enumerate(self.model_list):
            d = gpflow.utilities.parameter_dict(model)
            
            for key, value in d.items():
                df.loc[i, key] = value.numpy()
            
        return df

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
    