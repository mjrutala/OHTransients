#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:22:29 2025

@author: mrutala
"""

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries 
from sunpy.time import parse_time
from sunpy.data import cache
from sunpy.coordinates import spice, frames
from sunpy.coordinates import get_horizons_coord, get_body_heliographic_stonyhurst


from astropy import units as u
from astropy.time import Time


import matplotlib.pyplot as plt
import numpy as np 

try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

huxt_oh_dir = '/Users/mrutala/projects/huxt-oh/'
datasets_dict = {}

# %% Code for grabbing vSWIM (Mars Virtual Solar Wind Monitor)

#this reads a year of vSWIM hourly predictions into a pandas dataframe from a provided .csv
#file located at https://github.com/abbyazari/vSWIM/tree/main/Data

#format information is provided in https://github.com/abbyazari/vSWIM/blob/main/Data/format.md
vSWIM_skeleton_url = 'https://raw.githubusercontent.com/abbyazari/vSWIM/main/data/{filename}'
vSWIM_filenames = ['2014-2016_Hourly.csv',
                   '2016-2017_Hourly.csv',
                   '2017-2018_Hourly.csv',
                   '2018-2019_Hourly.csv',
                   '2019-2020_Hourly.csv',
                   '2020-2021_Hourly.csv',
                   '2021-2022_Hourly.csv',
                   '2022-2023_Hourly.csv',
                   '2023-2024_Hourly.csv']

df_list = []
for filename in vSWIM_filenames:
    df = pd.read_csv(vSWIM_skeleton_url.format(filename=filename), index_col=['Unnamed: 0'])
    df_list.append(df)
vSWIM_df = pd.concat(df_list, axis='index')
vSWIM_df.index = pd.to_datetime(vSWIM_df['date_[utc]'])

datasets_dict['vSWIM (@ Mars)'] = vSWIM_df

# %% 
# Mute sunpy warnings
import warnings
warnings.filterwarnings("ignore")

tstart = parse_time("2005-01-01 00:00")
tend = parse_time("2025-03-13 00:00")

omni_res = Fido.search(a.Time(tstart, tend), 
                       a.cdaweb.Dataset.omni2_h0_mrg1hr)

omni_files = Fido.fetch(omni_res, path=huxt_oh_dir+'/data/OMNI/')
while len(omni_files.errors) > 0:
    omni_files = Fido.fetch(omni_files, path=huxt_oh_dir+'/data/OMNI/')
omni_df = TimeSeries(omni_files, concatenate=True).to_dataframe()

datasets_dict['OMNI'] = omni_df

# %% Import STEREO A Data
sta_res = Fido.search(a.Time(tstart, tend), 
                      a.cdaweb.Dataset.sta_coho1hr_merged_mag_plasma)
sta_files = Fido.fetch(sta_res, path=huxt_oh_dir+'/data/STEREOA/')
while len(sta_files.errors) > 0:
    sta_files = Fido.fetch(sta_files, path=huxt_oh_dir+'/data/STEREOA/')
sta_df = TimeSeries(sta_files, concatenate=True).to_dataframe()

datasets_dict['STEREO A'] = sta_df

# %% Import STEREO B Data
stb_res = Fido.search(a.Time(tstart, tend), 
                      a.cdaweb.Dataset.stb_coho1hr_merged_mag_plasma)
stb_files = Fido.fetch(stb_res, path=huxt_oh_dir+'/data/STEREOB/')
while len(stb_files.errors) > 0:
    stb_files = Fido.fetch(stb_files, path=huxt_oh_dir+'/data/STEREOB/')
stb_df = TimeSeries(stb_files, concatenate=True).to_dataframe()

datasets_dict['STEREO B'] = stb_df

# %% Import PSP
PSP_res = Fido.search(a.Time(tstart, tend), 
                      a.cdaweb.Dataset.psp_coho1hr_merged_mag_plasma)
PSP_files = Fido.fetch(PSP_res, path=huxt_oh_dir+'/data/PSP/')
while len(PSP_files.errors) > 0:
    PSP_files = Fido.fetch(PSP_files, path=huxt_oh_dir+'/data/PSP/')
PSP_df = TimeSeries(PSP_files, concatenate=True).to_dataframe()

datasets_dict['Parker Solar Probe'] = PSP_df

# %% Import SolO
SolO_res = Fido.search(a.Time(tstart, tend), 
                      a.cdaweb.Dataset.solo_coho1hr_merged_mag_plasma)
SolO_files = Fido.fetch(SolO_res, path=huxt_oh_dir+'/data/SolO/')
while len(SolO_files.errors) > 0:
    SolO_files = Fido.fetch(SolO_files, path=huxt_oh_dir+'/data/SolO/')
SolO_df = TimeSeries(SolO_files, concatenate=True).to_dataframe()

datasets_dict['Solar Orbiter'] = SolO_df

# %% Grab the Ulysses data availability
Ulysses_res = Fido.search(a.Time(tstart, tend),
                          a.cdaweb.Dataset.uy_coho1hr_merged_mag_plasma)
Ulysses_files = Fido.fetch(Ulysses_res, path=huxt_oh_dir+'/data/Ulysses/')
while len(Ulysses_files.errors) > 0:
    Ulysses_files = Fido.fetch(Ulysses_files, path=huxt_oh_dir+'/data/Ulysses/')
Ulysses_df = TimeSeries(Ulysses_files, concatenate=True).to_dataframe()

datasets_dict['Ulysses'] = Ulysses_df

# %% Grab the WSA data availability
ISWAWSA_df = pd.read_pickle('/Users/mrutala/projects/huxt-oh/HUXt/data/boundary_conditions/ISWAWSA_reference_file.pk')

datasets_dict['WSA 5.4/ADAPT'] = ISWAWSA_df.loc[:, ['WSA5.4/ADAPT/GONG/R000', 'WSA5.4/ADAPT/GONG/R001',
                                                    'WSA5.4/ADAPT/GONG/R002', 'WSA5.4/ADAPT/GONG/R003',
                                                    'WSA5.4/ADAPT/GONG/R004', 'WSA5.4/ADAPT/GONG/R005',
                                                    'WSA5.4/ADAPT/GONG/R006', 'WSA5.4/ADAPT/GONG/R007',
                                                    'WSA5.4/ADAPT/GONG/R008', 'WSA5.4/ADAPT/GONG/R009',
                                                    'WSA5.4/ADAPT/GONG/R010', 'WSA5.4/ADAPT/GONG/R011']].dropna(how='all')
datasets_dict['WSA 5.4/GONGZ'] = ISWAWSA_df.loc[:, 'WSA5.4/GONG_Z/R000'].dropna(how='all')
datasets_dict['WSA 5.X/ADAPT'] = ISWAWSA_df.loc[:, ['WSA5.X/ADAPT/GONG/R000',
                                                    'WSA5.X/ADAPT/GONG/R001', 'WSA5.X/ADAPT/GONG/R002',
                                                    'WSA5.X/ADAPT/GONG/R003', 'WSA5.X/ADAPT/GONG/R004',
                                                    'WSA5.X/ADAPT/GONG/R005', 'WSA5.X/ADAPT/GONG/R006',
                                                    'WSA5.X/ADAPT/GONG/R007', 'WSA5.X/ADAPT/GONG/R008',
                                                    'WSA5.X/ADAPT/GONG/R009', 'WSA5.X/ADAPT/GONG/R010',
                                                    'WSA5.X/ADAPT/GONG/R011']].dropna(how='all')
datasets_dict['WSA 5.X/GONGZ'] = ISWAWSA_df.loc[:, 'WSA5.X/GONG_Z/R000'].dropna(how='all')
datasets_dict['WSA 5.X/GONGB'] = ISWAWSA_df.loc[:, 'WSA5.X/GONG_B/R000'].dropna(how='all')

# %% Grab the ICMECAT catalog
url='https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v22.csv'
icmecat_df = pd.read_csv(url)
icmecat_df.index = pd.to_datetime(icmecat_df['icme_start_time'])
icmecat_limited_df = icmecat_df.query('mo_sc_heliodistance > 2 & abs(mo_sc_lat_heeq) < 6')

datasets_dict['OH ICMEs'] = icmecat_limited_df

# %% Plot all datasets

fig, ax = plt.subplots()
yticks, yticklabels = [], []
for i, (key, val) in enumerate(datasets_dict.items()):
    
    yticks.append(i)
    yticklabels.append(key)
    
    ax.scatter(val.index, [i]*val.shape[0], marker='|', s=10)
    
ax.set(xlim = [tstart.to_datetime(), tend.to_datetime()], 
       yticks = yticks, yticklabels = yticklabels)
    