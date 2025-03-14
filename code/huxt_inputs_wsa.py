#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:58:11 2025

@author: mrutala
"""
import os
import httplib2
import numpy as np
import datetime
import requests
from bs4 import BeautifulSoup
import re
import tqdm
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time

sys.path.append('../HUXt/code/')
import huxt as H

def get_WSA_map_urls(start, stop, version='', source='', realization='', update_reference=False):
    
    all_versions = ['WSA5.4', 'WSA5.X']
    all_sources = ['ADAPT/GONG', 'GONG_Z', 'GONG_B']
    all_realizations = ['000', '001', '002', '003', '004', '005', 
                        '006', '007', '008', '009', '010', '011']
    
    if not version:
        version_order = all_versions
    else:
        version_order = [version.upper()]
        
    if not source:
        source_order = all_sources
    else:
        source_order = [source.upper()]
    
    if not realization:
        realization_order = all_realizations
    else:
        realization_order = [realization]
        
    # The skeleton url used for all files
    skeleton_url = ('https://iswa.gsfc.nasa.gov/iswa_data_tree/model/solar/'
                    '{version}/R21.5/WSA_VEL/{source}/{year}/{month}')
    suffixes = ['agong', 'gongz', 'gong_z', 'gongb', 'gong_b']
    filename = 'vel_{year}{month}??????R{realization}_{suffixes}.fits'
    column_format = '{version}/{source}/R{realization}'
     
    # Get the HUXt boundary condition directory & load the ISWA reference file
    dirs = H._setup_dirs_()
    _boundary_dir_ = dirs['boundary_conditions']
    reference_path = _boundary_dir_ + '/ISWAWSA_reference_file.pk'
    if os.path.exists(reference_path):
        reference_df = pd.read_pickle(reference_path)
        create_reference = False
    else:
        create_reference = True
    
    # Check if we need to update (or create) the local ISWA WSA url reference
    if update_reference or create_reference:
        
        # If we only need to update, get the last date in the existing reference
        if update_reference:
            breakpoint()
            
            
        df_list = []
        for version in all_versions:
            for source in all_sources:
                
                files = []
                uptime_fmt = ('[1-2][0-9][0-9][0-9]-'
                              '[0-1][0-9]-[0-3][0-9] '
                              '[0-2][0-9]:[0-6][0-9]')
                uptimes = []
                
                vs_url = skeleton_url.split('{year}')[0]
                vs_url = vs_url.format(version = version, source = source)

                # If the version-source url doesn't exist, break
                vs_r = requests.get(vs_url)
                if not vs_r.ok:
                    break
                
                # Parse the html page for years
                vs_html = BeautifulSoup(vs_r.text, 'lxml')
                years = []
                for a in vs_html.find_all('a'):
                    years.extend(re.findall('[1-2][0-9][0-9][0-9]', a['href']))
                
                for year in tqdm.tqdm(years):
                    
                    y_url = vs_url + '/' + year
                    
                    y_r = requests.get(y_url)
                    if not y_r.ok:
                        break
                    
                    # Parse the html page for months
                    y_html = BeautifulSoup(y_r.text, 'lxml')
                    months = []
                    for a in y_html.find_all('a'):
                        months.extend(re.findall('0[1-9]|1[0-2]', a['href']))
                        
                    for month in months:
                        
                        m_url = y_url + '/' + month
                        
                        m_r = requests.get(m_url)
                        if not m_r.ok:
                            break
                        
                        m_html = BeautifulSoup(m_r.text, 'lxml')
                        for a in m_html.find_all('a'):
                            if '.fits' in a['href']: 
                                files.append(a['href'])
                        for td in m_html.find_all('td'):
                            uptimes.extend(re.findall(uptime_fmt, str(td)))
                
                # At this point, we should have all the files for one version/source pair
                
                for realization in all_realizations:
                    column_name = column_format.format(version = version,
                                                       source = source,
                                                       realization = realization)
                    
                    # Select for a realization number
                    realization_indx = [True if 'R'+realization in f else False
                                        for f in files]
                    realization_files = np.array(files)[realization_indx]
                    realization_uptimes = np.array(uptimes)[realization_indx]
                    
                    # Pull dates out of filenames for df index
                    dates = [datetime.datetime.strptime(f[4:16], '%Y%m%d%H%M') 
                             for f in realization_files]
                    realization_uptimes = [datetime.datetime.strptime(u, '%Y-%m-%d %H:%M')
                                           for u in realization_uptimes]
                    
                    df = pd.DataFrame({column_name: realization_files,
                                       'uptimes': realization_uptimes},
                                      index = dates)
                    
                    # If multiple files have the same datetime, keep the most recent
                    df.index.name = 'df_index'
                    df = df.sort_values(by=['df_index', 'uptimes'], ascending=True)
                    df.loc[df.index.duplicated(keep='last'), :] = np.nan
                    df = df.dropna()
                    if df.index.duplicated().any():
                        breakpoint()
                    
                    # Ditch the upload time before adding to the list
                    df = df.drop('uptimes', axis='columns')
                    df_list.append(df)

                
                
        # Combine the list of dfs into one df   
        reference_df = pd.concat(df_list, axis=1)
        reference_df = reference_df.sort_index()
        
        # Drop all-NaN columns to save on space
        reference_df = reference_df.dropna(axis='columns', how='all')
        reference_df.to_pickle(reference_path)
    
    # If we've gotten here, we should have a reference_df containing lists of file names
    # Select the desired range
    query_df = reference_df.query('@start <= index < @stop')
    
    result_df = pd.DataFrame(index = np.arange(start, stop, datetime.timedelta(days=1)),
                             columns = ['filename'])
    filesfound = False
    
    # Sort reference_df columns by the supplied orders of preference
    ordered_column_names = []
    for version in version_order:
        for source in source_order:
            for realization in realization_order:
                col = column_format.format(version=version,
                                           source=source,
                                           realization=realization)
                if col in reference_df.columns:
                    ordered_column_names.append(col)
    
    # For each time in result_df, search within a day for the preferred closest file
    for index, row in result_df.iterrows():
        
        # Get the data within 1 day of the desired datetime
        inrange_start = index - datetime.timedelta(days=0.5)
        inrange_stop = index + datetime.timedelta(days=0.5)
        inrange_df = reference_df.query('@inrange_start <= index < @inrange_stop')
        
        # 
        for column_name in ordered_column_names:
            # If there's available file(s) in the reference, 
            # and the result row is still empty...
            reference_has_data = not inrange_df[column_name].isna().all()
            result_has_data = not row.isna().all()
            if (reference_has_data == True) and (result_has_data == False):
                
                # Set the result file to the nearest (timewise) reference file
                diff = [(t - index).total_seconds() for t in inrange_df.index]
                nearest_indx = np.argmin(np.abs(diff))
                result_df.loc[index, :] = inrange_df[column_name].iloc[nearest_indx]
                
    if result_df['filename'].isna().any() is True:
        if result_df['filename'].isna().all():
            print('No data coverage during specificed time period.')
            return -1
        else:
            print('Incomplete data coverage during specified time period. Returning incomplete dataframe...')
    return result_df
    


# def lookup_all_WSA():
#     # The skeleton url used for all files
#     baseurl_unformatted = 'https://iswa.gsfc.nasa.gov/iswa_data_tree/model/solar/{}/R21.5/WSA_VEL/{}'
    
#     # "Different" versions of WSA
#     possible_models = ['WSA5.X', 'WSA5.4']
    
#     # Different observatories or methods+observatories
#     possible_sources = ['ADAPT/GONG', 'GONG_B', 'GONG_Z']
    
#     # Get the years in each combination
#     # possible_urls = []
    
#     url_dict = {}
#     for model in possible_models:
#         for source in possible_sources:
            
#             # Create the url and check the server for a response
#             ms_url = baseurl_unformatted.format(model, source)
#             ms_response = requests.get(ms_url)
            
#             # If the response is good, get a list years on this page
#             if ms_response.ok:
#                 ms_urls = []
#                 possible_years = []
#                 year_regex = r'\b(1\d{3}|2\d{3})\b' # Matches 1000-2999
#                 bs = BeautifulSoup(ms_response.text, 'html.parser')
#                 for a in bs.find_all('a'):
#                     possible_years.extend(re.findall(year_regex, a.string))
                
#                 for year in sorted(possible_years):
#                     t1 = time.time()
#                     year_url = ms_url + '/' + year
#                     year_response = requests.get(year_url)
#                     print(time.time() - t1)
#                     # If the response is good, get a list of months
#                     if year_response.ok:
#                         possible_months = []
#                         month_regex = r'0[1-9]|1[0-2]' # Matches 01-12
#                         bs = BeautifulSoup(year_response.text, 'html.parser')
#                         for a in bs.find_all('a'):
#                             possible_months.extend(re.findall(month_regex, a.string))
                        
#                         # For each month, get all files
#                         for month in tqdm.tqdm(sorted(possible_months), 
#                                                '{}/{}/{}'.format(model, source, year)):
                            
#                             t2 = time.time()
#                             month_url = year_url + '/' + month
#                             month_response = requests.get(month_url)
#                             print(time.time() - t2)
#                             if month_response.ok:
#                                 month_files = []
#                                 bs = BeautifulSoup(month_response.text, 'html.parser')
                                
#                                 for a in bs.find_all('a'):
#                                     if '.fits' in a['href']:
#                                         month_files.append(a['href'])
                                
                            
#                                 # Add all hits to the list of all files
#                                 ms_urls.extend([month_url + '/' + f for f in month_files])
                                
#                             else:
#                                 breakpoint()
#                     else:
#                         breakpoint()
                                
#                 url_dict['{}/{}'.format(model, source)] = ms_urls
    
    
#     coverage_df_list = []
#     for ms in url_dict.keys():      
#         dates = [datetime.datetime.strptime(s.split('/')[-1][4:16], '%Y%m%d%H%M') 
#                  for s in url_dict[ms]]
        
#         # ADAPT runs 12 realizations, so the dates may have repeats. Ditch them.
#         dates = list(set(dates))
#         dates = sorted(dates)
        
#         # Append to list
#         coverage_df_list.append(pd.DataFrame(index = dates, 
#                                              columns = [ms], 
#                                              data = [1]*len(dates)))
        
    
        
#     # Concatenate
#     coverage_df = pd.concat(coverage_df_list, axis=1)
    
#     fig, ax = plt.subplots()
#     for y, ms in enumerate(coverage_df.columns):
#         ax.scatter(coverage_df.index, coverage_df[ms] * y,
#                    marker = '|', s=10, label = ms)
#     ax.axvline(datetime.datetime.now(), linestyle='--', color='black')
#     ax.legend()

# def get_WSA_boundary_conditions(starttime, endtime, observatory='', realization=''):
#     """
#     A function to grab the  solar wind speed (Vr) and radial magnetic field (Br) boundary conditions from MHDweb.
#     An order of preference for observatories is given in the function.
#     Checks first if the data already exists in the HUXt boundary condition folder.

#     Args:
#         cr: Integer Carrington rotation number
#         observatory: String name of preferred observatory (e.g., 'hmi','mdi','solis',
#             'gong','mwo','wso','kpo'). Empty if no preference and automatically selected.
#         runtype: String name of preferred MAS run type (e.g., 'mas','mast','masp').
#             Empty if no preference and automatically selected
#         runnumber: String Name of preferred MAS run number (e.g., '0101','0201').
#             Empty if no preference and automatically selected
#         masres: String, specify the resolution of the MAS model run through 'high' or 'medium'.

#     Returns:
#     flag: Integer, 1 = successful download. 0 = files exist, -1 = no file found.
#     """

#     # assert (np.isnan(cr) == False)

#     # The order of preference for different WSA run results
#     # overwrite = False
#     # if not masres:
#     #     masres_order = ['high', 'medium']
#     # else:
#     #     masres_order = [str(masres)]
#     #     overwrite = True  # If the user wants a specific observatory, overwrite what's already downloaded

#     if not observatory:
#         observatories_order = ['ADAPT/GONG']
#     else:
#         observatories_order = [str(observatory)]
#         overwrite = True  # If the user wants a specific observatory, overwrite what's already downloaded

#     # if not runtype:
#     #     runtype_order = ['masp', 'mas', 'mast']
#     # else:
#     #     runtype_order = [str(runtype)]
#     #     overwrite = True

#     # if not runnumber:
#     #     runnumber_order = ['0201', '0101']
#     # else:
#     #     runnumber_order = [str(runnumber)]
#     #     overwrite = True

#     # Get the HUXt boundary condition directory
#     dirs = H._setup_dirs_()
#     _boundary_dir_ = dirs['boundary_conditions']

#     # Example URL: https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/model/solar/WSA5.4/R21.5/WSA_VEL/ADAPT/GONG/2025/03/vel_202503100800R011_agong.fits
#     url_front = 'https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/model/solar/WSA5.4/R21.5/WSA_VEL'
#     url_end = '.fits'
    
#     # Make hourly filenames
#     delta = datetime.timedelta(hours=1)
#     possible_times = np.arange(starttime, endtime+delta, delta, dtype=datetime.datetime) 
#     possible_urls = []
#     for t in possible_times:
#         for r in np.arange(0, 12, 1):
#             url = url_front + '/' + '{}' + '/' + f'{t.year:04}' + '/' + f'{t.month:02}'
#             filename = 'vel_{}R{:03d}'.format(t.strftime('%Y%m%d%H%M'), r)
            
#             possible_urls.append(url + '/' + filename)
    
    
#     breakpoint()

#     if (os.path.exists(os.path.join(_boundary_dir_, brfilename)) is False or
#             os.path.exists(os.path.join(_boundary_dir_, vrfilename)) is False or
#             overwrite is True):  # Check if the files already exist

#         # Search MHDweb for a HelioMAS run, in order of preference
#         h = httplib2.Http(disable_ssl_certificate_validation=False)
#         foundfile = False
#         urlbase = None
#         for res in masres_order:
#             for masob in observatories_order:
#                 for masrun in runtype_order:
#                     for masnum in runnumber_order:
#                         urlbase = (heliomas_url_front + str(int(cr)) + '-' +
#                                    res + '/' + masob + '_' +
#                                    masrun + '_mas_std_' + masnum + '/helio/')
#                         url = urlbase + 'br' + heliomas_url_end

#                         # See if this br file exists
#                         resp = h.request(url, 'HEAD')
#                         if int(resp[0]['status']) < 400:
#                             foundfile = True

#                         # Exit all the loops - clumsy, but works
#                         if foundfile:
#                             break
#                     if foundfile:
#                         break
#                 if foundfile:
#                     break
#             if foundfile:
#                 break

#         if foundfile is False:
#             print('No data available for given CR and observatory preferences')
#             return -1

#         # Download the vr and br files
#         ssl._create_default_https_context = ssl._create_unverified_context

#         print('Downloading from: ', urlbase)
#         urllib.request.urlretrieve(urlbase + 'br' + heliomas_url_end,
#                                    os.path.join(_boundary_dir_, brfilename))
#         urllib.request.urlretrieve(urlbase + 'vr' + heliomas_url_end,
#                                    os.path.join(_boundary_dir_, vrfilename))

#         return 1
#     else:
#         print('Files already exist for CR' + str(int(cr)))
#         return 0