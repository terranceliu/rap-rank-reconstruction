import numpy as np
import pandas as pd

from plot_utils.utils import filter_masks
from plot_utils.ppmf_data_seeds import get_geoids
from plot_utils.ppmf_data_blocks import get_geoids_blocks

ALL_METHODS = [
    'RAP-RANK (random init)', 
    'RAP-RANK (init to D$_{holdout})$',
    'Baseline-D$_{holdout}$', 
    'Baseline-D$_{tract}$', 
    'Baseline-D$_{county}$', 
    'Baseline-D$_{state}$', 
    'Baseline-D$_{national}$',
]

METHOD_PLOT_PARAMS = {
    'RAP-RANK (random init)': {'c': 'tab:blue'}, 
    #
    'RAP-RANK (init to D$_{holdout})$': {'c': 'tab:pink'},
    'RAP-RANK (init to D$_{block})$': {'c': 'tab:pink'},
    'RAP-RANK (init to D$_{tract})$': {'c': 'tab:pink'},
    #
    'Baseline-D$_{holdout}$': {'c': 'tab:green'},
    'Baseline-D$_{block}$': {'c': 'tab:orange'}, 
    'Baseline-D$_{tract}$': {'c': 'tab:green'}, 
    'Baseline-D$_{county}$': {'c': 'tab:purple'}, 
    'Baseline-D$_{state}$': {'c': 'tab:brown'}, 
    'Baseline-D$_{national}$': {'c': 'tab:red'},
}

def get_datasets(experiment_type, seed=0):
    if experiment_type.startswith('tracts'):
        geoids = get_geoids(seed=seed)
    else:
        geoids = get_geoids_blocks()

    if experiment_type == 'tracts':
        datasets = ['ppmf_{}'.format(x) for x in geoids]
    else:
        datasets = ['ppmf_{}-ib'.format(x) for x in geoids]
    
    return datasets

def get_method_results(methods, dataset, dfs):
    df_reconstruct, df_reconstruct_init_split, df_baseline = dfs
    
    method_results = []

    for method in methods:
        if method == ALL_METHODS[0]:
            method_results.append(filter_masks(df_reconstruct, {'dataset': dataset}))
        elif method == ALL_METHODS[1]:
            method_results.append(filter_masks(df_reconstruct_init_split, {'dataset': dataset}))
        elif method == ALL_METHODS[2]:
            method_results.append(filter_masks(df_baseline, {'dataset': dataset, 'dataset_baseline': 'split'}))
        elif method == ALL_METHODS[3]:
            method_results.append(filter_masks(df_baseline, {'dataset': dataset, 'dataset_baseline': dataset[:16]}))
        elif method == ALL_METHODS[4]:
            method_results.append(filter_masks(df_baseline, {'dataset': dataset, 'dataset_baseline': dataset[:10]}))
        elif method == ALL_METHODS[5]:
            method_results.append(filter_masks(df_baseline, {'dataset': dataset, 'dataset_baseline': dataset[:7]}))
        elif method == ALL_METHODS[6]:
            method_results.append(filter_masks(df_baseline, {'dataset': dataset, 'dataset_baseline': 'ppmf_national'}))
        else:
            assert False, 'invalid method'
            
    return method_results

def get_fips_mapping():
    fips_mapping = pd.read_csv('../../datasets/raw/fips_mapping.csv')
    fips_mapping = pd.Series(fips_mapping['state_abbr'].values, index=fips_mapping['fips'].values)
    fips_mapping.index = [str(x).zfill(2) for x in fips_mapping.index.values]
    fips_mapping = fips_mapping.to_dict()
    return fips_mapping

def get_label(label, experiment_type):
    if label == 'Baseline-D$_{holdout}$':
        if experiment_type == 'blocks':
            label = 'Baseline-D$_{block}$'
        else:
            label = 'Baseline-D$_{tract}$'
    elif label == 'RAP-RANK (init to D$_{holdout})$':
        if experiment_type == 'blocks':
            label = 'RAP-RANK (init to D$_{block})$'
        else:
            label = 'RAP-RANK (init to D$_{tract})$'
    return label