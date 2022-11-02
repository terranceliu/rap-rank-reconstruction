import numpy as np

from plot_utils.utils import filter_masks

ALL_METHODS = [
    'RAP-RANK (random init, 2-way)', 
    'RAP-RANK (random init, 3-way)',
    'Baseline-D$_{holdout}$',
]

METHOD_PLOT_PARAMS = {
    'RAP-RANK (random init, 2-way)': {'c': 'tab:blue'}, 
    'RAP-RANK (random init, 3-way)': {'c': 'tab:orange'},
    'Baseline-D$_{holdout}$': {'c': 'tab:green'},
}

def get_datasets(year, tasks, states):
    datasets = []
    for task in tasks:
        for state in states:
            dataset = 'folktables_{}_{}_{}'.format(task, year, state)
            datasets.append(dataset)
    return datasets


def get_method_results(methods, dataset, dfs):
    df_reconstruct, _, df_baseline = dfs
    
    method_results = []

    for method in methods:
        if method == ALL_METHODS[0]:
            method_results.append(filter_masks(df_reconstruct, {'dataset': dataset, 'marginal': 2}))
        elif method == ALL_METHODS[1]:
            method_results.append(filter_masks(df_reconstruct, {'dataset': dataset, 'marginal': 3}))
        elif method == ALL_METHODS[2]:
            method_results.append(filter_masks(df_baseline, {'dataset': dataset, 'dataset_baseline': 'split'}))
        else:
            assert False, 'invalid method'
            
    return method_results