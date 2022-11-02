import os
import pickle
import argparse
import pandas as pd

from plot_utils.utils import filter_masks, collect_results
from plot_utils.utils_ppmf import get_datasets, get_method_results, ALL_METHODS

def get_args():
    parser = argparse.ArgumentParser()

    # datasets
    parser.add_argument('--experiment_type', type=str, default='tract')
    # result filters
    parser.add_argument('--process_data', type=str, default='split')
    parser.add_argument('--subsample', type=str, default='all')
    parser.add_argument('--synth', type=str, default='fixed')
    parser.add_argument('--num_models', type=int, default=100)
    parser.add_argument('--K', type=int, default=1000)
    parser.add_argument('--T', type=int, default=1000)
    # plot params
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--max_matches', type=int, default=None)
    parser.add_argument('--min_confidence', type=int, default=2)
    parser.add_argument('--expected_match_rate', action='store_true')

    args = parser.parse_args()
    if args.max_matches is not None:
        args.max_matches /= args.step

    assert args.experiment_type in ['tracts', 'tracts_ib', 'blocks']

    return args

args = get_args()

mask_dict_reconstruct = {
    'process_data': args.process_data,
    'marginal': -1,
    'subsample': args.subsample,
    'attack_type': 'reconstruct-{}'.format(args.synth),
    'num_models': args.num_models,
    'K': args.K,
    'T': args.T,
    'sample_method': 'sample'
}
df_reconstruct = filter_masks(pd.read_csv('../results/results_reconstruct.csv'), mask_dict_reconstruct)

mask_dict_reconstruct_init = mask_dict_reconstruct.copy()
mask_dict_reconstruct_init['K'] = -1
df_reconstruct_init_split = filter_masks(pd.read_csv('../results/results_reconstruct_init_split.csv'), mask_dict_reconstruct_init)

mask_dict_baseline = {
    'process_data': args.process_data,
}
df_baseline = filter_masks(pd.read_csv('../results/results_baseline.csv'), mask_dict_baseline)

dfs = df_reconstruct, df_reconstruct_init_split, df_baseline

datasets = get_datasets(args.experiment_type)

all_methods = ALL_METHODS.copy()
if args.experiment_type != 'blocks':
    all_methods = [x for x in all_methods if x != ALL_METHODS[3]]
if args.experiment_type == 'tracts_ib':
    all_methods = [x for x in all_methods if x != ALL_METHODS[1]]

if not os.path.exists('./results'):
    os.makedirs('./results')
results_path = './results/ppmf_{}.pkl'.format(args.experiment_type)

results = collect_results(get_method_results,
                          datasets, all_methods, dfs,
                          step=args.step, max_matches=args.max_matches, min_confidence=args.min_confidence,
                          tiebreak=not args.expected_match_rate)

with open(results_path, 'wb') as handle:
    pickle.dump(results, handle)