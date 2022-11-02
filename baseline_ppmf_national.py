import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils import get_data as _get_data
from utils import get_args, get_base_dir, get_data, convert_data2string, add_ppmf_block, append_results

ALL_FIPS = ['01', '02', '04', '05', '06', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22',
            '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
            '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56', '11'
            ]

counts_dir = './out/ppmf_counts/'
if not os.path.exists(counts_dir):
    os.makedirs(counts_dir)

counts_path = os.path.join(counts_dir, 'ppmf_national.csv')
if os.path.exists(counts_path):
    counts = pd.read_csv(counts_path, index_col=0, header=0)['0']
else:
    counts = None
    for fips in tqdm(ALL_FIPS):
        df = _get_data('ppmf_{}'.format(fips), root_path='./datasets/ppmf', cols=None).df
        df['TABBLK'] = 0

        x = convert_data2string(df)
        x, unique_samples = pd.factorize(x)
        x = np.bincount(x)
        x = pd.Series(x, index=unique_samples)

        if counts is None:
            counts = x
        else:
            counts = counts.add(x, fill_value=0)
    counts.to_csv(counts_path)

args = get_args()
assert args.dataset.startswith('ppmf')

base_dir = get_base_dir(args, reconstruct=False)

data = get_data(args)
if args.dataset.startswith('ppmf') and args.ignore_block:
    args.dataset += '-ib'
    data.df['TABBLK'] = 0

true_samples = convert_data2string(data.df.drop_duplicates())

counts = counts.sort_values(ascending=False)[:args.max_candidates]
if not args.ignore_block:
    counts = add_ppmf_block(counts, data.df, max_candidates=args.max_candidates)

ranked_candidates_grouped = [x.index.values for _, x in counts.groupby(counts)][::-1]
for i in tqdm(range(len(ranked_candidates_grouped))):
    x = ranked_candidates_grouped[i]
    ranked_candidates_grouped[i] = np.isin(x, true_samples, assume_unique=True)

thresholds = np.sort(counts.unique())[::-1]
num_candidates = np.array([len(x) for x in ranked_candidates_grouped])
num_matched = np.array([np.sum(x) for x in ranked_candidates_grouped])

df_results = pd.DataFrame(
    {'dataset': args.dataset,
     'dataset_baseline': 'ppmf_national',
     'data_dim': sum(data.domain.shape),
     'process_data': args.process_data,
     'N': len(data),
     'attack_type': 'baseline',
     'threshold': thresholds,
     'num_candidates': num_candidates,
     'num_matched': num_matched,
     'match_rate': np.divide(num_matched, num_candidates,
                             out=np.zeros_like(num_matched, dtype=float), where=num_candidates!=0),
     'num_candidates_above_thresh': np.cumsum(num_candidates),
     'num_matched_above_thresh': np.cumsum(num_matched),
     'match_rate_above_thresh': np.divide(np.cumsum(num_matched), np.cumsum(num_candidates),
                                   out=np.zeros_like(num_matched, dtype=float), where=np.cumsum(num_candidates)!=0),
     }
)

print(df_results)

results_dir = os.path.join('results', base_dir, 'baseline')
append_results(df_results, 'results.csv', results_dir, remove_existing=args.remove_existing_results)