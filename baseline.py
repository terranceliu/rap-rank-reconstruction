import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import get_args, get_base_dir, get_data, convert_data2string, add_ppmf_block, append_results

import pdb

args = get_args()
assert args.process_data == 'split'

base_dir = get_base_dir(args, reconstruct=False)

data, data_split = get_data(args, return_both_splits=True)

true_samples = convert_data2string(data.df.drop_duplicates())
df_split = data_split.df

counts_path = './out/ppmf_counts/{}.csv'.format(args.dataset_baseline)
if os.path.exists(counts_path):
    counts = pd.read_csv(counts_path, index_col=0, header=0)['0']
else:
    x = convert_data2string(df_split)
    x, unique_samples = pd.factorize(x)
    counts = np.bincount(x)
    counts = pd.Series(counts, index=unique_samples)
    counts = counts.sort_values(ascending=False)[:args.max_candidates]
    if args.save_counts:
        counts.to_csv(counts_path)

if args.dataset_baseline is not None and args.dataset_baseline.startswith('ppmf') and not args.ignore_block:
    counts = add_ppmf_block(counts, data.df, max_candidates=args.max_candidates)

ranked_candidates_grouped = [x.index.values for _, x in counts.groupby(counts)][::-1]
for i in tqdm(range(len(ranked_candidates_grouped))):
    x = ranked_candidates_grouped[i]
    ranked_candidates_grouped[i] = np.isin(x, true_samples, assume_unique=True)

thresholds = np.sort(counts.unique())[::-1]
num_candidates = np.array([len(x) for x in ranked_candidates_grouped])
num_matched = np.array([np.sum(x) for x in ranked_candidates_grouped])

if args.ignore_block:
    args.dataset += '-ib'
df_results = pd.DataFrame(
    {'dataset': args.dataset,
     'dataset_baseline': args.dataset_baseline if args.dataset_baseline is not None else 'split',
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