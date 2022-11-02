import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.algo.nondp import IterativeAlgoNonDP
from utils import get_args, get_base_dir, get_data, get_qm, get_model, append_results, \
    get_bincounts, calculate_matches

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args = get_args()
base_dir = get_base_dir(args)

data, data_split = get_data(args, return_both_splits=True)
K = args.K
if args.init_split:
    if args.K == -1:
        args.K = len(data_split)

query_manager = get_qm(args, data, device)

G = get_model(args.synth, query_manager, args.K, device, -1)['G']
algo = IterativeAlgoNonDP(G, -1)

model_base_dir = os.path.join('out', base_dir, 'runs/seed_{}')

print('Generating samples...')
samples = []
num_samples = len(data) if args.num_samples is None else args.num_samples
for seed in tqdm(range(args.num_models)):
    model_save_dir = model_base_dir.format(seed)
    algo.default_dir = model_save_dir
    algo.load('last.pt')

    if args.eval_gen_method == 'argmax':
        syndata = algo.G.get_syndata(num_samples=num_samples, how='max')
    elif args.eval_gen_method == 'sample':
        syndata = algo.G.get_syndata(num_samples=num_samples, how='sample')
    else:
        assert False, 'invalid argument: eval_gen_method'

    samples.append(syndata.df)

counts = get_bincounts(samples, check_freq=args.eval_check_freq, fixed_score=args.eval_fixed_score)
thresholds, num_candidates, num_matched, num_uniq = calculate_matches(counts, data.df, check_freq=args.eval_check_freq)

if args.ignore_block:
    args.dataset += '-ib'
subsample_queries = 'all' if args.subsample_queries < 0 else 'q_{}'.format(args.subsample_queries)
df_results = pd.DataFrame(
    {'dataset': args.dataset,
     'data_dim': sum(data.domain.shape),
     'process_data': args.process_data,
     'N': len(data),
     'N_uniq':num_uniq,
     'marginal': args.marginal,
     'subsample': subsample_queries,
     'num_queries': query_manager.num_queries,
     'attack_type': 'reconstruct-{}'.format(args.synth),
     'num_models': args.num_models,
     'K': K,
     'T': args.T,
     'sample_method': args.eval_gen_method,
     'samples': num_samples,
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

results_dir = os.path.join('results', base_dir)
wc = ''
if args.eval_check_freq:
    wc += '_freq'
if args.eval_fixed_score:
    wc += '_fixed'
append_results(df_results, 'results{}.csv'.format(wc), results_dir, remove_existing=args.remove_existing_results)