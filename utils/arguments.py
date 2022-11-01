import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--process_data', type=str, default='full')
    parser.add_argument('--marginal', type=int, default=2)
    parser.add_argument('--subsample_queries', type=float, default=-1)
    # reconstruct
    parser.add_argument('--synth', type=str)
    parser.add_argument('--K', type=int, default=1000)
    parser.add_argument('--num_models', type=int, default=50)
    # train
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--max_idxs', type=int, default=1024)
    parser.add_argument('--init_split', action='store_true')
    # eval
    parser.add_argument('--eval_gen_method', type=str, default='argmax')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--eval_fixed_score', action='store_true')
    parser.add_argument('--eval_check_freq', action='store_true')
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--remove_existing_results', action='store_true')
    parser.add_argument('--max_candidates', type=int, default=100000)
    # baseline
    parser.add_argument('--dataset_baseline', type=str, default=None)
    parser.add_argument('--save_counts', action='store_true')
    # ppmf
    parser.add_argument('--ignore_block', action='store_true')


    args = parser.parse_args()

    if args.process_data not in ['full', 'drop_dups', 'split']:
        assert False, 'invalid argument for process_data'
    if args.init_split:
        assert args.process_data == 'split'
        assert args.synth == 'fixed'
    if args.ignore_block:
        assert args.dataset.startswith('ppmf'), 'ignore_blocked reserved for ppmf dataset'
    if args.subsample_queries > 0 and args.subsample_queries.is_integer():
        args.subsample_queries = int(args.subsample_queries)
    if args.eval_gen_method == 'argmax':
        args.num_samples = args.K

    return args

def get_base_dir(args, reconstruct=True):
    dir_subsample_queries = 'all' if args.subsample_queries < 0 else 'q_{}'.format(args.subsample_queries)

    dataset = args.dataset
    if dataset.startswith('ppmf') and args.ignore_block:
        dataset += '-ib'
    base = '{}/{}/'.format(dataset, args.process_data)
    if args.init_split:
        base = os.path.join(base, 'init_split')
    if reconstruct:
        queries = 'k_{}/{}'.format(args.marginal, dir_subsample_queries)
        params = '{}/K_{}/T_{}'.format(args.synth, args.K, args.T)
        base = os.path.join(base, queries, params)

    return base
