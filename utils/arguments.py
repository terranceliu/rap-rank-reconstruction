import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--process_data', type=str, default='split',
                        help='Options:\n'
                             ' * split: Splits data in half, using one for the holdout set required for Baseline-D_holdout\n'
                             ' * full: Runs experiments on the full dataset'
                             ' * drop_dups: Runs experiments on the full dataset without duplicates'
                        )
    parser.add_argument('--marginal', type=int, default=2,
                        help='Determines k for k-way marginal queries. '
                             'Set as -1 for evaluating on PPMF data with Census queries'
                        )
    parser.add_argument('--subsample_queries', type=float, default=-1,
                        help='Options:\n'
                             ' * -1: Runs on all queries\n'
                             ' * interger > 0: Randomly samples <args.subsample_queries> queries'
                             ' * float between 0 and 1: Randomly samples <args.subsample_queries> of the total # of queries'
                        )
    # reconstruct
    parser.add_argument('--synth', type=str, default='fixed')
    parser.add_argument('--K', type=int, default=1000,
                        help='Corresponds to the # of rows of parameters in the generator. '
                             'For example, this is written down as N\' in the original RAP paper'
                        )
    parser.add_argument('--num_models', type=int, default=100, help='# runs of RAP-RANK to train')
    parser.add_argument('--init_split', action='store_true', help='Initializes to the split/holdout distribution')
    # train
    parser.add_argument('--T', type=int, default=1000, help='# step to train for each run')
    parser.add_argument('--max_idxs', type=int, default=1024, help='# queries to optimize over on each training step')
    # eval
    parser.add_argument('--eval_gen_method', type=str, default='sample')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='# samples to generate per model trained. '
                             'If set as -1, then <num_samples> corresponds to the size of the dataset N'
                        )
    parser.add_argument('--max_candidates', type=int, default=100000,
                        help='Evaluates results for only the top <max_candidates> candidates'
                        )
    parser.add_argument('--remove_existing_results', action='store_true', help='Overwrites existing results file')
    parser.add_argument('--eval_fixed_score', action='store_true')
    parser.add_argument('--eval_check_freq', action='store_true')
    # baseline
    parser.add_argument('--dataset_baseline', type=str, default=None, help='Specifies what dataset to use for the baseline')
    parser.add_argument('--save_counts', action='store_true', help='Saves intermediate files needed to generate baseline results')
    # ppmf
    parser.add_argument('--ignore_block', action='store_true',
                        help='Run experiments excluding the BLOCK attribute for Census (PPMF) datasets'
                        )

    args = parser.parse_args()

    if args.process_data not in ['split', 'full', 'drop_dups']:
        assert False, 'invalid argument for process_data'
    if args.init_split:
        assert args.process_data == 'split'
        assert args.synth == 'fixed'
    if args.ignore_block:
        assert args.dataset.startswith('ppmf'), 'ignore_blocked reserved for PPMF datasets'
    if args.marginal == -1:
        assert args.dataset.startswith('ppmf'), 'Census queries reserved fir PPMF datasets'
    assert args.subsample_queries == -1 or args.subsample_queries > 0
    if not args.subsample_queries.is_integer():
        assert args.subsample_queries < 0

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
