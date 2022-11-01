import os
import torch
import pickle
import numpy as np
from src.utils import Dataset
from src.qm import KWayMarginalQMTorch, KWayMarginalSetQMTorch
from src.utils import Domain, get_data as _get_data
from src.utils.data import get_all_workloads
from src.syndata import FixedGenerator, NeuralNetworkGenerator

def split_data(df, frac=0.5, seed=0):
    df_split1 = df.sample(frac=frac, random_state=seed)
    df_split2 = df.drop(df_split1.index)
    return df_split1.reset_index(drop=True), df_split2.reset_index(drop=True)

def get_data(args, root_path='./datasets/', return_both_splits=False):
    if args.dataset.startswith('ppmf'):
        root_path = os.path.join(root_path, 'ppmf')
    data = _get_data(args.dataset, root_path=root_path, cols=None)
    data_split = None

    df_data = data.df
    if args.process_data == 'full':
        pass
    elif args.process_data == 'drop_dups':
        data.df = df_data.drop_duplicates().reset_index(drop=True)
    elif args.process_data == 'split':
        df_data, df_split = split_data(df_data)
        data.df = df_data
        data_split = Dataset(df_split, data.domain)
    else:
        assert False, 'invalid argument for process_data'

    if args.dataset_baseline is not None:
        data_split = _get_data(args.dataset_baseline, root_path=root_path, cols=None)

    if args.dataset.startswith('ppmf') and args.ignore_block:
        config = data.domain.config
        config['TABBLK'] = 1
        new_domain = Domain.fromdict(config)

        data.df['TABBLK'] = 0
        data.domain = new_domain
        if data_split is not None:
            data_split.df['TABBLK'] = 0
            data_split.domain = new_domain

    if return_both_splits:
        return data, data_split
    return data

def filter_queries(args, query_manager, seed=0):
    if args.subsample_queries < 0:
        return

    num_subsample = args.subsample_queries
    if isinstance(num_subsample, float):
        num_subsample = int(num_subsample * query_manager.num_queries)
    prng = np.random.RandomState(seed)
    idxs = prng.choice(query_manager.num_queries, size=num_subsample, replace=False)
    query_manager.filter_queries(idxs)

def get_qm(args, data, device):
    if args.dataset.startswith('ppmf') and args.marginal == -1:
        queries_path = './datasets/ppmf/queries/{}-set.pkl'.format(args.dataset)
        with open(queries_path, 'rb') as handle:
            queries = pickle.load(handle)
        if args.ignore_block:
            for q in queries:
                if 'TABBLK' in q.keys():
                    del q['TABBLK']
            queries = [q for q in queries if len(q) > 0]

        query_manager = KWayMarginalSetQMTorch(data, queries, verbose=True, device=device)
    else:
        workloads = get_all_workloads(data, args.marginal)
        query_manager = KWayMarginalQMTorch(data, workloads, verbose=True, device=device)
    filter_queries(args, query_manager)
    return query_manager

def get_model(synth_type, query_manager, K, device, seed):
    model_dict = {}
    if synth_type == 'fixed':
        model_dict['G'] = FixedGenerator(query_manager, K=K, device=device, init_seed=seed)
        model_dict['lr'] = 1e-1
        model_dict['eta_min'] = None
    elif synth_type == 'nn':
        dim = query_manager.dim
        model_dict['G'] = NeuralNetworkGenerator(query_manager, K=K, device=device, init_seed=seed, resample=True,
                                                 embedding_dim=512, gen_dims=[1024, 1024],
                                                 # embedding_dim=512, gen_dims=[dim] * 2,
                                                 )
        model_dict['lr'] = 5e-4
        model_dict['eta_min'] = 1e-7
    else:
        assert False, 'invalid synthesizer'
    return model_dict


def update_model_init(G, oh_split, seed):
    weights = G.generator.syndata.weight
    if len(weights) != len(oh_split):
        prng = np.random.RandomState(seed)
        idxs = prng.choice(oh_split.shape[0], size=len(weights))
    else:
        idxs = np.arange(len(oh_split))
    mask = oh_split[idxs]

    maxes = weights.max(axis=1)[0].detach()
    mins = weights.min(axis=1)[0].detach() * 2
    # new_std = np.abs(mins.mean().item() * 0.1)

    weights_new = []
    st = 0
    for item in G.transformer.output_info:
        ed = st + item[0]
        x = weights[:, st:ed].detach()
        _mask = torch.tensor(mask[:, st:ed])

        x[_mask] = maxes
        x[~_mask] = mins.repeat_interleave(_mask.shape[-1] - 1)
        weights_new.append(x)
        st = ed

    weights_new = torch.cat(weights_new, axis=1)
    G.generator.syndata.weight = torch.nn.Parameter(weights_new)