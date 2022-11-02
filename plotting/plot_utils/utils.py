import numpy as np
from tqdm import tqdm

COLS_UNIQUE = ['dataset', 'data_dim', 'process_data', 'N', 'marginal', 'subsample', 'num_queries', 'attack_type',
               'num_models', 'K', 'T', 'sample_method', 'samples']

def filter_masks(df, mask_dict):
    mask = np.ones(len(df)).astype(bool)
    for key, val in mask_dict.items():
        mask &= df[key] == val
    if 'sample_method' in mask_dict.keys() and mask_dict['sample_method'] == 'sample' and 'samples' not in mask_dict.keys():
        mask &= df['samples'] == df['N']
    df = df.loc[mask].copy()
    df.sort_values('threshold', ascending=False, inplace=True)
    return df.reset_index(drop=True)

def check_results(df_results):
    for col in COLS_UNIQUE:
        if col in df_results.columns:
            assert len(df_results[col].unique()) == 1, '{}, {}'.format(col, df_results[col].unique())
    assert len(df_results['threshold'].unique()) == len(df_results)

def calculate_ordered_match_rate(df_results, step=1, min_confidence=2, tiebreak=False, seed=0):
    check_results(df_results)
    if min_confidence > 1:
        df_results = df_results.iloc[:-(min_confidence - 1)]

    mask = df_results['num_candidates'] != 0
    df_results = df_results[mask]

    num_candidates = df_results['num_candidates'].values
    if tiebreak:
        prng = np.random.RandomState(seed)
        num_matched = df_results['num_matched'].values
        matches = []
        for i in range(len(num_matched)):
            x = np.zeros(num_candidates[i])
            idxs = prng.choice(len(x), size=num_matched[i], replace=False)
            x[idxs] = 1
            matches.append(x)
        matches = np.concatenate(matches)
        x = np.cumsum(matches)
        x = x / (np.arange(len(x)) + 1)
    else:
        match_rates = df_results['match_rate_above_thresh'].values
        x = [np.ones(num_candidates[i]) * match_rates[i] for i in range(len(num_candidates))]
        x = np.concatenate(x)

    idxs = np.arange(len(x))[step-1::step]
    x = x[idxs]
    return x

def collect_results(get_method_results_fn,
                    datasets, all_methods, dfs,
                    step=1, max_matches=None, min_confidence=2, tiebreak=True
                    ):
    results = {}

    for dataset in tqdm(datasets):
        method_results = get_method_results_fn(all_methods, dataset, dfs)

        for i in range(len(method_results)):
            _min_confidence = min_confidence
            if all_methods[i].startswith('Baseline'):
                _min_confidence = 1
            method_results[i] = calculate_ordered_match_rate(method_results[i], step=step,
                                                             min_confidence=_min_confidence, tiebreak=tiebreak)

        num_keep = np.min([len(x) for x in method_results])
        if max_matches is not None:
            num_keep = np.minimum(num_keep, max_matches)
        for i in range(len(method_results)):
            method_results[i] = method_results[i][:num_keep]

        results[dataset] = dict(zip(all_methods, method_results))

    return results

def interpolate_results(results):
    results_interpolate = {}
    
    for dataset in results.keys():
        results_interpolate[dataset] = {}
        for method in results[dataset].keys():
            fp  = results[dataset][method]

            xp = np.arange(len(fp))
            xp = xp / xp.max()
            x = np.linspace(0, 1, 10000)

            results_interpolate[dataset][method] = np.interp(x, xp, fp)
            
    return results_interpolate

def get_results_avg(results):
    results_avg = {}
    counts_avg = np.zeros(10000)

    for dataset in results.keys():
        result = results[dataset]

        for method in result.keys():
            a = result[method]
            if method in results_avg.keys():
                b = results_avg[method]
                if len(a) < len(b):
                    results_avg[method] = b.copy()
                    results_avg[method][:len(a)] += a
                else:
                    results_avg[method] = a.copy()
                    results_avg[method][:len(b)] += b
            else:
                results_avg[method] = a
        counts_avg[:len(a)] += 1

    for key in results_avg.keys():
        results_avg[key] /= counts_avg[:len(results_avg[key])]
        
    return results_avg