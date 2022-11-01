import numpy as np
import pandas as pd
from tqdm import tqdm

def convert_data2string(df):
    vals = df.values.astype(str)
    vals = np.array([' '.join(x) for x in vals])

    # should be fine now, but this checks that nothing is getting truncated due to type casting
    test = np.array([[y.isdigit() for y in x.split(' ')] for x in vals])
    assert (test.sum(axis=-1) == df.shape[-1]).all()

    return vals

def convert_str2data(string_values, data_cols, dtype=int):
    values = np.array([x.split('-')[0].split(' ') for x in string_values]).astype(dtype)
    return pd.DataFrame(values, columns=data_cols)

def add_freq_suffix(x):
    unique_samples, counts = np.unique(x, return_counts=True)
    counts = pd.Series(counts, index=unique_samples)

    out = []
    for count in counts.unique():
        mask = counts == count
        y = counts[mask].index.values
        for i in np.arange(count):
            out.append(y + '-{}'.format(i))
    out = np.concatenate(out)

    return out

def get_bincounts(samples, check_freq=False, fixed_score=False):
    assert not (check_freq and fixed_score)
    samples_converted = []
    for df in samples:
        if fixed_score:
            df = df.drop_duplicates().reset_index(drop=True)
        x = convert_data2string(df)
        if check_freq:
            x = add_freq_suffix(x)
        samples_converted.append(x)
    samples = np.concatenate(samples_converted)
    unique_samples, counts = np.unique(samples, return_counts=True)
    counts = pd.Series(counts, index=unique_samples)
    return counts

def _calculate_matches_by_freq(counts, df_true):
    true_samples = convert_data2string(df_true)
    true_samples = add_freq_suffix(true_samples)
    true_counts = np.vectorize(lambda x: x.split('-')[1])(true_samples)

    thresholds = np.sort(counts.unique())[::-1]
    num_candidates = [] * len(thresholds)
    num_matched = [0] * len(thresholds)
    for threshold in tqdm(thresholds):
        candidate_samples = counts[counts == threshold].index.values
        if len(candidate_samples) == 0:
            continue
        candidate_counts = np.vectorize(lambda x: x.split('-')[1])(candidate_samples)
        for count in np.unique(candidate_counts):
            _candidate_samples = candidate_samples[candidate_counts == count]
            _true_samples = true_samples[true_counts == count]
            _matched_samples = np.intersect1d(_true_samples, _candidate_samples)
            num_candidates[-1] += len(_candidate_samples)
            num_matched[-1] += len(_matched_samples)
    num_candidates, num_matched = np.array(num_candidates), np.array(num_matched)

    return thresholds, num_candidates, num_matched

def _calculate_matches(counts, df_true):
    df_true = df_true.drop_duplicates().reset_index(drop=True)
    true_samples = convert_data2string(df_true)
    thresholds = np.sort(counts.unique())[::-1]

    num_candidates, num_matched = [], []
    for threshold in tqdm(thresholds):
        candidate_samples = counts[counts == threshold].index.values
        matched_samples = np.intersect1d(true_samples, candidate_samples)
        num_candidates.append(len(candidate_samples))
        num_matched.append(len(matched_samples))
    num_candidates, num_matched = np.array(num_candidates), np.array(num_matched)

    return thresholds, num_candidates, num_matched

def calculate_matches(counts, df_true, check_freq=False):
    if check_freq:
        return _calculate_matches_by_freq(counts, df_true)
    return _calculate_matches(counts, df_true)