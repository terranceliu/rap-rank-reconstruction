import os
import pandas as pd

import pdb

fnames_baseline = {
    'results': ['results_baseline', []],
}

fnames_reconstruct = {
    'results': ['results_reconstruct', []],
}

fnames_reconstruct_init_split = {
    'results': ['results_reconstruct_init_split', []],
}

for root, dirs, files in os.walk("./results"):
    for file in files:
        if 'init_split' in root:
            for fn in fnames_reconstruct_init_split.keys():
                if file.endswith("{}.csv".format(fn)):
                    path = os.path.join(root, file)
                    fnames_reconstruct_init_split[fn][1].append(path)
        else:
            if 'baseline' in root:
                for fn in fnames_baseline.keys():
                    if file.endswith("{}.csv".format(fn)):
                        path = os.path.join(root, file)
                        fnames_baseline[fn][1].append(path)
            for fn in fnames_reconstruct.keys():
                if file.endswith("{}.csv".format(fn)):
                    path = os.path.join(root, file)
                    fnames_reconstruct[fn][1].append(path)

for path_dict in [fnames_baseline, fnames_reconstruct, fnames_reconstruct_init_split]:
    for save_fn, paths in path_dict.values():
        results = []
        for path in paths:
            df = pd.read_csv(path)
            results.append(df)
        results = pd.concat(results)
        results.to_csv('./results/{}.csv'.format(save_fn), index=False)