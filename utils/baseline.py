import pandas as pd

def add_ppmf_block(counts, df_true, max_candidates=1000000):
    index_base = counts.index.values
    index_base = [c.split(' ')[1:] for c in index_base]

    counts_new = []
    block_sizes = df_true.groupby('TABBLK').size()
    for block in block_sizes.index:
        size = block_sizes[block]
        x = counts * size
        index = [' '.join([str(block)] + c) for c in index_base]
        x.index = index
        counts_new.append(x)
    counts_new = pd.concat(counts_new)
    counts_new = counts_new.sort_values(ascending=False)[:max_candidates]
    return counts_new