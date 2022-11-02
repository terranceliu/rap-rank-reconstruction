import os
import pickle
import argparse
import numpy as np

from plot_utils.utils import interpolate_results, get_results_avg
from plot_utils.utils_ppmf import get_datasets, get_label, get_fips_mapping, METHOD_PLOT_PARAMS, ALL_METHODS

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'mathtext.default':  'regular' })

dpi = 100
marker = None
plot_alpha = 0.2
line_thickness = 2

scale = 1.5
sns.set(style='white', font_scale=1.25 * scale)
mpl.rcParams['lines.linewidth'] = line_thickness * scale

x_size = 8
y_size = 6
label_size = 48

FIPS_MAPPING = get_fips_mapping()

def get_args():
    parser = argparse.ArgumentParser()

    # datasets
    parser.add_argument('--experiment_type', type=str, default='tracts')
    # plot params
    parser.add_argument('--min_rate', type=float, default=None)

    args = parser.parse_args()

    assert args.experiment_type in ['tracts', 'tracts_ib', 'blocks']

    return args

def get_results_by_state():
    results_by_state = {}

    for fips_code in FIPS_MAPPING.keys():
        result_avg = {}
        counts_avg = np.zeros(10000)

        for dataset, result in results.items():
            if dataset[5:7] != fips_code:
                continue

            for method in all_methods:
                if method not in result_avg.keys():
                    result_avg[method] = []

                a = result[method].copy()
                b = result_avg[method].copy()
                if len(a) < len(b):
                    result_avg[method] = b.copy()
                    result_avg[method][:len(a)] += a
                else:
                    result_avg[method] = a.copy()
                    result_avg[method][:len(b)] += b
            counts_avg[:len(a)] += 1

        for key in result_avg.keys():
            result_avg[key] /= counts_avg[:len(result_avg[key])]

        if len(result_avg) > 0:
            results_by_state[fips_code] = result_avg

    return results_by_state

#####

args = get_args()

datasets = get_datasets(args.experiment_type)

all_methods = ALL_METHODS.copy()
if args.experiment_type != 'blocks':
    all_methods = [x for x in all_methods if x != ALL_METHODS[3]]
if args.experiment_type == 'tracts_ib':
    all_methods = [x for x in all_methods if x != ALL_METHODS[1]]

results_path = './results/ppmf_{}.pkl'.format(args.experiment_type)
with open(results_path, 'rb') as handle:
    results = pickle.load(handle)

results = interpolate_results(results)

plot_dir = 'images/ppmf/{}'.format(args.experiment_type)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

plot_datasets = datasets.copy()
plot_methods = all_methods.copy()
method_plot_params = METHOD_PLOT_PARAMS.copy()

##### Plot average over all experiments #####

results_average = get_results_avg(results)

figsize = plt.figure(figsize=(10, 8))

for method in plot_methods:
    label = get_label(method, args.experiment_type)
    color = method_plot_params[label]['c']
    linestyle = 'solid' if method.startswith('RAP') else 'dashed'

    y = results_average[method]
    x = np.arange(len(y)) + 1
    x = x / x.max()
    
    if args.min_rate is not None:
        mask = (y >= args.min_rate)
        y = y[mask]
        x = x[mask]
    plt.plot(x, y, label=label, marker=marker, color=color, linestyle=linestyle)

plt.xlabel('k / u\n(proportion of the # of unique rows in D)', labelpad=20)
plt.xlabel('k / u', labelpad=20)
plt.ylabel('MATCH-RATE$_{D, k}$', labelpad=20)

ax = plt.subplot(111)
legend = [get_label(method, args.experiment_type) for method in plot_methods]
ax.legend(labels=legend, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.4))

plt.savefig('images/ppmf/{}/avg.png'.format(args.experiment_type), bbox_inches='tight', dpi=dpi)



##### Plot average by state #####

for plot_idx in range(2):
    results_by_state = get_results_by_state()

    plot_datasets = np.array(list(results_by_state.keys()))
    plot_titles = np.array([FIPS_MAPPING[dataset] for dataset in plot_datasets])

    num_cols = 5
    num_rows = 5

    figsize = (x_size * num_cols, y_size * num_rows)
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

    subplots = ax.reshape(-1)

    idx = 0
    for i in range(num_cols * num_rows):
        dataset = plot_datasets[idx + (num_cols * num_rows) * plot_idx]
        result = results_by_state[dataset]

        subplot = subplots[idx]

        for method in plot_methods:
            label = get_label(method, args.experiment_type)
            color = method_plot_params[label]['c']
            linestyle = 'solid' if method.startswith('RAP') else 'dashed'

            y = result[method]
            x = np.arange(len(y))
            x = x / x.max()

            if args.min_rate is not None:
                mask = (y >= args.min_rate)
                y = y[mask]
                x = x[mask]

            subplot.plot(x, y, label=method, marker=marker, color=color, linestyle=linestyle)

        title = plot_titles[idx + (num_cols * num_rows) * plot_idx]
        subplot.set_title(title)

        subplot.tick_params(axis='x', which='major', labelsize=18,
                            bottom=True, top=False, labelbottom=True, rotation=40)

        idx += 1

    plt.subplots_adjust(hspace=0.2 * scale, wspace=0.15 * scale)

    legend = [get_label(method, args.experiment_type) for method in plot_methods]
    if num_cols % 2 == 1:
        idx = len(subplots) - 1 - num_cols // 2
        legend_ncol = len(legend)
        if legend_ncol > 3:
            legend_ncol = np.ceil(len(legend) / 2).astype(int)
        subplots[idx].legend(labels=legend, ncol=legend_ncol,
                             fontsize=label_size, loc='lower center', bbox_to_anchor=(0.5, -1.45))
    else: # TODO
        pass

    # create a big axis, hide tick and tick label of the big axis
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    ax.set_xlabel('k / u\n(proportion of the # of unique rows in D)', fontsize=label_size, labelpad=40)
    ax.set_ylabel('MATCH-RATE$_{D, k}$', fontsize=label_size, labelpad=20)

    plt.savefig(os.path.join(plot_dir, 'all{}.png'.format(plot_idx + 1)), bbox_inches='tight', dpi=dpi)
