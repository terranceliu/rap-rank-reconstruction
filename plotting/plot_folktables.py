import os
import pickle
import argparse
import numpy as np

from plot_utils.utils import interpolate_results, get_results_avg
from plot_utils.utils_folktables import get_datasets, METHOD_PLOT_PARAMS, ALL_METHODS

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

def get_args():
    parser = argparse.ArgumentParser()

    # datasets
    parser.add_argument('--year', type=int, default=2018)
    parser.add_argument('--tasks', nargs='+', default=['employment', 'coverage', 'mobility'])
    parser.add_argument('--states', nargs='+', default=['CA', 'NY', 'TX', 'FL', 'PA'])
    # plot params
    parser.add_argument('--min_rate', type=float, default=None)

    args = parser.parse_args()

    return args

def get_results_by_task(tasks):
    results_by_task = {}

    for task in tasks:
        result_avg = {}
        counts_avg = np.zeros(100000)

        for dataset, result in results.items():
            if not dataset[11:].startswith(task):
                continue

            for method in result.keys():
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
            results_by_task[task] = result_avg

    return results_by_task

#####

args = get_args()

datasets = get_datasets(args.year, args.tasks, args.states)

all_methods = ALL_METHODS.copy()

results_path = './results/folktables.pkl'
with open(results_path, 'rb') as handle:
    results = pickle.load(handle)

results = interpolate_results(results)

# Plot

plot_dir = 'images/folktables'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

plot_datasets = datasets.copy()
plot_methods = all_methods.copy()
plot_titles = [dataset.replace('folktables_', '').replace('2018_', '').replace('_', '-') for dataset in datasets]
method_plot_params = METHOD_PLOT_PARAMS.copy()



##### Plot average over all experiments #####

results_average = get_results_avg(results)

figsize = plt.figure(figsize=(10, 8))

for method in plot_methods:
    y = results_average[method]
    x = np.arange(len(y)) + 1
    x = x / x.max()
    
    if args.min_rate is not None:
        mask = (y >= args.min_rate)
        y = y[mask]
        x = x[mask]

    color = method_plot_params[method]['c']
    linestyle = 'solid' if method.startswith('RAP') else 'dashed'
    plt.plot(x, y, label=method, marker=marker, color=color, linestyle=linestyle)

plt.xlabel('K\n(as a proportion of the # of unique rows in D)', labelpad=20)
plt.ylabel('MATCH-RATE$_{K}$', labelpad=20)

ax = plt.subplot(111)
ax.legend(labels=plot_methods, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.45))

plt.savefig(os.path.join(plot_dir, 'avg.png'), bbox_inches='tight', dpi=dpi)



##### Plot average by task #####

results_by_task = get_results_by_task(args.tasks)

num_rows, num_cols = 1, len(args.tasks)

figsize = (x_size * num_cols, y_size * num_rows)
fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

subplots = ax.reshape(-1)

for idx, task in enumerate(args.tasks):
    result = results_by_task[task]

    subplot = subplots[idx]

    for method in plot_methods:
        y = result[method]
        x = np.arange(len(y))
        x = x / x.max()

        if args.min_rate is not None:
            mask = (y >= args.min_rate)
            y = y[mask]
            x = x[mask]

        color = method_plot_params[method]['c']
        linestyle = 'solid' if method.startswith('RAP') else 'dashed'
        subplot.plot(x, y, label=method, marker=marker, color=color, linestyle=linestyle)

    subplot.set_title(task)

    subplot.tick_params(axis='x', which='major', labelsize=18,
                        bottom=True, top=False, labelbottom=True, rotation=40)

plt.subplots_adjust(hspace=0.2 * scale, wspace=0.15 * scale)

if num_cols % 2 == 1:
    idx = len(subplots) - 1 - num_cols // 2
    subplots[idx].legend(labels=plot_methods, ncol=2,
                         fontsize=label_size / 2, loc='lower center', bbox_to_anchor=(0.5, -0.7))
else: # TODO
    pass

# create a big axis, hide tick and tick label of the big axis
ax = fig.add_subplot(111, frameon=False)
ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

ax.set_xlabel('k / u\n(as a proportion of the # of unique rows in D)', fontsize=label_size / 2, labelpad=40)
ax.set_ylabel('MATCH-RATE$_{K}$', fontsize=label_size / 2, labelpad=40)

plt.savefig(os.path.join(plot_dir, 'avg_by_task.png'), bbox_inches='tight', dpi=dpi)



##### Plot all #####

num_rows = len(args.tasks)
num_cols = len(args.states)

figsize = (x_size * num_cols, y_size * num_rows)
fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

subplots = ax.reshape(-1)

idx = 0
for i in range(num_cols * num_rows):
    dataset = plot_datasets[idx]
    result = results[dataset]

    subplot = subplots[idx]

    for method in plot_methods:
        y = result[method]
        x = np.arange(len(y))
        x = x / x.max()

        if args.min_rate is not None:
            mask = (y >= args.min_rate)
            y = y[mask]
            x = x[mask]

        color = method_plot_params[method]['c']
        linestyle = 'solid' if method.startswith('RAP') else 'dashed'
        subplot.plot(x, y, label=method, marker=marker, color=color, linestyle=linestyle)

    subplot.set_title(plot_titles[idx])

    subplot.tick_params(axis='x', which='major', labelsize=18,
                        bottom=True, top=False, labelbottom=True, rotation=40)

    idx += 1

plt.subplots_adjust(hspace=0.2 * scale, wspace=0.15 * scale)

if num_cols % 2 == 1:
    idx = len(subplots) - 1 - num_cols // 2
    subplots[idx].legend(labels=plot_methods, ncol=2, # len(plot_methods),
                         fontsize=label_size, loc='lower center', bbox_to_anchor=(0.5, -1.4))
else: # TODO
    pass

# create a big axis, hide tick and tick label of the big axis
ax = fig.add_subplot(111, frameon=False)
ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

ax.set_xlabel('k / u\n(proportion of the # of unique rows in D)', fontsize=label_size, labelpad=40)
ax.set_ylabel('MATCH-RATE$_{K}$', fontsize=label_size, labelpad=40)

plt.savefig(os.path.join(plot_dir, 'all.png'), bbox_inches='tight', dpi=dpi)