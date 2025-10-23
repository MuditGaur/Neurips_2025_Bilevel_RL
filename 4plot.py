import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def smooth_reward_curve(x, y, padding, a_range=None):
    if a_range is not None:
        len_m = a_range
    else:
        len_m = len(x)
    halfwidth = int(np.ceil(len_m / 100))
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(padding * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(padding * k + 1), mode='same')
    return xsmoo, ysmoo

def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])
    padded_xs = []
    index = -1
    max_index = 0
    for idx, x in enumerate(xs):
        if len(x) >= maxlen:
            padded_xs.append(x)
            index = idx
            max_index = idx
        else:
            padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
            x_padded = np.concatenate([x, padding], axis=0)
            padded_xs.append(x_padded)
    return np.array(padded_xs), index, max_index

def format_xticks(x):
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}k'
    else:
        return str(int(x))

def load_csv_runs(csv_files, step_key='step', reward_key='episode_reward', smoothing=5, padding=10, max_range=None, range_limit=None):
    xs, ys = [], []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        x = df[step_key].values
        y = df[reward_key].values
        # Apply range limit if specified
        if range_limit is not None:
            mask = x <= range_limit
            x = x[mask]
            y = y[mask]
        if smoothing > 1:
            x, y = smooth_reward_curve(x, y, padding, a_range=max_range)
        xs.append(x)
        ys.append(y)
    return xs, ys

def plot_with_mean_std(xs, ys, label, color):
    xs, index, _ = pad(xs)
    ys, _, _ = pad(ys)
    mean = np.nanmean(ys, axis=0)
    std = np.nanstd(ys, axis=0)
    plt.plot(xs[index], mean, color=color, label=label)
    plt.fill_between(xs[index], mean - std, mean + std, color=color, alpha=0.25)

def main():
    # # pebble_seeds = ['main_exp/walker/surf/seed_1/train.csv', 'main_exp/walker/surf/seed_2/train.csv', 'main_exp/walker/surf/seed_3/train.csv']
    # pebble_seeds = ['main_exp/walker/pebble/seed_1/train.csv','main_exp/walker/pebble/seed_2/train.csv','main_exp/walker/pebble/seed_3/train.csv']
    # # ours_seeds = ['main_exp/walker/surf/seed_1/train.csv','main_exp/walker/pebble/seed_1/train.csv']
    # ours_seeds = ['main_exp/walker/value/seed_1/train.csv','main_exp/walker/value/seed_2/train.csv']

    # pebble_seeds = ['main_exp/walker/surf/seed_1/train.csv', 'main_exp/walker/surf/seed_2/train.csv', 'main_exp/walker/surf/seed_3/train.csv']
    pebble_seeds = ['main_exp/dooropen/pebble/seed_1/train.csv','main_exp/dooropen/pebble/seed_2/train.csv']#,'main_exp/dooropen/pebble/seed_3/train.csv']
    # ours_seeds = ['main_exp/walker/surf/seed_1/train.csv','main_exp/walker/pebble/seed_1/train.csv']
    ours_seeds = ['main_exp/dooropen/value/seed_2/train.csv','main_exp/dooropen/value/seed_3/train.csv','main_exp/dooropen/pebble_1/seed_2/train.csv']

    csv_groups = [pebble_seeds, ours_seeds]
    labels = ['PEBBLE', 'OURS', 'SURF']
    colors = ['#1f77b4', '#ff0000', '#2ca02c']  # Changed OURS color to red

    # Updated font sizes
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (10, 6),
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
    })

    plt.figure()
    range_limit = 260000
    for i, csv_files in enumerate(csv_groups):
        xs, ys = load_csv_runs(csv_files, step_key='step', reward_key='episode_reward', 
                               smoothing=5, padding=10, max_range=None, range_limit=range_limit)
        plot_with_mean_std(xs, ys, label=labels[i], color=colors[i % len(colors)])

    plt.xlabel('Environment Steps', fontsize=30)
    plt.ylabel('Episode Reward', fontsize=30)
    plt.title('Door Open', fontsize=30)  # Updated title
    plt.grid(True, linestyle='--', alpha=0.25)

    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)

    current_values = plt.gca().get_xticks()
    plt.gca().set_xticklabels([format_xticks(x) for x in current_values])

    plt.legend(fontsize=26)
    plt.tight_layout()
    plt.savefig('comparison_dooropen.pdf')
    plt.show()

if __name__ == "__main__":
    main()
