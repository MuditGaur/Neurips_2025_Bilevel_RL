import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

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

def load_csv_runs(csv_files, step_key='step', reward_key='episode_reward', smoothing=5, padding=10, max_range=None):
    xs, ys = [], []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        x = df[step_key].values
        y = df[reward_key].values
        if smoothing > 1:
            x, y = smooth_reward_curve(x, y, padding, a_range=max_range)
        xs.append(x)
        ys.append(y)
    return xs, ys

def plot_with_iqr(xs, ys, label, color):
    xs, index, _ = pad(xs)
    ys, _, _ = pad(ys)
    median = np.nanmedian(ys, axis=0)
    q25 = np.nanpercentile(ys, 25, axis=0)
    q75 = np.nanpercentile(ys, 75, axis=0)
    plt.plot(xs[index], median, color=color, label=label)
    plt.fill_between(xs[index], q25, q75, color=color, alpha=0.25)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvs', nargs='+', required=True, help='List of CSV groups, each group is comma-separated runs for an algorithm')
    parser.add_argument('--labels', nargs='+', required=True, help='Legend labels for each group')
    parser.add_argument('--colors', nargs='+', default=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], help='Colors for each group')
    parser.add_argument('--smoothing', type=int, default=5, help='Smoothing window')
    parser.add_argument('--padding', type=int, default=10, help='Padding for smoothing')
    parser.add_argument('--max_range', type=int, default=None, help='Max range for smoothing')
    parser.add_argument('--step_key', type=str, default='step', help='Column name for steps')
    parser.add_argument('--reward_key', type=str, default='episode_reward', help='Column name for rewards')
    parser.add_argument('--title', type=str, default='Learning Curve')
    parser.add_argument('--xlabel', type=str, default='Environment Steps')
    parser.add_argument('--ylabel', type=str, default='Episode Reward')
    parser.add_argument('--save_path', type=str, default='pebble_style_plot.png')
    args = parser.parse_args()

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.figsize': (10, 6),
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
    })

    plt.figure()
    for i, group in enumerate(args.csvs):
        csv_files = group.split(',')
        xs, ys = load_csv_runs(csv_files, step_key=args.step_key, reward_key=args.reward_key, 
                               smoothing=args.smoothing, padding=args.padding, max_range=args.max_range)
        plot_with_iqr(xs, ys, label=args.labels[i], color=args.colors[i % len(args.colors)])

    plt.xlabel(args.xlabel, fontsize=15)
    plt.ylabel(args.ylabel, fontsize=15)
    plt.title(args.title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.25)

    # Format x-ticks
    current_values = plt.gca().get_xticks()
    plt.gca().set_xticklabels([format_xticks(x) for x in current_values])

    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.savefig(args.save_path)
    plt.show()

if __name__ == "__main__":
    main()
