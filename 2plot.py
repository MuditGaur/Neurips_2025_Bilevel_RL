import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

# Set publication-quality plot settings
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

def load_and_process_data(file_path, window=5):
    """Load data and apply smoothing if needed"""
    data = pd.read_csv(file_path)
    # Smoothing (optional)
    if window > 1:
        data['episode_reward'] = data['episode_reward'].rolling(window=window, min_periods=1).mean()
    return data

def format_axis(x, pos):
    """Format x-axis to show K for thousands"""
    if x >= 1000:
        return f'{x/1000:.0f}K'
    else:
        return f'{x:.0f}'

def plot_learning_curves(files, labels, colors, markers, linestyles, title="Learning Curves", 
                         save_path="learning_curve_plot.pdf", smoothing_window=5):
    """Generate publication-quality learning curve plots comparing multiple runs"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, file_path in enumerate(files):
        data = load_and_process_data(file_path, window=smoothing_window)
        
        # Plot main line
        ax.plot(data['step'], data['episode_reward'], 
                label=labels[i], 
                color=colors[i],
                marker=markers[i], 
                linestyle=linestyles[i],
                markevery=max(1, len(data)//20))  # Show ~20 markers per line
    
    # Customize plot appearance
    ax.set_xlabel('Environment Steps', fontweight='bold')
    ax.set_ylabel('Episode Rewards', fontweight='bold')
    ax.xaxis.set_major_formatter(FuncFormatter(format_axis))
    
    # Add grid, legend, and background color
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f8f8f8')
    
    # Add legend with a subtle frame
    legend = ax.legend(frameon=True, fancybox=True, framealpha=0.8, loc='best')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#e0e0e0')
    
    # Set scientific notation for y-axis if values are very large
    if ax.get_ylim()[1] > 1000:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    return fig, ax

# Example usage
if __name__ == "__main__":
    # File paths
    files = ['exp/walker_walk/H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/MRN_maxfeed100_Rbatch10_meta1000_seed45123/train.csv', 'exp/walker_walk/H1024_L2_lr0.0005/teacher_b-1_g1_m0_s0_e0/init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0/PEBBLE_maxfeed100_Rbatch10_seed34512/train.csv']
    
    # Plot settings
    labels = ['Algorithm A', 'Algorithm B']
    colors = ['#1f77b4', '#ff7f0e']  # Professional blue and orange
    markers = ['o', 's']
    linestyles = ['-', '-']
    
    # Create plot
    plot_learning_curves(
        files=files,
        labels=labels,
        colors=colors,
        markers=markers,
        linestyles=linestyles,
        title="Comparison of RL Algorithms",
        save_path="pebble_style_plot.png",
        smoothing_window=5
    )

