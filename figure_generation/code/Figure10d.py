import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator, FuncFormatter
import os

def scientific_formatter(x, pos):
    exponent = int(np.log10(x))
    return f'$10^{{{exponent}}}$'

def plot_grouped_bar_chart(data, labels, xtick_labels, width=0.3, colors=None, hatch_patterns=None, ylabel="Throughput (samples/s)"):
    x = np.arange(len(xtick_labels))

    if colors is None:
        colors = ['#219EBC', '#2A9D8C', '#4B74B2']
    
    if hatch_patterns is None:
        hatch_patterns = ['//', '\\\\', '--']

    plt.figure(figsize=(9, 6))

    for i in range(3):
        for j in range(len(data[0])):
            plt.bar(x[j] + (i - 1) * width, data[i][j], width=width, 
                    label=labels[i] if j == 0 else "",
                    color='white', edgecolor=colors[i], 
                    hatch=hatch_patterns[i]*2, linewidth=1.5, fill=False)

    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=24, rotation=30, ha='right', rotation_mode='anchor') 
    
    plt.ylabel(ylabel, fontsize=24)
    plt.yscale('log')
    ax = plt.gca()
    plt.yscale('log')
    plt.yticks([1e6, 1e8], [r'$10^6$', r'$10^8$'])  
    plt.yticks(fontsize=24)
    plt.grid(axis='y', linestyle='--', alpha=0.7, which='major')
    plt.grid(axis='y', which='minor', alpha=0)  

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=3, fontsize=24, frameon=False)

    plt.tight_layout()
    folder_name = "figures"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig('figures/Figure10_d.pdf', dpi=300, bbox_inches='tight')
    plt.show()

data = [
    [10**9, 10**9, 10**9, 10**9, 10**9],  # Pegasus
    [160*10**4, 122*10**4, 547*10**4, 520*10**4, 99*10**4],   # GPU
    [55*10**4, 3*10**4, 30*10**4, 8*10**4, 4*10**4]  # CPU
]

labels = ["Pegasus", "GPU", "CPU"]
xtick_labels = ["MLP-B", "RNN-B", "CNN-B", "CNN-M", "CNN-L"]


plot_grouped_bar_chart(data, labels, xtick_labels, 
                      colors=['#2c8d5c', '#147daf', '#794d9b'], 
                       hatch_patterns=['//', '\\\\', '-'])  