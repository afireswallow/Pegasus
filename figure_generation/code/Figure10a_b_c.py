import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os

# #ISCXVPN
data1 = [
    [75.74, 76.17, 75.20, 77.88, 98.72],  # Pegasus
    [77.14, 77.27, 76.49, 79.56, 98.98]   # GPU/CPU
]


#PeerRush
data2 = [
    [88.23, 90.90, 90.57, 92.07, 99.66],  # Pegasus
    [88.57, 92.28, 91.56, 93.76, 99.90]   # GPU/CPU
]


#CICIOT
data3 = [
    [85.81, 87.07, 86.59, 88.29, 93.80],  # Pegasus
    [87.16, 88.48, 87.57, 89.49, 94.71]   # GPU/CPU
]


datalist = [data1, data2, data3]

dataset = ["ISCXVPN", "PeerRush", "CICIOT"] 



def plot_grouped_bar_chart(dataall, labels, xtick_labels, width=0.4, colors=None, ylabel="Macro-F1"):
    data = datalist[dataall]  
    datasetname = dataset[dataall]
    x = np.arange(len(xtick_labels))  
    if colors is None:
        colors = ['#A6CE39', '#5E9CD3']
    plt.figure(figsize=(9, 6))
    data_decimal = [[x/100 for x in row] for row in data]
    hatches = ['//', '\\\\']
    for i in range(2):  
        for j in range(len(data_decimal[0])):
            x_pos = x[j] + (i - 0.5)*width + (i*0.015) 
            plt.bar(x_pos, data_decimal[i][j], width=width, 
                    label=labels[i] if j == 0 else "",
                    color='white', 
                    edgecolor=colors[i],  
                    hatch=hatches[i]*2,
                    linewidth=2,
                    fill=True,
                    zorder=3)
    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=24, rotation=30, ha='right', rotation_mode='anchor') 
    
    plt.ylabel(ylabel, fontsize=24)
    plt.ylim(0.7, 1.005)
    
    major_ticks = np.arange(0.7, 1.01, 0.1)
    minor_ticks = np.arange(0.75, 1.01, 0.1)  
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    
    plt.grid(axis='y', which='major', linestyle='--', alpha=0.7, zorder=1)  
    plt.grid(axis='y', which='minor', linestyle=':', alpha=0.5, zorder=1)   
    
    plt.yticks(fontsize=20)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=2, prop={'size': 24}, frameon=False)

    plt.tight_layout()
    folder_name = "figures"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if datasetname == "ISCXVPN":
        number = 'c'
    elif datasetname == "PeerRush":
        number = 'a'
    else:
        number = 'b'
    plt.savefig(f'figures/Figure10_{number}.pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.show()



labels = ["Pegasus", "GPU/CPU"]
xtick_labels = ["MLP-B", "RNN-B", "CNN-B", "CNN-M", "CNN-L"]

for i in range(3):
    plot_grouped_bar_chart(i, labels, xtick_labels, 
                        colors=['#4A8B60', '#3A7BAB'])





