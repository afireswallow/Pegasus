import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

max_bits_per_flow = [28, 44, 72]
f1_score_peerrush = [0.991179,0.992867,0.995416]
f1_score_ciciot = [0.92931,0.9380,0.939618]
f1_score_vpn = [0.97153, 0.986369, 0.989009]
colors = ['#219EBC', '#2A9D8C', '#DB3124', '#4B74B2', '#b05da5', '#000000']

x_labels = ["17.0%\n (28)", "25.5%\n (44)", "38.3%\n (72)"]

# Plotting
plt.figure(figsize=(5, 4))
plt.plot(f1_score_peerrush, label="PeerRush", marker='o', color=colors[2], markersize=7)
plt.plot(f1_score_ciciot, label="CICIOT", marker='^', color=colors[1], markersize=9)
plt.plot(f1_score_vpn, label="ISCXVPN", marker='s', color=colors[4], markersize=7)

# Customizing the x-axis
plt.xticks(range(len(x_labels)), x_labels, fontsize=16, )
plt.xlabel("Num of bits per flow", fontsize=20, )

# Customizing the y-axis
plt.ylabel("F1 score", fontsize=20, )
plt.yticks(fontsize=16, )
plt.ylim(0.9, 1.0)

# Add vertical lines
for x in range(len(x_labels)):
    plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)

# Add grid and legend
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc="lower right", prop={'size': 12})

# Show the plot
# plt.show()
plt.margins(x=0.1)
plt.tight_layout()
plt.savefig('./figures/Figure8.pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)