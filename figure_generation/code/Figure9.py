

import os
import json
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import argparse



def load_roc_data(data_dir):
    fpr_list = []
    tpr_list = []
    auc_list = []
    labels = []

    # 遍历目录中的所有JSON文件
    for filename in os.listdir(data_dir):
        if filename.endswith("_ROC_data.json"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r') as f:
                roc_data = json.load(f)
                fpr_list.append(roc_data['fpr'])
                tpr_list.append(roc_data['tpr'])
                auc_list.append(roc_data['auc'])
                labels.append(roc_data['label'])
    
    return fpr_list, tpr_list, auc_list, labels

def plot_auc_curves(fpr_list, tpr_list, auc_list, labels, title='ROC Curve Comparison', save_path=None):
    colors = ['#219EBC', '#2A9D8C', '#DB3124', '#4B74B2', '#b05da5', '#FFA500']
    # markers = ['o', '^', 's', 'v', 'D', 'p', '*']

    plt.figure(figsize=(5, 4))

    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    markers = ['o', 's', '^', 'v', 'D', '*']
    linewidths = [1.5, 1.8, 2.0, 2.2, 1.6, 1.9]
    markersizes = [7,7,8,8,7,10]

    for i in range(len(fpr_list)):
        label = f'{labels[i]:<6} (AUC = {auc_list[i]:>6.4f})'
        plt.plot(
            fpr_list[i], tpr_list[i],
            label=label,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markevery=0.2,              # 只在曲线上每隔一点加 marker，减少干扰
            markersize=markersizes[i % len(markersizes)],             # 放大 marker
            linewidth=linewidths[i % len(linewidths)],
            color=colors[i % len(colors)]
        )
    
    # 设置网格
    plt.grid(True, linestyle='-', alpha=0.5)
    # 设置图形属性
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    # plt.title(title, fontsize=14)
    plt.xticks(fontsize=16, fontfamily='serif')
    plt.yticks(fontsize=16, fontfamily='serif')
    # 设置图例，确保AUC分数对齐
    plt.legend(loc="lower right", prop={'size': 14})  # 使用等宽字体确保对齐

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')  
    
    # plt.show()

if __name__ == '__main__':
    # 定义命令行参数
    # parser = argparse.ArgumentParser(description="Choose dataset ")
    # parser.add_argument("--dataset", type=str, required=True, choices=["CICIOT2022", "PeerRush", "ISCXVPN"])
    # args = parser.parse_args()

    # 根据输入的数据集名称选择对应的数据目录
    # data_dirs = ['./dataset/roc/CICIOT2022_ROC','./dataset/roc/PeerRush_ROC','./dataset/roc/ISCXVPN_ROC']
    data_dirs = {
        "CICIOT2022": './dataset/roc/CICIOT2022_ROC',
        "PeerRush": './dataset/roc/PeerRush_ROC',
        "ISCXVPN": './dataset/roc/ISCXVPN_ROC'
    }
    
    for dataset in data_dirs:
        data_dir = data_dirs[dataset]
        # 加载ROC数据
        fpr_list, tpr_list, auc_list, labels = load_roc_data(data_dir)
        
        # 绘制ROC曲线
        save_path = f'figures/Figure9_{dataset}.pdf'
        plot_auc_curves(fpr_list, tpr_list, auc_list, labels, title='ROC Curve Comparison', save_path=save_path)
        