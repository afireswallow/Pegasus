## CNNL 目录说明

### 1. `cnnl_output3/`

存放 **三分类** 模型的 P4 程序代码和对应的控制平面代码，用于CICIOT和PeerRush两个数据集。

### 2. `cnnl_output6/`

存放 **六分类** 模型的 P4 程序代码和对应的控制平面代码，用于ISCXVPN数据集。

### 3. `cnnl_pt2pkt/`

存放三个数据集经软件训练得到的模型权重文件，以及转换结果：

* `.pt` 文件：PyTorch 训练得到的权重文件
* `.pkl` 文件：将 `.pt` 转换后用于控制平面加载的权重文件
* 转换脚本：`cnnl-pt2pkt.py`

#### `cnnl-pt2pkt.py` 使用说明

该脚本用于将 **PyTorch 权重文件（.pt）** 转换为 **P4 控制平面可加载的权重文件（.pkl）**。

运行前需要手动修改脚本内的两个位置：

1. **第 9 行**：
   将`CICIOT2022_finalLUT.pt`改为要转换的 `.pt` 文件路径。

2. **倒数第三行**：
   将`with open('cnnl-CICIOT.pkl','wb') as f:`改为希望生成的 `.pkl` 文件路径。
