# 📘 running_version Directory Guide

This folder contains the **CNN-L codebase** that can be directly deployed in real environments for **traffic analysis**.

---

## 📂 Project Structure

- [**`cnnl_output3`**](./cnnl_output3/) - Contains **3-class classification** P4 programs and corresponding control-plane code, designed for the **CICIOT** and **PeerRush** datasets.

- [**`cnnl_output6`**](./cnnl_output6/) - Contains **6-class classification** P4 programs and corresponding control-plane code, designed for the **ISCXVPN** dataset.

- [**`cnnl_pt2pkt`**](./cnnl_pt2pkt/) - Stores the trained model weights and conversion results for all three datasets:  
  - `.pt`: PyTorch-trained weight files  
  - `.pkl`: Converted weight files, ready to be loaded by the P4 control plane  
  - Conversion script: `cnnl-pt2pkt.py`

- [**`screenshots`**](./screenshots/) - Contains runtime execution screenshots and results.

---

## ⚙️ Usage of `cnnl-pt2pkt.py`

This script converts **PyTorch weight files (`.pt`)** into **P4 control-plane compatible weight files (`.pkl`)**.

Before running, two modifications are required inside the script:

1. **Line 9**  
    Replace ```CICIOT2022_finalLUT.pt``` with the path of the ```.pt``` file you want to convert.

2. **Third line from the bottom**
    Replace ```with open('cnnl-CICIOT.pkl', 'wb') as f:``` with the desired output path for the ***.pkl*** file.
