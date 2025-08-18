# Pegasus — Software Artifact

This repository contains the software components of **Pegasus**, including training/testing code, processed datasets, and pre-trained model weights to reproduce the paper’s key results.

---

## 📂 Project Structure

- [**`dataset`**](./dataset/) — Processed datasets  
  - `ISCXVPN`, `PeerRush`, `CICIOT2022`: normal traffic  
  - `malicious_traffic`: for autoencoder testing
- [**`util`**](./util/) - Toolkits
- [**`model`**](./model/) — Implementations of CNN (B/M/L), MLP, RNN, Autoencoder  
- [**`save`**](./save/) — Pre-trained & quantized model weights  
- [**`quickdemo`**](./quickdemo/) — Reproduce results with pre-trained weights 
- [**`train_net`**](./train_net/) — Optional, for model retraining

---

## 🛠 Environment

Please clone the repo and install the required environment by runing the following commands.
```
conda create-n Pegasus python=3.8.18

conda activate Pegasus

git clone https://github.com/afireswallow/Pegasus.git

cd Pegasus

pip install -r requirements.txt

```

---

## 🚀 Usage

### 1. Quick Demo with Provided Weights

* Run ```python -m quickdemo.MODELtest --dataset /DATASET_NAME --ptpth /WEIGHTS_PTH --testpth /DATASET_PTH``` to obtain the simulated hardware deployment results of the specified models(MLP, CNN-B, CNN-M, CNN-L, RNN). For example, to verify the result of CNN-L of PeerRush, you can run ```python -m quickdemo.cnnLtest --dataset PeerRush --ptpth save/cnnL/PeerRush/CNNL_PeerRush.pt --testpth dataset/PeerRush/redeal_test.json --device 0```

* ⚠️ **Notes on Accuracy Differences**: The accuracy results reproduced with this artifact may show slight deviations from those reported in the paper, due to the following factors:
	1.	Missing initial model weights – The original .pth files used in the paper were lost. The current models were retrained from scratch, which may lead to minor accuracy differences.
	2.	Hardware–software mismatch – The hardware implementation is not fully identical to the software simulation. In practice, issues such as register conflicts or packet loss may occur, slightly affecting the results.

    Overall, these differences are small and do not affect the validity of the conclusions. 

# todo: 还需要更改

### 2. Model Training and Quantization (Optional)

#### 🔍 Fuzzy Matching
- CNN-B / CNN-M / RNN / MLP

```
python -m train_net.runmodel \
  --dataset DATASET_NAME \
  --model MODEL_NAME \
  --savepth /YOUR_SAVE_PTH \
  --trainpth /TRAIN_PTH \
  --testpth /TEST_PTH \
  --dlist YOUR_QUANTIZATION_SETTINGS \
  --device DEVICENAME \
  --modelnum MODELNUM
```

- TRAIN_PTH / TEST_PTH: use ISCXVPN, CICIOT2022, or PeerRush
    - MLP → CSV file
    - Others → JSON file
- Recommended --dlist:
    - MLP: 121
	- CNN-B: 22
	- CNN-M: 2
	- RNN: 21
	- Autoencoder: 2
- Optional: --modelnum / --ptpth to start from an intermediate layer


- CNN-L
```
python -m train_net.runcnnLmodel \
  --dataset DATASET_NAME \
  --savepth /YOUR_SAVE_PTH \
  --trainpth /TRAIN_PTH \
  --testpth /TEST_PTH \
  --dlist 22 \
  --device DEVICENAME \
  --modelnum MODELNUM
```
Requires intermediate outputs from the previous model:
```
python cnnl_getsegdata.py
python cannl_modifydata.py
```



- Autoencoder
```
python -m train_net.run_origin_autoencoder \
  --dataset DATASET_NAME \
  --savepth /YOUR_SAVE_PTH \
  --trainpth /TRAIN_PTH \
  --device DEVICENAME

python -m train_net.run_mm_autoencoder \
  --dataset DATASET_NAME \
  --savepth /YOUR_SAVE_PTH \
  --trainpth /TRAIN_PTH \
  --device DEVICENAME \
  --ptpth /TRAINED_WEIGHT_PTH \
  --LSTPTH /LOOKUP_INIT_VALUES
```


⚙️ Fixed-point Quantization + Primitive Fusion
- CNN-B / CNN-M / RNN / MLP: repalce MODEL_convert with specific model name,such as cnnb_convert.
```
python -m convert.MODEL_convert \
  --dataset DATASET_NAME \
  --savepth /YOUR_SAVE_PTH \
  --testpth /TEST_PTH \
  --ptpth /TRAINED_WEIGHT_PTH \
```

- Autoencoder:
```
python -m convert.autoencoder_convert \
  --dataset DATASET_NAME \
  --ptpth /TRAINED_WEIGHT_PTH \
```

- CNN-L:
```
python -m convert.MODEL_convert \
  --dataset DATASET_NAME \
  --savepth /YOUR_SAVE_PTH \
  --testpth /TEST_PTH \
  --ptpth1 /TRAINED_WEIGHT_OF_MODELNUM2_PTH \
  --ptpth1 /TRAINED_WEIGHT_OF_MODELNUM4_PTH \
```



To minimize precision loss, Quantization-Aware Training (QAT) can be employed. Pre-trained & converted weights are available in /save — see Section 5 for quick demo instructions.
