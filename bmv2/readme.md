# **MLP_B – BMv2 Version**
## 📌 Overview
This repository provides the **BMv2** implementation of **MLP_B**, enabling researchers to validate our approach without the need for hardware switches.  
Although **BMv2** is a flexible, software-based programmable switch simulator, there are notable differences from real hardware:

- **Priority Handling:** BMv2’s priority mechanism differs from that of hardware switches, which may cause discrepancies in priority-based rule matching.
- **Performance Limitations:** Under heavy load, BMv2 can suffer from high packet loss, affecting the completeness of test results.
- **Throughput Gap:** BMv2 cannot fully reflect the forwarding capacity of real hardware.
- **Computation Differences:** Complex calculations may behave differently, with potential bit-width or overflow issues.

Given these differences — and our limited development resources — **this release implements only the MLP-B model**.

- To address BMv2’s packet loss issue (which could distort accuracy), our implementation uses **generated packets** instead of raw packets, removing the need for feature extraction in experiments.  
- Tests show that software and hardware exhibit **consistent behavior**, confirming the validity of our method.

> For further support or inquiries about real-hardware integration, please contact **afireswallow@gmail.com**.
---

## ⚙️ Environment Setup
Our project is built on an Ubuntu 20.04 virtual machine, with the environment fully installed using the /p4-guide/bin/install-p4dev-v8.sh script.
```shell
$ sudo apt install git     # For Ubuntu
$ sudo dnf install git     # For Fedora
$ git clone https://github.com/jafingerhut/p4-guide
$ ./p4-guide/bin/install-p4dev-v8.sh |& tee log.txt

# If you used v8 version of the install script, see Note 1 below.
```

More necessary dependencies have been included in the requirements.txt
```shell
pip install -r requirement.txt
```

Or you may choose to download and use the [virtual machine image](https://drive.google.com/drive/folders/1lsdXo_Fx4KgGiibc1FZCnFRQjFEMTlAK?usp=drive_link
) we provide.

---
 
## 📂 Project Structure
```bash
├── make_dataset               # Dataset & model preparation for BMv2
│   ├── convert.py
│   ├── convert_pkl_mlp.py
│   └── mlp/
│       ├── CICIOT2022
│       ├── ISCXVPN
│       └── PeerRush
│
├── p4_CICIOT2022               # BMv2 P4 code for CICIOT2022
├── p4_ISCXVPN                  # BMv2 P4 code for ISCXVPN
├── p4_PeerRush                 # BMv2 P4 code for PeerRush
├── send_recieve/               # Traffic send & receive scripts
└── readme.md
```
Dataset Notes:
- PeerRush → same of 3 outputs.
- ISCXVPN → 6 outputs.

```shell
CICIOT2022 as an example:
.
├── basic.p4                  
├── headers.p4
├── parsers.p4
│ 
├── build
│   ├── mlp_CICIOT.json
│   └── mlp_CICIOT.p4info.txt
├── control
│   ├── mlp32_control.py
│   └── p4runtime_lib
│       
├── bmv2logs
├── start_switch.sh
└── start_veth.sh
```
---
## 🚀 Usage Guide

```shell
# convert dataset and model

## Full test，convert and test
python convert_pkl_mlp.py --model_path "model.pt" --test_data_path "test.csv" --dataset_type "CICIOT2022" --output_pkl_path "output.pkl" --dataset_pkl_path "dataset.pkl"

## Only test
python convert_pkl_mlp.py --model_path "model.pt" --test_data_path "test.csv" --dataset_type "CICIOT2022" --test_only

## specific device
python convert_pkl_mlp.py --model_path "model.pt" --test_data_path "test.csv" --dataset_type "CICIOT2022" --device "cpu" --test_only


# At first you need to execute ./start_veth.sh

./start_veth.sh


# Use Scripts to simplify the process

./p4_your_project/start_switch.sh

python3 p4_your_project/control/mlp32_control.py


# Or execute commands below:

p4c-bm2-ss --target bmv2 --arch v1model --p4runtime-files p4_CICIOT2022/build/mlp_CICIOT_.p4info.txt -o p4_CICIOT2022/build/mlpCICIOT.json p4_CICIOT2022/basic.p4

mkdir bmv2logs

mkdir build

./start_veth.sh

sudo simple_switch_grpc -i 0@veth0 -i 1@veth1 --log-console --no-p4  -- --grpc-server-addr 127.0.0.1:50051 build/mlp_CICIOT.json > bmv2logs/run_switch.log 2>&1

python3 control/mlp32_control.py

sudo $P4_HOME/p4dev-python-venv/bin/python send_recieve/listen_veth.py

sudo tcpreplay --pps=10 -i veth0 /path/to/your.pcap
```

