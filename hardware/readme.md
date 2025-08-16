# Pegasus — Hardware Artifact

This repository contains the hardware components of **Pegasus**, including the P4 code and the scripts used to generate it.

---

## 📂 Project Structure

- [**`dataset`**](../dataset/) — PCAP files used in experiments
  - `ISCXVPN`, `PeerRush`, `CICIOT2022`: normal traffic  
  - `malicious_traffic`: for autoencoder testing
- [**`normal_version`**](./normal_version/) — Standard P4 code, for resource testing and accuracy evaluation
- [**`running_version`**](./running_version/) — Complete P4 code, ready to be deployed for traffic analysis
- [**`generate`**](./generate/) — Scripts used by the authors to generate P4 code
- [**`p4_syntax`**](./p4_syntax/) — One-click scripts for generating P4 code using P4 syntax templates

## ⚠️ Note
Due to the demanding hardware requirements, running this version can be complex.  
For ease of use, we provide a [BMv2 version](../bmv2/) for quick validation.  
Additionally, because this repository contains a large amount of code and scripts, if you encounter issues or need real-environment support, please contact the authors by emailing afireswallow@gmail.com.