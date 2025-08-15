# Pegasus

Repository for the paper [**Pegasus: A Universal Framework for Scalable Deep Learning Inference on the Dataplane**](https://doi.org/10.1145/3718958.3750529), published at SIGCOMM 2025.  

---

## 📂 Repository Overview

- **[`dataset`](dataset)** — Contains the datasets used in our experiments and the corresponding preprocessing scripts.  
- **[`software`](software)** — All code for software simulation, including model training, quantization, pretrained model parameters, and testing scripts.  
  Since Pegasus implements neural network models using a lookup-table approach, the software and hardware outputs are identical. Therefore, the software test results closely approximate the actual hardware execution results.  
- **[`hardware`](hardware)** — All code for hardware execution, including the P4 implementations for each model and the corresponding code generators.
- **[`bmv2`](bmv2)** — A lightweight BMv2 implementation of Pegasus (for the MLP-B model) to help users run the system in a software environment and understand the core design.

---

## 📬 Contact

Please open a GitHub issue or email **afireswallow@gmail.com** for questions or support.

---

## 📜 License

This project is licensed under the **Apache-2.0 License** — see the [LICENSE](LICENSE) file for details.