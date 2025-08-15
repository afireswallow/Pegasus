#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate PCAP file from dataset with progress bar
-------------------------------------------------
1. Load features / labels (pickle)
2. Assemble FeatureHeader and Label for each record
3. Use tqdm to show processing progress
4. Write all packets at once using wrpcap
"""

import pickle
from scapy.all import Ether, IP, TCP, wrpcap
from tqdm import tqdm
# from featureHeader import FeatureHeader, Label   # Your custom protocol headers
from resultHeader import FeatureHeader, Label, ResultHeader, ReservedResultHeader

# ---------- Path Configuration ----------
PKL_PATH   = "/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/CICIOT2022/CICIOT2022_dataset.pkl"
PCAP_PATH  = "ciciot_dataset.pcap"
# ------------------------------

def main() -> None:
    # Load dataset
    with open(PKL_PATH, "rb") as f:
        features, labels = pickle.load(f)

    # Unified Ethernet/IP/TCP base header, adjustable as needed
    base = Ether(src="ff:ff:ff:ff:ff:ff",
                 dst="00:00:00:00:00:00") \
         / IP(len=96) / TCP()

    pkts = []

    # Add progress bar with tqdm
    for i in tqdm(range(len(features)), desc="Building packets", unit="pkt"):
        # print(labels[i])
        pkt = (base
               / FeatureHeader(
                     ip_len      = features[i][0],
                     ip_total_len= features[i][1],
                     protocol    = features[i][2],
                     tos         = features[i][3],
                     offset      = features[i][4],
                     max_byte    = features[i][5],
                     min_byte    = features[i][6],
                     max_ipd     = features[i][7],
                     min_ipd     = features[i][8],
                     ipd         = features[i][9])
               / Label(label = labels[i]))

        pkts.append(pkt)

    # Write all packets at once
    wrpcap(PCAP_PATH, pkts)
    print(f"\n✅ Completed writing: {PCAP_PATH} ({len(pkts)} packets total)")

if __name__ == "__main__":
    main()
