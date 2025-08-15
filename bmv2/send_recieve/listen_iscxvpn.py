#!/usr/bin/env python3
import os
import sys

from resultHeader_vpn import ResultHeader, FeatureHeader , ReservedResultHeader, Label
from scapy.all import TCP, sniff, Padding

# Configure veth interfaces
veth1 = "veth1"
veth1_peer = "veth1-peer"

# Statistics variables
n = 0
t = 0
noname = 0
total = 0

def UnsignedToSigned(x):
    t = (x >> 31) & 1
    x = (x & 0x7fffffff) - (t << 31)
    return x

result = []

def summary(pkt):
    label0 = UnsignedToSigned(pkt[ResultHeader].l32_1)
    label1 = UnsignedToSigned(pkt[ResultHeader].l32_2)
    label2 = UnsignedToSigned(pkt[ResultHeader].l32_3)
    label3 = UnsignedToSigned(pkt[ResultHeader].l32_4)
    label4 = UnsignedToSigned(pkt[ResultHeader].l32_5)
    label5 = UnsignedToSigned(pkt[ResultHeader].l32_6)
    values = [label0, label1, label2, label3, label4, label5]
    label = values.index(max(values))  
    
    # Abstruct last byte ofPadding 
    padding_last_byte = None
    packet_last_byte = None
    
    if Padding in pkt:
        padding_data = pkt[Padding].load
        if len(padding_data) > 0:
            padding_last_byte = padding_data[-1]
    
    # get last byte
    raw_bytes = bytes(pkt)
    if len(raw_bytes) > 0:
        packet_last_byte = raw_bytes[-1]
    
    pkt[Label].label = packet_last_byte


    return [
            label==pkt[Label].label,
            label,
            pkt[Label].label,
            pkt[FeatureHeader].ip_len,
            pkt[FeatureHeader].ip_total_len,
            pkt[FeatureHeader].protocol,
            pkt[FeatureHeader].tos,
            pkt[FeatureHeader].offset,
            pkt[FeatureHeader].max_byte,
            pkt[FeatureHeader].min_byte,
            pkt[FeatureHeader].max_ipd,
            pkt[FeatureHeader].min_ipd,
            pkt[FeatureHeader].ipd,
            label0,
            label1,
            label2,
            label3,
            label4,
            label5,
            ]

def handle_pkt(pkt):
    global t
    global n
    global total
    global noname
    total += 1
    if Label in pkt:
        # print(f"Received custom packet - Interface: {pkt.sniffed_on if hasattr(pkt, 'sniffed_on') else 'unknown'}")
        # pkt.show2()
        print(pkt[ResultHeader].show2())
        print("{}; {}; {}.".format(UnsignedToSigned(pkt[ResultHeader].l32_1),UnsignedToSigned(pkt[ResultHeader].l32_2),UnsignedToSigned(pkt[ResultHeader].l32_3)))
        sys.stdout.flush()
        #pkts.append(pkt)
        #result.append(summary(pkt))
        s = summary(pkt)
        print(s)
        
        
        if s[0] == True:
            t += 1
        n += 1
        # if n % 1000 == 0 :
        #     print(f"Correct classification: {t}, Unknown packets: {noname}, Processed: {n}, Total: {total}, Accuracy: {t / n * 100:.2f}%")
        #     print(f"  -> Target packet ratio: {n/total*100:.2f}%, Non-target packet ratio: {noname/total*100:.2f}%")
        print(f"Correct classification: {t}, Unknown packets: {noname}, Processed: {n}, Total: {total}, Accuracy: {t / n * 100:.2f}%")
        # print(f"  -> Target packet ratio: {n/total*100:.2f}%, Non-target packet ratio: {noname/total*100:.2f}%")
    else :
        pkt.show2()
        noname += 1
        if noname % 1000 == 0:
            print(f"Received {noname} non-target packets (Total received: {total})")
        # Display packet type information every 1000 packets
        if noname <= 10:
            print(f"Non-target packet #{noname}: {pkt.summary()}")
            if hasattr(pkt, 'show'):
                pkt.show2()
    

    

def main(iface, count = -1):
    print(f"Starting to listen on network interface: {iface}")
    print("Press Ctrl+C to stop listening")
    try:
        if count == -1:
            sniff(iface = iface, prn = lambda x: handle_pkt(x))
        else :
            sniff(iface = iface, prn = lambda x: handle_pkt(x), count = count)
        print("\nFinal statistics")
        print(f"Correct classification: {t}, Unknown packets: {noname}, Processed: {n}, Total: {total}, Accuracy: {t / n * 100:.2f}%")
        print(f"  -> Target packet ratio: {n/total*100:.2f}%, Non-target packet ratio: {noname/total*100:.2f}%")
    except KeyboardInterrupt:
        print(f"\nListening stopped")
        print(f"Final statistics: Correct classification={t}, Processed={n}, Total={total}")
        if n > 0:
            print(f"Final accuracy: {t / n * 100:.2f}%")



if __name__ == '__main__':
    print(f"1. Listening {veth1}")
    print("2. Your Port")
    
    choice = input("Please choose the listening Port (1/2): ").strip()
    
    if choice == "1":
        main(veth1)
    elif choice == "2":
        custom_iface = input("Please Input your Port's name: ").strip()
        main(custom_iface)
    else:
        print("Now Listening veth1")
        main(veth1)
