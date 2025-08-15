import json
import copy
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

CICIOT2022_rule = {
    "Power-Audio": 0,
    "Power-Cameras": 0,
    "Power-Other": 0,
    "Idle": 1,
    "Interact-Audio": 2,
    "Interact-Cameras": 2,
    "Interact-Other":2
}
ISCXVPN_rule = {
    "browsing": 0,
    "chat": 1,
    "ftp": 2,
    "mail": 3,
    "p2p": 4,
    "streaming": 5,
    "voip": 6
}
PeerRush_rule = {
    "emule": 0,
    "utorrent": 1,
    "vuze": 2
}


class bigcnnFlowDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, filename, window_size, args):
        super().__init__()
        self.flows = []
        self.window_size = window_size

        with open(filename) as fp:
            instances = json.load(fp)
        for ins in instances:
            if args.dataset == "CICIOT2022":
                ins['label'] = CICIOT2022_rule[ins['label']]
            elif args.dataset == "PeerRush":
                ins['label'] = PeerRush_rule[ins['label']]
            elif args.dataset == "ISCXVPN":
                ins['label'] = ISCXVPN_rule[ins['label']]
                
            if args.dataset == "ISCXVPN" and ins['label'] == 0:
                continue
            # 如果是 ISCXVPN 数据集，调整标签值
            label = ins['label'] - 1 if args.dataset == "ISCXVPN" else ins['label']

            len_seq = ins['len_seq']
            tos_seq = ins['ip_tos_seq']
            ip_hl_seq = ins['ip_hl_seq']
            #ip_ttl_seq =  ins['ip_ttl_seq']
            #ip_tos_seq =  ins['ip_tos_seq']
            tcp_off_seq = ins['tcp_off_seq']
            #tcp_win_seq =  ins['tcp_win_seq']
            real_len_seq = copy.deepcopy(len_seq)
            payload_sequences = ins['payload_sequences']
            #print(len(len_seq))
            # Truncate the packet length
            for i in range(len(len_seq)):
                len_seq[i] = min(len_seq[i], len_vocab - 1)
            
            ts_seq = ins['ts_seq']
            ipd_seq = [0]
            ipd_seq.extend([ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))])
            real_ipd_seq_us = [i * 1e6 for i in ipd_seq]
            # Truncate the ipd, unit: 16384 ns
            for i in range(len(ipd_seq)):
                ipd_seq[i] = min(round(ipd_seq[i] * 10000), ipd_vocab - 1)
                #ipd_seq[i] = min(ipd_seq[i] * 1e9 // 16384, ipd_vocab - 1)
                assert ipd_seq[i] >= 0
            
            x = 4096
            
            # Truncate the flow
            if len(len_seq) > x:
                len_seq = len_seq[:x]
                ipd_seq = ipd_seq[:x]
                real_len_seq = real_len_seq[:x]
                real_ipd_seq_us = real_ipd_seq_us[:x]
            """
            payload_sequences.append(len_seq)
            payload_sequences.append(ipd_seq)
            payload_sequences.append(tos_seq)
            payload_sequences.append(ip_hl_seq)
            payload_sequences.append(tcp_off_seq)
            """
            seg_extradata = torch.tensor(payload_sequences)
            flow_packets = len(len_seq)
            if flow_packets >= self.window_size:
                segs_idx = [idx for idx in range(0, flow_packets - self.window_size + 1)]
                for idx in segs_idx:
                    bit=3000
                    seg_len = torch.tensor(len_seq[idx: idx + self.window_size])
                    seg_ipd = torch.tensor(ipd_seq[idx: idx + self.window_size])
                    seg_len = torch.clamp(torch.tensor(len_seq[idx: idx + self.window_size]), max=bit)
                    seg_ipd = torch.clamp(torch.tensor(ipd_seq[idx: idx + self.window_size]), max=bit)
                    seg_tos = torch.tensor(tos_seq[idx: idx + self.window_size])
                    seg_iphl = torch.tensor(ip_hl_seq[idx: idx + self.window_size])
                    seg_tcpoff = torch.tensor(tcp_off_seq[idx: idx + self.window_size])
                    extradata = seg_extradata[:,idx:idx + self.window_size]
                    if torch.max(seg_len)>bit or torch.max(seg_ipd)>bit or torch.max(seg_tos)>bit or torch.max(seg_iphl)>bit or torch.max(seg_tcpoff)>bit:
                        print(torch.max(seg_len))
                        print(torch.max(seg_ipd))
                        print(torch.max(seg_tos))
                        print(torch.max(seg_iphl))
                        print(torch.max(seg_tcpoff))
                        print(torch.max(extradata))
                    packetfeatures = torch.stack([seg_len, seg_ipd, seg_tos, seg_iphl, seg_tcpoff], dim=1)
                    combined_seg = torch.cat((extradata.T, packetfeatures), dim=1) 
                    self.flows.append({
                        'x': combined_seg,
                        'label': label
                    })
            else:
                raise Exception('Flow packets < window size!!!')

    def __len__(self):
        return len(self.flows)
    
    def __getitem__(self, index):
        flow = self.flows[index]
        return flow['x'], flow['label']
    