import json
import copy
from torch.utils.data import Dataset
import torch




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
CICIOT2022_rule = {
    "Power-Audio": 0,
    "Power-Cameras": 0,
    "Power-Other": 0,
    "Idle": 1,
    "Interact-Audio": 2,
    "Interact-Cameras": 2,
    "Interact-Other":2
}

class FlowDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, filename, window_size, args):
        super().__init__()
        self.flows = []
        self.window_size = window_size

        with open(filename) as fp:
            instances = json.load(fp)
        for ins in instances:

            if args.dataset == "ISCXVPN":
                ins['label'] = int(ISCXVPN_rule[ins['label']])
                if ins['label'] == 0:
                    continue
                else:
                    label = ins['label'] - 1
            elif args.dataset == "PeerRush":
                ins['label'] = int(PeerRush_rule[ins['label']])
                label = ins['label']
            elif args.dataset == "CICIOT2022":
                ins['label'] = int(CICIOT2022_rule[ins['label']])
                label = ins['label']
            else:
                print("Dataset not supported!")
                raise NotImplementedError

            len_seq = ins['len_seq']
            real_len_seq = copy.deepcopy(len_seq)
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

            combined_seq = torch.tensor([len_seq, ipd_seq])
            
            flow_packets = len(len_seq)
            if flow_packets >= self.window_size:
                segs_idx = [idx for idx in range(0, flow_packets - self.window_size + 1)]
                for idx in segs_idx:
                    seg_len = torch.LongTensor(len_seq[idx: idx + self.window_size])
                    seg_ipd = torch.LongTensor(ipd_seq[idx: idx + self.window_size])
                    combined_seg = torch.stack([seg_len, seg_ipd], dim=-1)
                    #print(combined_seg.shape)
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











