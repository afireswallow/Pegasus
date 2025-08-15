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

class FlowDataset(Dataset):
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
            label = ins['label'] - 1 if args.dataset == "ISCXVPN" else ins['label']

            len_seq = ins['len_seq']
            real_len_seq = copy.deepcopy(len_seq)
            for i in range(len(len_seq)):
                len_seq[i] = min(len_seq[i], len_vocab - 1)
            
            ts_seq = ins['ts_seq']
            ipd_seq = [0]
            ipd_seq.extend([ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))])
            real_ipd_seq_us = [i * 1e6 for i in ipd_seq]
            for i in range(len(ipd_seq)):
                ipd_seq[i] = min(round(ipd_seq[i] * 10000), ipd_vocab - 1)
                assert ipd_seq[i] >= 0
            
            x = 4096
            
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



class encoderFlowDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, filename, window_size, args, isTrained=True):
        super().__init__()
        self.flows = []
        self.window_size = window_size
        self.isTrained = isTrained

        with open(filename) as fp:
            instances = json.load(fp)

        for ins in instances:
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

                    # 转换多分类为二分类
                    original_label = ins['label']
                    if original_label == args.bad:
                        binary_label = 1  # 恶意流量
                    else:
                        binary_label = 0  # 善意流量
                    
                    # 根据 isTrained 参数决定是否保留恶意流量
                    if self.isTrained and binary_label == 1:
                        # 如果是训练集，跳过恶意流量
                        continue
                    
                    self.flows.append({
                        'x': combined_seg,
                        'label': binary_label
                    })
            else:
                raise Exception('Flow packets < window size!!!')

    def __len__(self):
        return len(self.flows)
    
    def __getitem__(self, index):
        flow = self.flows[index]
        return flow['x'], flow['label']





class newencoderDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, window_size, filenames):
        """
        :param filenames: 数据集文件名 {'CICIOT': 'ciciot.json', 'ISCXVPN': 'iscxvpn.json', 'PeerRush': 'peerrush.json'}
        """
        super().__init__()
        self.flows = []
        self.window_size = window_size

        # 加载所有数据集并统一
        for dataset_name, filename in filenames.items():
            with open(filename) as fp:
                instances = json.load(fp)
            for ins in instances:
                if dataset_name == "FLOOD" or dataset_name == "DDOS" or dataset_name == "FUZZ" or dataset_name == "test":
                    ins['label'] = 1
                else:
                    ins['label'] = 0
                len_seq = ins['len_seq']
                # Truncate packet length
                len_seq = [min(i, len_vocab - 1) for i in len_seq]
                ts_seq = ins['ts_seq']
                ipd_seq = [0] + [ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))]
                # Truncate ipd
                if ins['label'] == 1:
                    ipd_seq = [min(round(i * 1e3), ipd_vocab - 1) for i in ipd_seq]
                else:
                    ipd_seq = [min(round(i * 1e4), ipd_vocab - 1) for i in ipd_seq]
                # Truncate the flow
                if len(len_seq) > 4096:
                    len_seq = len_seq[:4096]
                    ipd_seq = ipd_seq[:4096]
                    # Combine sequences
                combined_seq = torch.tensor([len_seq, ipd_seq])

                # Slice into segments
                flow_packets = len(len_seq)
                if flow_packets >= self.window_size:
                    segs_idx = [idx for idx in range(0, flow_packets - self.window_size + 1)]
                    for idx in segs_idx:
                        seg_len = torch.LongTensor(len_seq[idx: idx + self.window_size])
                        seg_ipd = torch.LongTensor(ipd_seq[idx: idx + self.window_size])
                        combined_seg = torch.stack([seg_len, seg_ipd], dim=-1)
                        self.flows.append({
                            'x': combined_seg,
                            'label': ins['label']
                        })
                else:
                    raise Exception('Flow packets < window size!!!')
    def __len__(self):
        return len(self.flows)

    def __getitem__(self, index):
        flow = self.flows[index]
        return flow['x'], flow['label']




















class SeqFlowDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, filename, window_size):
        super().__init__()
        self.flows = []

        with open(filename) as fp:
            instances = json.load(fp)
        for ins in instances:
            len_seq = ins['len_seq']
            real_len_seq = copy.deepcopy(len_seq)
            for i in range(len(len_seq)):
                len_seq[i] = min(len_seq[i], len_vocab - 1)

            ts_seq = ins['ts_seq']
            ipd_seq = [0]
            ipd_seq.extend([ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))])
            real_ipd_seq_us = [i * 1e6 for i in ipd_seq]
            for i in range(len(ipd_seq)):
                ipd_seq[i] = min(ipd_seq[i] * 1e9 // 16384, ipd_vocab - 1)
                assert ipd_seq[i] >= 0

            if len(len_seq) > 100:
                len_seq = len_seq[:100]
                ipd_seq = ipd_seq[:100]
                real_len_seq = real_len_seq[:100]
                real_ipd_seq_us = real_ipd_seq_us[:100]

            combined_seq = torch.tensor([len_seq, ipd_seq], dtype=torch.long).T

            self.flows.append(
                {
                    'combined_seq': combined_seq,
                    'label': ins['label']
                }
            )

        self.flows.sort(key=lambda x: x['combined_seq'].shape[0])

    def __len__(self):
        return len(self.flows)
    
    def __getitem__(self, index):
        flow = self.flows[index]
        return flow['combined_seq'], flow['label']



def seq_collate_fn(batch):

    batch_x, labels = zip(*batch)
    

    len_x_batch = [x[:, 0] for x in batch_x]  
    ipd_x_batch = [x[:, 1] for x in batch_x] 
    padded_len_x = pad_sequence(len_x_batch, batch_first=True)
    padded_ipd_x = pad_sequence(ipd_x_batch, batch_first=True)

    padded_batch_x = torch.stack((padded_len_x, padded_ipd_x), dim=-1)

    labels = torch.tensor(labels, dtype=torch.long)

    return padded_batch_x, labels



def split_sequence_with_overlap(sequence, max_length=100, step_size=50):

    split_sequences = []
    length = sequence.shape[0]

    for start_idx in range(0, length, step_size):
        end_idx = min(start_idx + max_length, length)
        split_sequences.append(sequence[start_idx:end_idx])

    return split_sequences




def split_sequence(sequence):
    min_length = 8
    max_length = len(sequence)
    split_sequences = []

    if max_length < min_length or max_length > 4096:
        raise ValueError("输入序列长度必须在8到4096之间")

    for i in range(min_length, max_length + 1):
        split_sequences.append(sequence[:i])
    
    return split_sequences

class newtestSeqFlowDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, filename):
        super().__init__()
        self.flows = []

        with open(filename) as fp:
            instances = json.load(fp)
        for ins in instances:
            len_seq = ins['len_seq']
            real_len_seq = copy.deepcopy(len_seq)
            for i in range(len(len_seq)):
                len_seq[i] = min(len_seq[i], len_vocab - 1)

            ts_seq = ins['ts_seq']
            ipd_seq = [0]
            ipd_seq.extend([ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))])
            real_ipd_seq_us = [i * 1e6 for i in ipd_seq]
            for i in range(len(ipd_seq)):
                ipd_seq[i] = min(ipd_seq[i] * 1e9 // 16384, ipd_vocab - 1)
                assert ipd_seq[i] >= 0

            if len(len_seq) > 4096:
                len_seq = len_seq[:4096]
                ipd_seq = ipd_seq[:4096]
                real_len_seq = real_len_seq[:4096]
                real_ipd_seq_us = real_ipd_seq_us[:4096]
            
            combined_seq = torch.tensor([len_seq, ipd_seq], dtype=torch.long).T

            split_sequences = split_sequence(combined_seq)

            for seq in split_sequences:
                self.flows.append({
                    'combined_seq': seq,
                    'label': ins['label']
                })

        self.flows.sort(key=lambda x: x['combined_seq'].shape[0])

    def __len__(self):
        return len(self.flows)
    
    def __getitem__(self, index):

        flow = self.flows[index]
        return flow['combined_seq'], flow['label']




class newtrainSeqFlowDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, filename):
        super().__init__()
        self.flows = []


        with open(filename) as fp:
            instances = json.load(fp)
        for ins in instances:
            len_seq = ins['len_seq']
            real_len_seq = copy.deepcopy(len_seq)

            for i in range(len(len_seq)):
                len_seq[i] = min(len_seq[i], len_vocab - 1)

            ts_seq = ins['ts_seq']
            ipd_seq = [0]
            ipd_seq.extend([ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))])
            real_ipd_seq_us = [i * 1e6 for i in ipd_seq]
            for i in range(len(ipd_seq)):
                ipd_seq[i] = min(ipd_seq[i] * 1e9 // 16384, ipd_vocab - 1)
                assert ipd_seq[i] >= 0

            if len(len_seq) > 4096:
                len_seq = len_seq[:4096]
                ipd_seq = ipd_seq[:4096]
                real_len_seq = real_len_seq[:4096]
                real_ipd_seq_us = real_ipd_seq_us[:4096]

            combined_seq = torch.tensor([len_seq, ipd_seq], dtype=torch.long).T

            split_sequences = split_sequence_with_overlap(combined_seq, max_length=100, step_size=50)

            for seq in split_sequences:
                self.flows.append({
                    'combined_seq': seq,
                    'label': ins['label']
                })

        self.flows.sort(key=lambda x: x['combined_seq'].shape[0])

    def __len__(self):
        return len(self.flows)
    
    def __getitem__(self, index):
        flow = self.flows[index]
        return flow['combined_seq'], flow['label']