import os
import time
import json
import argparse
import torch
from sklearn.metrics import f1_score
from model.rnnmodel import (
    rnn_lut
)
from distutils.util import strtobool
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from util.early_stopping import *
from util.testbrnndataloader import(
    FlowDataset
)
from util.seed import set_seed
import os

def build_data_loader(len_vocab, ipd_vocab, window_size, filename, batch_size, args, is_train=False, shuffle=True):
    dataset = FlowDataset(len_vocab, ipd_vocab, filename, window_size, args)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print('The size of {}_set is {}.'.format('train' if is_train else 'test', len(dataset)))
    return data_loader


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset
parser.add_argument("--len_vocab", default=1501)
parser.add_argument("--len_embedding_bits",default=10)
parser.add_argument("--ipd_vocab", type=int, default=2561)
parser.add_argument("--ipd_embedding_bits", type=int, default=8)
    # Model
parser.add_argument("--model",default="mlp")
parser.add_argument("--mlphs1",type=int,default=32)
parser.add_argument("--mlphs2",type=int,default=16)
parser.add_argument("--num",type=int, default=2022)
parser.add_argument("--window_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--num_classes", type=int, default=None)
parser.add_argument("--device",default=0)
parser.add_argument("--savepth", type=str, default=None)
parser.add_argument("--dlist", type=str, default=121)
parser.add_argument("--ptpth", type=str, default=None)
parser.add_argument("--trainpth",type=str)
parser.add_argument("--testpth",type=str)
parser.add_argument("--staticpth",type=str)
parser.add_argument("--step",type = int)
parser.add_argument("--patience",type = int)
parser.add_argument("--masked",type = int)
parser.add_argument("--trained",type = lambda x: bool(strtobool(x)),default=True)
parser.add_argument("--shuffle",type=lambda x: bool(strtobool(x)), default = True)
parser.add_argument("--onlypacket",type=lambda x: bool(strtobool(x)), default = False)
parser.add_argument("--modelnum",type=int,default=0)
parser.add_argument("--dataset", default= "CICIOT2022")
args = parser.parse_args()

def run(args):
    test_loader = build_data_loader(1501, 2561, 8, args.testpth, 10000, args, is_train=False, shuffle=False)
    state = torch.load(args.ptpth, weights_only=True)
    if args.device !="cpu":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device}")
            args.device = device
        else:
            print("device error")
            return "device error"
    else:
        device = torch.device("cpu")
        args.device = device

    if args.dataset == "ISCXVPN":
        num_classes = 6
    else:
        num_classes = 3


    model = rnn_lut(8, 8, num_classes, 1501, 2561, 10, 8, device, 0, 1, 1)

    state['MM1.T'] = torch.nan_to_num(state['MM1.T'],nan=0.0,posinf=2 ** 11 ,neginf=-2 ** 11)

    model.load_state_dict(state)


    test_loss = 0
    correct = 0
    model.eval()
    nsum = 0
    all_preds = []
    all_labels = []
        
    with torch.no_grad():
        for x_batch, label_batch in test_loader:
            x_batch = x_batch
            #print(x_batch)
            label_batch = label_batch.to(model.device)
            nsum += label_batch.shape[0]
                
            output= model(x_batch)
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct += pred.eq(label_batch.data).sum()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())

        all_preds = np.concatenate(all_preds)  
        all_labels = np.concatenate(all_labels)  
        f1_macro = f1_score(all_labels, all_preds, average='macro')

    return f1_macro

if __name__ == '__main__':
    f1_macro = run(args)
    print(f"F1 Macro: {f1_macro:.4f}")





"""
python rnntest.py --dataset CICIOT2022 --testpth /mnt/sdc_space/oldz/zlx/githubversion/finalfile/dataset/CICIOT2022/redeal_test.json --ptpth /mnt/sdc_space/oldz/zlx/githubversion/finalfile/save/rnn/CICIOT2022/rnn_CICIOT2022_0.8838013484275483.pt --device 0 
python rnntest.py --dataset ISCXVPN --testpth /mnt/sdc_space/oldz/zlx/githubversion/finalfile/dataset/ISCXVPN/redeal_test.json --ptpth /mnt/sdc_space/oldz/zlx/githubversion/finalfile/save/rnn/ISCXVPN/rnn_ISCXVPN_0.7661085936716739.pt --device 0 
python rnntest.py --dataset PeerRush --testpth /mnt/sdc_space/oldz/zlx/githubversion/finalfile/dataset/PeerRush/redeal_test.json --ptpth /mnt/sdc_space/oldz/zlx/githubversion/finalfile/save/rnn/PeerRush/rnn_PeerRush_0.9102976579479009.pt --device 0
"""