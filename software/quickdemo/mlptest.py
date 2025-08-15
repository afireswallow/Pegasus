import os
import time
import json
import argparse
import torch
from sklearn.metrics import f1_score
from util.reset import(
    reset_fc,
    reset_relu,
    adjust_norm_layer,
    reset_norm_to_T,
    reset_norm_to_LUT,
    reset_int,
    reset_int_double,
    reset_int_single,
    reset_norm_to_T_
)
from model.mlpmodel import (
    mlp_lut
)

from distutils.util import strtobool
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from util.early_stopping import *
from util.data_loader import (
    FlowDataset,
    SeqFlowDataset,
    seq_collate_fn
)
from util.newmlpdataloader import new_mlp_dataloader
from util.seed import set_seed
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset
parser.add_argument("--len_vocab", default=1501)
parser.add_argument("--len_embedding_bits",default=10)
parser.add_argument("--ipd_vocab", type=int, default=2561)
parser.add_argument("--ipd_embedding_bits", type=int, default=8)
parser.add_argument("--dataset", default= "ISCXVPN")
# Model
parser.add_argument("--model",default="mlp")
parser.add_argument("--mlphs1",type=int,default=32)
parser.add_argument("--mlphs2",type=int,default=16)
parser.add_argument("--num",type=int, default=2021)
parser.add_argument("--window_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--num_classes", type=int, default=None)
parser.add_argument("--device",default=0)
parser.add_argument("--savepth", type=str, default=None)
parser.add_argument("--dlist", type=str, default=111)
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
args = parser.parse_args()








def run(args):
    test_loader = new_mlp_dataloader(args.testpth, args.model, args, batch_size=10000, shuffle=False, istrained = False)
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
        args.num_classes = 6
    else:
        args.num_classes = 3
    model = mlp_lut(10, args.mlphs1, args.mlphs2, args.num_classes, device,1,2,1)
    state = torch.load(args.ptpth, weights_only=True)
    model.load_state_dict(state,strict=False)
    test_loss = 0
    correct = 0
    model.eval()
    nsum = 0
    all_preds = []
    all_labels = []
        
    with torch.no_grad():
        for x_batch, label_batch in test_loader:
            x_batch = x_batch
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


