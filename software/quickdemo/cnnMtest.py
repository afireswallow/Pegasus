import torch
from model.cnnmmodel import (
    enlargesegTextCNN_LUT,
)
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
import numpy as np
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from util.data_loader import (
    FlowDataset
)
import argparse
from distutils.util import strtobool


def build_data_loader(len_vocab, ipd_vocab, window_size, filename, batch_size, args, is_train=False, shuffle=True):
    dataset = FlowDataset(len_vocab, ipd_vocab, filename, window_size, args)
    val_ratio = 0.1
    known_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    if is_train:
        labels = [dataset[i][1] for i in range(len(dataset))] 
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=val_ratio,
            stratify=labels 
        )
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader, val_loader, known_loader
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

def test(model, data):
    test_loss = 0
    correct = 0
    model.eval()
    nsum = 0
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x_batch, label_batch in data:
            x_batch = x_batch
            label_batch = label_batch.to(model.device)
            nsum += label_batch.shape[0]
            
            output = model(x_batch)
            test_loss += F.nll_loss(output, label_batch, reduction="sum")
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct += pred.eq(label_batch.data).sum()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())

        test_loss = test_loss / nsum
        all_preds = np.concatenate(all_preds) 
        all_labels = np.concatenate(all_labels)  
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        return 100. * correct / nsum, test_loss, f1_macro, precision_macro, recall_macro



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset
parser.add_argument("--len_vocab", default=1501)
parser.add_argument("--len_embedding_bits",default=10)
parser.add_argument("--ipd_vocab", type=int, default=2561)
parser.add_argument("--ipd_embedding_bits", type=int, default=8)

parser.add_argument("--dataset", default= "PeerRush")


    # Model
parser.add_argument("--model",default="mlp")
parser.add_argument("--mlphs1",type=int,default=16)
parser.add_argument("--mlphs2",type=int,default=8)
    
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
    ptpth = args.ptpth
    test_pth = args.testpth
    weight = torch.load(ptpth, map_location=device, weights_only=True)
    model = enlargesegTextCNN_LUT(8, 64, num_classes, 1501, 2561, 10, 8, 32, 4, 2, device)
    model.eval()
    model.load_state_dict(weight)
    batchsize = 1000
    test_loader = build_data_loader(1501, 2561, 8, test_pth, batchsize, args, is_train=False, shuffle=False)


    acc, loss, f1_macro, pr_macro, rc_macro = test(model,test_loader)

    return f1_macro


if __name__ == '__main__':
    f1_macro = run(args)
    print(f"F1 Macro: {f1_macro:.4f}")
