import torch
import os
import time
import json
import argparse
from model.bigcnnmodel import (
    bigcnn_lut
)
from sklearn.metrics import f1_score,precision_score, recall_score
import numpy as np
import torch.nn.functional as F
from util.bigcnn_data_loader import (
    bigcnnFlowDataset
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

def test(model, data):
    test_loss = 0
    correct = 0
    model.eval()
    nsum = 0
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        if len(data)!=0:
            for x_batch, label_batch in data:
                x_batch = x_batch
                label_batch = label_batch.to(model.device)
                nsum += label_batch.shape[0]
                

                output= model(x_batch)
                test_loss += F.nll_loss(output, label_batch, reduction="sum")
                pred = torch.max(output, dim=-1, keepdim=False)[-1]
                correct += pred.eq(label_batch.data).sum()
                all_preds.append(pred.cpu().numpy())
                all_labels.append(label_batch.cpu().numpy())
            
            
            test_loss = test_loss / nsum
            all_preds = np.concatenate(all_preds)  
            all_labels = np.concatenate(all_labels)  
            f1_macro = f1_score(all_labels, all_preds, average='macro')
            macro_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            macro_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)

            print(f1_macro)

            return f1_macro

def build_data_loader(len_vocab, ipd_vocab, window_size, filename, batch_size, args, is_train=False, shuffle=True):
    dataset = bigcnnFlowDataset(len_vocab, ipd_vocab, filename, window_size, args)
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




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset",type=str)
parser.add_argument("--testpth",type=str)
parser.add_argument("--ptpth",type=str)
parser.add_argument("--device",default=0)
parser.add_argument("--model_name",type=str)
args = parser.parse_args()




def run(args):
    test_path = args.testpth
    weight = torch.load(args.ptpth, weights_only=True)

    if args.dataset == "ISCXVPN":
        numclass = 6
    else:
        numclass = 3
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
    model = bigcnn_lut(numclass,device,6,16)


    batch_size = 256
    test_loader = build_data_loader(1501, 2561, 8, test_path, batch_size, args, is_train=False, shuffle=False)
    model.load_state_dict(weight)

    f1 = test(model, test_loader)

if __name__ == "__main__":
    run(args)

