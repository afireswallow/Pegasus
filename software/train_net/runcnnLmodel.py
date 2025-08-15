import os
import time
import json
import argparse
import torch
from sklearn.metrics import f1_score


from model.bigcnnmodel import(
    bigcnn1_linear,
    bigcnn1_quantilinear,
    bigcnn2_seg_conv,
    bigcnn3_linear_after_conv,

)
from util.bigcnn_data_loader import (
    bigcnnFlowDataset
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
from .autoquanti import quanti_sequence
import csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd


def save_data(dataloader,output_file,args,train=True):
    data = []

    for inputs, labels in dataloader:
        inputs = inputs.numpy()  # 转为 NumPy
        labels = labels.numpy()

        for i in range(len(inputs)):
            flattened_input = inputs[i].flatten()  # 展平 [8, 60] -> [480]
            row = list(flattened_input) + [labels[i]]  # 拼接 label
            data.append(row)

    # 保存到 CSV
    header = [f'feature_{i+1}' for i in range(8 * 60)] + ['label']  # 特征列名
    if train:
        name = "_train"
    else:
        name = "_test"
    with open(output_file+"/"+args.dataset+name+".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入表头
        writer.writerows(data)  # 写入数据
    print("save data success")


def getweights(ptpth,LSTpth,modelnum):
    # if ptpth == None:
    #     weights = {}
    #     print("only LST")
    if modelnum == 2:
        if ptpth !=None:
            oldweights = torch.load(ptpth)
            weights = {}
            weights['fc1.weight'] = oldweights['fc3.weight']
            weights['fc1.bias'] = oldweights['fc3.bias']
        else:
            weights = {}
    else:
        weights = torch.load(ptpth)
    weights[f'MM{modelnum}.S'] = torch.load(LSTpth+"/S_all.pth",weights_only=False)
    weights[f'MM{modelnum}.LUT'] = torch.load(LSTpth+"/LUT_all.pth",weights_only=False)
    weights[f'MM{modelnum}.T'] = torch.load(LSTpth+"/T_all.pth",weights_only=False)
    weights[f'MM{modelnum}.H'] = torch.load('./util/H.pth',weights_only=False)
    return weights

class CustomDataset(Dataset):
    def __init__(self, csv_file,args):
        data = pd.read_csv(csv_file)
        self.labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)  # 第一列为标签
        self.features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32)  # 后面128列为特征

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

def build_bigcnn_data_loader(len_vocab, ipd_vocab, window_size, filename, batch_size, args, is_train=False, shuffle=True):
    if args.mode == "csv":
        dataset = CustomDataset(filename, args)
    else:
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



def run(args):
    batch_size = 256
    dataset = args.dataset
    len_vocab=args.len_vocab
    ipd_vocab=args.ipd_vocab
    len_embedding_bits=args.len_embedding_bits
    ipd_embedding_bits=args.ipd_embedding_bits
    window_size=args.window_size
    device_args =args.device
    if device_args !="cpu":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device_args}")
            args.device = device
        else:
            print("device error")
            return "device error"
    else:
        device = torch.device("cpu")
        args.device = device
    print(args.device)
    if args.dataset == "ISCXVPN":
        args.num_classes = 6
    else:
        args.num_classes = 3
    if args.modelnum >2:
        args.mode = "csv"
    train_path = args.trainpth
    test_path = args.testpth


    if args.model == "bigcnn":
        train_loader, val_loader, known_loader = build_bigcnn_data_loader(len_vocab, ipd_vocab, window_size, train_path, batch_size, args, is_train=True, shuffle=args.shuffle)
        test_loader = build_bigcnn_data_loader(len_vocab, ipd_vocab, window_size, test_path, batch_size, args,is_train=False, shuffle=False)
    else:
        print("model not supported")
    device = args.device
    dlist = args.dlist
    dlist = [int(char) for char in dlist]
    droprate=0
    ptpth = args.ptpth
    modellist = []

    if args.model == "bigcnn":
        if args.modelnum == 1:   
            model1 = bigcnn1_linear(args.num_classes,device,6,16)
            if args.modelnum == 1 and args.trained and args.ptpth != None:
                weights = torch.load(args.ptpth,weights_only=False)
                model1.load_state_dict(weights,strict=False)
            modellist.append(model1)         
        if args.modelnum <= 2:
            model2 = bigcnn1_quantilinear(args.num_classes,device,6,16)
            modellist.append(model2)
        # For convenience in training, the intermediate outputs from the previous model are extracted as the training data for the subsequent stage. 
        # Therefore, continuous end-to-end training is not supported.
        # To obtain the intermediate outputs from the previous layer, please run cnnl_getsegdata.py and cannl_modifydata.py.
        if args.modelnum == 3:
            model3 = bigcnn2_seg_conv(args.num_classes,device,6,16)
            if args.modelnum == 3 and args.trained and args.ptpth != None:
                weights = torch.load(args.ptpth,weights_only=False)
                model3.load_state_dict(weights,strict=False)
            modellist.append(model3)
        if args.modelnum <= 4 and args.modelnum >=3:
            model4 = bigcnn3_linear_after_conv(args.num_classes,device,6,16)
            modellist.append(model4)
            
        savepth = args.savepth+"/"+str(args.model)+"/"+str(args.dataset)+"_"+str(args.lr)+"l_"+str(args.step)+"s_"+str(args.patience)+"p"

 

    patience = args.patience
    step = args.step
    lr = args.lr
    
    
    quanti_sequence(
                args.model,
                modellist, 
                train_loader,
                val_loader,
                known_loader,
                test_loader,
                device, patience, 
                dlist, savepth, 
                lr, step, args,
                istrained=args.trained,
                taskname=f"{args.dataset}"+f"_{args.lr}lr"+f"_{args.step}s"+f"_{args.patience}p"+f"_{args.dlist}D")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset
    parser.add_argument("--len_vocab", default=1501)
    parser.add_argument("--len_embedding_bits",default=10)
    parser.add_argument("--ipd_vocab", type=int, default=2561)
    parser.add_argument("--ipd_embedding_bits", type=int, default=8)

    parser.add_argument("--dataset", required=True,
                        choices=["ISCXVPN", "BOTIOT", "PeerRush", "UNSWNB15","CICIOT2022"])


    # Model
    parser.add_argument("--model",default='bigcnn')
    
    #Training
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--device",choices=["0","1","2","3","cpu"], required=True)
    parser.add_argument("--savepth", type=str, default=None,required=True)
    parser.add_argument("--dlist", type=str, default="22")
    parser.add_argument("--ptpth", type=str, default=None)
    parser.add_argument("--LSTpth", type=str, default=None)
    parser.add_argument("--trainpth",type=str, required=True)
    parser.add_argument("--testpth",type=str, required=True)
    parser.add_argument("--step",type = int, default = 10)
    parser.add_argument("--nk",type = int, default = 16)
    parser.add_argument("--patience",type = int, default = 20)
    parser.add_argument("--masked",type = int,default=0)
    parser.add_argument("--trained",type = lambda x: bool(strtobool(x)),default=False)
    parser.add_argument("--shuffle",type=lambda x: bool(strtobool(x)), default = True)
    parser.add_argument("--onlypacket",type=lambda x: bool(strtobool(x)), default = False)
    parser.add_argument("--seg",type=lambda x: bool(strtobool(x)), default = False)
    parser.add_argument("--modelnum",type = int,default=0,required=True)
    parser.add_argument("--confirmterm",type = int, default = 10)
    parser.add_argument("--mode", type=str,default="json")
    

    args = parser.parse_args()
    run(args)
