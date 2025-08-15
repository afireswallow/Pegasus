import os
import time
import json
import argparse
import torch
from sklearn.metrics import f1_score

from model.bigcnnmodel import (
    bigcnn2_getdata_w,
    bigcnn2_getdata_b,
)



from distutils.util import strtobool
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from util.early_stopping import *
from util.testbrnndataloader import (
    FlowDataset
)

from util.newmlpdataloader import new_mlp_dataloader
from util.seed import set_seed
import os
from .autoquanti import quanti_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import csv
from util.bigcnn_data_loader import (
    bigcnnFlowDataset
)
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
class CustomDataset(Dataset):
    def __init__(self, csv_file,args):
        data = pd.read_csv(csv_file)
        self.labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.long) 
        self.features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32) 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def build_bigcnn_data_loader(len_vocab, ipd_vocab, window_size, filename, batch_size, args, is_train=False, shuffle=True):
    if args.filetype == "json":
        dataset = bigcnnFlowDataset(len_vocab, ipd_vocab, filename, window_size, args)
    elif args.filetype == "csv":
        dataset = CustomDataset(filename, args)
    else:
        print("filewrong")
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



def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        


def test(model, data, args,train=False):
    check_and_create_directory(args.savepth)
    test_loss = 0
    correct = 0
    model.eval()
    nsum = 0
    all_preds = []
    all_labels = []
    model.eval()
    alldata = []
    if train:
        filepth = args.savepth+"/"+args.dataset+"_"+args.mode+"_train.csv"
        labelpth = args.savepth+"/"+args.dataset+"_label_"+args.mode+"_train.csv"
        print("save train data")
    else:
        filepth = args.savepth+"/"+args.dataset+"_"+args.mode+"_test.csv"
        labelpth = args.savepth+"/"+args.dataset+"_label_"+args.mode+"_test.csv"
        print("save test data")

    if args.mode == 'w':
        with open(labelpth, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            with torch.no_grad():
                for x_batch, label_batch in data:
                    label_batch = label_batch.cpu().numpy()
                    for label in label_batch:
                        csv_writer.writerow([label])
    
    with open(filepth, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        with torch.no_grad():
            for x_batch, label_batch in data:
                x_batch = x_batch
                label_batch = label_batch.to(model.device)
                nsum += label_batch.shape[0]
                
                output, data = model(x_batch)
                data = data[0]
                if isinstance(data, torch.Tensor):
                    data = data.cpu().numpy()
                if isinstance(label_batch, torch.Tensor):
                    label_batch = label_batch.cpu().numpy()
                for features in data:
                    csv_writer.writerow(features.tolist())
                




def run(args):
    batch_size = 256
    set_seed(3407)
    len_vocab=args.len_vocab
    ipd_vocab=args.ipd_vocab
    len_embedding_bits=args.len_embedding_bits
    ipd_embedding_bits=args.ipd_embedding_bits
    window_size=args.window_size
    device_args =args.device
    if device_args !="cpu":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device_args}")
        else:
            print("device error")
            return "device error"
    else:
        device = torch.device("cpu")
    

    if args.dataset == "ISCXVPN":
        args.num_classes = 6
    else:
        args.num_classes = 3

    test_path = args.testpth
    train_path = args.trainpth

    train_loader, val_loader, known_loader = build_bigcnn_data_loader(len_vocab, ipd_vocab, window_size, train_path, batch_size, args, is_train=True, shuffle=False)
    test_loader = build_bigcnn_data_loader(len_vocab, ipd_vocab, window_size, test_path, batch_size, args,is_train=False, shuffle=False)


    droprate=0
    modellist = []
    patience = args.patience
    step = args.step
    lr = args.lr
    weight = torch.load(args.ptpth)
    epsilon = 1e-5
    print(args.mode)
    if args.mode == "w":
        weight['norm1.bias'] = torch.zeros_like(weight['norm1.bias'])
        weight['norm2.bias'] = torch.zeros_like(weight['norm2.bias'])
        weight['norm3.bias'] = torch.zeros_like(weight['norm3.bias'])
        weight['norm4.bias'] = torch.zeros_like(weight['norm4.bias'])
        weight['fc1.bias'] = torch.zeros_like(weight['fc1.bias'])
        weight['fc2.bias'] = torch.zeros_like(weight['fc2.bias'])
        weight['fc3.bias'] = torch.zeros_like(weight['fc3.bias'])
        weight['fc4.bias'] = torch.zeros_like(weight['fc4.bias'])
        weight['norm1.running_mean'] = torch.zeros_like(weight['norm1.running_mean'])
        weight['norm2.running_mean'] = torch.zeros_like(weight['norm2.running_mean'])
        weight['norm3.running_mean'] = torch.zeros_like(weight['norm3.running_mean'])
        weight['norm4.running_mean'] = torch.zeros_like(weight['norm4.running_mean'])
        model = bigcnn2_getdata_w(args.num_classes,device,6,16)
        #norm(x)=k*x+b
    elif args.mode == "b":
        model = bigcnn2_getdata_b(args.num_classes,device,6,16)
        pass
    else:
        print("please choose mode w or b")


    print(model.load_state_dict(weight,strict=False))
    test(model, test_loader, args, train=False)
    test(model, known_loader, args, train=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset
    parser.add_argument("--len_vocab", default=1501)
    parser.add_argument("--len_embedding_bits",default=10)
    parser.add_argument("--ipd_vocab", type=int, default=2561)
    parser.add_argument("--ipd_embedding_bits", type=int, default=8)
    parser.add_argument("--dataset",type=str)



    # Model
    #rnn
    parser.add_argument("--rnnhs",type=int,default=8)
    parser.add_argument("--rnnin", type=int, default=8)
    #cnn
    parser.add_argument("--nk",type=int,default=2)
    parser.add_argument("--cnnin",type=int,default=4)
    #mlp
    parser.add_argument("--mlphs1",type=int,default=16)
    parser.add_argument("--mlphs2",type=int,default=8)
    parser.add_argument("--mlphs3",type=int,default=8)
    parser.add_argument("--mlphs4",type=int,default=8)
    parser.add_argument("--mlphs5",type=int,default=8)

    
    
    #Training
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--device",choices=["0","1","2","3","cpu"], default="3")
    parser.add_argument("--ptpth", type=str, default=None)
    parser.add_argument("--trainpth", type=str, default=None)
    parser.add_argument("--testpth", type=str, default=None)
    parser.add_argument("--savepth", type=str, default=None)
    parser.add_argument("--staticpth", type=str, default=None)
    parser.add_argument("--step",type = int, default = 10)
    parser.add_argument("--patience",type = int, default = 20)
    parser.add_argument("--masked",type = int,default=0)
    parser.add_argument("--trained",type = lambda x: bool(strtobool(x)),default=True)
    parser.add_argument("--shuffle",type=lambda x: bool(strtobool(x)), default = True)
    parser.add_argument("--onlypacket",type=lambda x: bool(strtobool(x)), default = False)
    parser.add_argument("--model",type=str)
    parser.add_argument("--taskid",type=str)
    parser.add_argument("--mode",type=str)
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--filetype",type=str,default="json")


    """
    Example:
    python -m train_net.cnnl_getsegdata --dataset CICIOT2022 --ptpth ptpth.pt --savepth middleoutput \
        --trainpth dataset/CICIOT2022/redeal_train.json --testpth dataset/CICIOT2022/redeal_test.json \
        --device 0
    """

    args = parser.parse_args()
    args.mode = "w"
    run(args)
    args.mode = "b"
    run(args)
