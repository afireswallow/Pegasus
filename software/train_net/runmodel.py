import os
import time
import json
import argparse
import torch
from sklearn.metrics import f1_score

from model.cnnmodel import (
    TextCNN1,
    TextCNN2,
    TextCNN3
)

from model.mlpmodel import (
    TwoLayerPerceptron_norm1,
    TwoLayerPerceptron_norm2,
    TwoLayerPerceptron_norm3,
    TwoLayerPerceptron_norm4
)

from model.rnnmodel import(
    RNN1,
    RNN2,
    RNN3
)


from model.cnnmmodel import (
    originenlargeTextCNN,
    enlargesegTextCNN1,
    enlargesegTextCNN2,
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

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

def build_data_loader(len_vocab, ipd_vocab, window_size, filename, batch_size, args, is_train=False, shuffle=True):
    dataset = FlowDataset(len_vocab, ipd_vocab, filename, window_size, args)
    val_ratio = args.val_ratio
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
    
    train_path = args.trainpth
    test_path = args.testpth


    if args.model == "mlp":
        train_loader, val_loader, known_loader  = new_mlp_dataloader(train_path, args.model, args, batch_size=batch_size, shuffle=args.shuffle, istrained = True)
        test_loader = new_mlp_dataloader(test_path, args.model, args, batch_size=batch_size, shuffle=False, istrained = False)
    else:
        train_loader, val_loader, known_loader = build_data_loader(len_vocab, ipd_vocab, window_size, train_path, batch_size, args, is_train=True, shuffle=args.shuffle)
        test_loader = build_data_loader(len_vocab, ipd_vocab, window_size, test_path, batch_size, args,is_train=False, shuffle=False)
    

    device = args.device
    dlist = args.dlist
    dlist = [int(char) for char in dlist]
    droprate=0
    ptpth = args.ptpth
    modellist = []

    
    if args.model == "rnn":
        if args.modelnum == 1:
            model1 = RNN1(
                    args.rnnin, args.rnnhs,
                    args.num_classes,
                    len_vocab, ipd_vocab,
                    len_embedding_bits, ipd_embedding_bits,
                    device, droprate)
            if args.ptpth!=None and args.trained and args.modelnum==1:
                weights = torch.load(args.ptpth)
                model1.load_state_dict(weights)
            modellist.append(model1)
        if args.modelnum <= 2:
            model2 = RNN2(
                    args.rnnin, args.rnnhs,
                    args.num_classes,
                    len_vocab, ipd_vocab,
                    len_embedding_bits, ipd_embedding_bits,
                    device, droprate,
                    dlist[0])
            if args.ptpth!=None and args.trained and args.modelnum==2:
                weights = torch.load(args.ptpth)
                model2.load_state_dict(weights)
            modellist.append(model2)
        if args.modelnum <= 3:
            model3 = RNN3(
                    args.rnnin, args.rnnhs,
                    args.num_classes,
                    len_vocab, ipd_vocab,
                    len_embedding_bits, ipd_embedding_bits,
                    device, droprate,
                    dlist[0],dlist[1])
            modellist.append(model3)
        savepth = args.savepth+"/"+str(args.model)+"/"+str(args.dataset)+"_"+str(args.rnnhs)+"_"+str(args.rnnin)+"_"+str(args.lr)+"l_"+str(args.step)+"s_"+str(args.patience)+"p"

    if args.model == "mlp":
        if args.modelnum == 1:
            model1 = TwoLayerPerceptron_norm1(
                10, args.mlphs1, 
                args.mlphs2, args.num_classes, device)
            if args.ptpth!=None and args.trained and args.modelnum==1:
                weights = torch.load(args.ptpth)
                model1.load_state_dict(weights)
            modellist.append(model1)
        if args.modelnum <= 2:
            model2 = TwoLayerPerceptron_norm2(
                10, args.mlphs1, 
                args.mlphs2, args.num_classes, device,
                dlist[0])
            if args.modelnum == 2 and args.ptpth!=None and args.trained:
                weights = torch.load(args.ptpth)
                model2.load_state_dict(weights)
            modellist.append(model2)
        if args.modelnum <= 3:
            model3 = TwoLayerPerceptron_norm3(
                10, args.mlphs1, 
                args.mlphs2, args.num_classes, device,
                dlist[0],dlist[1])
            if args.modelnum == 3 and args.ptpth!=None and args.trained:
                weights = torch.load(args.ptpth)
                model3.load_state_dict(weights)
            modellist.append(model3)
        if args.modelnum <= 4:
            model4 = TwoLayerPerceptron_norm4(
                10, args.mlphs1, 
                args.mlphs2, args.num_classes, device,
                dlist[0],dlist[1],dlist[2])
            modellist.append(model4)
        savepth = args.savepth+"/"+str(args.model)+"/"+str(args.dataset)+"_"+str(args.mlphs1)+"_"+str(args.mlphs2)+"_"+str(args.lr)+"l_"+str(args.step)+"s_"+str(args.patience)+"p"

    if args.model == "cnnb":
        if args.modelnum == 1:
            model1 = TextCNN1(8, args.num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 2, 4,
                 device)
            if args.ptpth!=None and args.trained and args.modelnum==1:
                weights = torch.load(args.ptpth)
                model1.load_state_dict(weights)
            modellist.append(model1)
        if args.modelnum <= 2:
            model2 = TextCNN2(8, args.num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 2, 4,
                 device,2)
            if args.ptpth!=None and args.modelnum == 2 and args.trained:
                weights = torch.load(args.ptpth)
                model2.load_state_dict(weights)
            modellist.append(model2)       
        if args.modelnum <= 3:
            model3 = TextCNN3(8, args.num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 2, 4,
                 device,2,2)
            modellist.append(model3)
        savepth = args.savepth+"/"+str(args.model)+"/"+str(args.dataset)+"_"+str(args.nk)+"_"+str(args.lr)+"l_"+str(args.step)+"s_"+str(args.patience)+"p"
    
    if args.model == "cnnm":
        if args.modelnum == 1:
            model1 = originenlargeTextCNN(8, 64,args.num_classes,
                                          len_vocab,ipd_vocab,
                                          len_embedding_bits,ipd_embedding_bits,
                                          32,4,device)
            modellist.append(model1)
        # No lookup-table quantization is required between the first and second stages of cnnm. 
        # The stages are trained separately to provide better-initialized weights for the sliced model.
        if args.modelnum == 2:
            model2 = enlargesegTextCNN1(8, 64,args.num_classes,
                                          len_vocab,ipd_vocab,
                                          len_embedding_bits,ipd_embedding_bits,
                                          32,4,device)
            if args.ptpth!=None :
                weights = torch.load(args.ptpth)
                model2.load_state_dict(weights)
            modellist.append(model2)
        if args.modelnum == 3:
            model3 = enlargesegTextCNN2(8, 64,args.num_classes,
                                          len_vocab,ipd_vocab,
                                          len_embedding_bits,ipd_embedding_bits,
                                          32,4,2,device)
            modellist.append(model3)
            savepth = args.savepth+"/"+str(args.model)+"/"+str(args.dataset)+"_"+str(args.nk)+"_"+str(args.lr)+"l_"+str(args.step)+"s_"+str(args.patience)+"p"
            




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
    parser.add_argument("--model",required=True,choices=["rnn","mlp","cnnm","cnnb"])
    #rnn
    parser.add_argument("--rnnhs",type=int,default=8)
    parser.add_argument("--rnnin", type=int, default=8)
    #cnn
    parser.add_argument("--nk",type=int,default=2)
    parser.add_argument("--cnnin",type=int,default=4)
    parser.add_argument("--largein",type=int,default=64)
    #mlp
    parser.add_argument("--mlphs1",type=int,default=16)
    parser.add_argument("--mlphs2",type=int,default=8)
    parser.add_argument("--mlphs3",type=int,default=8)
    parser.add_argument("--mlphs4",type=int,default=8)
    parser.add_argument("--mlphs5",type=int,default=8)
    
    
    #Training
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--device",choices=["0","1","2","3","cpu"], required=True)
    parser.add_argument("--savepth", type=str, default=None,required=True)
    parser.add_argument("--dlist", type=str, default=None,required=True)
    parser.add_argument("--ptpth", type=str, default=None)
    parser.add_argument("--LSTpth", type=str, default=None)
    parser.add_argument("--trainpth",type=str, required=True)
    parser.add_argument("--testpth",type=str, required=True)
    parser.add_argument("--step",type = int, default = 10)
    parser.add_argument("--patience",type = int, default = 20)
    parser.add_argument("--masked",type = int,default=0)
    parser.add_argument("--trained",type = lambda x: bool(strtobool(x)),default=False)
    parser.add_argument("--shuffle",type=lambda x: bool(strtobool(x)), default = True)
    parser.add_argument("--onlypacket",type=lambda x: bool(strtobool(x)), default = False)
    parser.add_argument("--seg",type=lambda x: bool(strtobool(x)), default = False)
    parser.add_argument("--modelnum",type = int,default=0,required=True)
    parser.add_argument("--confirmterm",type = int,default=10)
    

    args = parser.parse_args()
    run(args)