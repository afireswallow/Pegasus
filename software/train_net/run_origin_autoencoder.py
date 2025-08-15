import os
import time
import json
import argparse
import torch
from sklearn.metrics import f1_score
import copy

from model.splitAutoencoder import(
    original_splitautoencoder
)
from distutils.util import strtobool
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from util.early_stopping import *
from util.data_loader import (
    newencoderDataset,
    FlowDataset
)
from util.newmlpdataloader import new_mlp_dataloader
from util.seed import set_seed
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,confusion_matrix
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import json


def write_log(log_path, log_str):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_str)

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def build_unified_data_loader(len_vocab, ipd_vocab, window_size, filenames, batch_size, shuffle=True):
    dataset = newencoderDataset(len_vocab, ipd_vocab, window_size, filenames)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def build_data_loader(len_vocab, ipd_vocab, window_size, filename, batch_size, args, is_train=False, shuffle=True):
    dataset = FlowDataset(len_vocab, ipd_vocab, filename, window_size, args)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def train(optimizer, model, data):
    model.train()
    n =0
    train_loss = 0
    criterion = torch.nn.L1Loss()
    for x_batch, label_batch in data:
        x_batch = x_batch.to(model.device)
        label_batch = label_batch.to(model.device)
        
        output, ebd = model(x_batch)
        output = output.view(output.shape[0],-1)
        ebd = ebd[0].view(x_batch.shape[0],-1)
        loss = criterion(output, ebd)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n+=label_batch.shape[0]
        train_loss += loss 
    train_loss = train_loss / n
    return train_loss



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset
parser.add_argument("--len_vocab", default=1501)
parser.add_argument("--len_embedding_bits",default=10)
parser.add_argument("--ipd_vocab", type=int, default=2561)
parser.add_argument("--ipd_embedding_bits", type=int, default=8)
parser.add_argument("--ebdin", type=int, default=4)
parser.add_argument("--dataset", type=str)
parser.add_argument("--savepth", type=str)
parser.add_argument("--trainpth", type=str)
parser.add_argument("--ptpth", type=str)
parser.add_argument("--LSTpth", type=str)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--window_size", type=int, default=8)
parser.add_argument("--device",type=int)
parser.add_argument("--step", type=int, default=10)

args = parser.parse_args()

train_loader= build_data_loader(args.len_vocab, args.ipd_vocab, args.window_size, args.trainpth, 256, args, is_train=True, shuffle=True)
device = torch.device(f"cuda:{args.device}")

model = original_splitautoencoder(8, 
            args.len_vocab,args.ipd_vocab,
            args.len_embedding_bits,args.ipd_embedding_bits,
            args.ebdin,64,48,24,12,
            device)

patience = args.patience
step  = args.step
lr = args.lr

logpth = f"{args.savepth}/autoencoder/{args.dataset}/{lr}/log.txt"
savepth = f"{args.savepth}/autoencoder/{args.dataset}/{lr}"
ensure_dir(savepth)
early_stop = EarlyStopping(patience=patience, delta=0, verbose=False)
acc = 0
f1 = 0
epoch = 0
weights=[]
auc = 0
while True:
    if epoch == 30:
        break
    train_loss = float('inf')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_loss=train(optimizer,model,train_loader)
    epoch+=1
    if epoch%step==0 :
        lr /= 2
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
    modelname = "AUTOencoder"
    num = 1
    txt = 'modelnum: {}{}|lr:{}|epoch: {}|trainloss: {:.4f}'.format(
                modelname,num,args.lr,epoch,train_loss
            )
    print(txt)

torch.save(model.state_dict(), f"{savepth}/autoencoder_origin_{epoch}e.pt")   


from util.quanti import model_quanti
from util.getdata import get_data
weight = model.state_dict()
train_data = get_data(model, train_loader, weight)
pth = args.LSTpth
if len(train_data) == 1:
    S, T, LUT = model_quanti(train_data[0],2,1,pth)
    torch.save(S, f"{savepth}/S.pt")
    torch.save(T, f"{savepth}/T.pt")
    torch.save(LUT, f"{savepth}/LUT.pt")
