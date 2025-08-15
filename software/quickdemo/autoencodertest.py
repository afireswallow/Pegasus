import os
import time
import json
import argparse
import torch
from sklearn.metrics import f1_score
import copy

from model.splitAutoencoder import(
    original_splitautoencoder,
    MM_splitautoencoder,
    table_splitautoencoder_template
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


def build_unified_data_loader(len_vocab, ipd_vocab, window_size, filenames, batch_size, shuffle=True):
    dataset = newencoderDataset(len_vocab, ipd_vocab, window_size, filenames)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def build_data_loader(len_vocab, ipd_vocab, window_size, filename, batch_size, args, is_train=False, shuffle=True):
    dataset = FlowDataset(len_vocab, ipd_vocab, filename, window_size, args)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def test(model, data):
    test_loss = 0
    nsum = 0
    all_reconstruction_errors = []  
    all_labels = []  
    model.eval()
    with torch.no_grad():
        for x_batch, label_batch in data:
            x_batch = x_batch.to(model.device)
            label_batch = label_batch.to(model.device)
            nsum += label_batch.shape[0]
            reconstructed,ebd = model(x_batch)
            ebd = ebd[0].view(label_batch.shape[0],-1)
            reconstructed = reconstructed.view(label_batch.shape[0],-1)
            reconstruction_error = torch.mean(torch.abs(ebd - reconstructed), dim=1)
            test_loss += reconstruction_error.sum().item()
            reconstruction_error = reconstruction_error.view(label_batch.shape[0],-1)
            all_reconstruction_errors.append(reconstruction_error.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())
    test_loss /= nsum
    all_reconstruction_errors = np.concatenate(all_reconstruction_errors)
    all_labels = np.concatenate(all_labels)   
    auc = roc_auc_score(all_labels, all_reconstruction_errors)
    return auc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset
parser.add_argument("--len_vocab", default=1501)
parser.add_argument("--len_embedding_bits",default=10)
parser.add_argument("--ipd_vocab", type=int, default=2561)
parser.add_argument("--ipd_embedding_bits", type=int, default=8)
parser.add_argument("--ebdin", type=int, default=4)
parser.add_argument("--dataset", type=str)

parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--window_size", type=int, default=8)
parser.add_argument("--device",type=int)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--ptpth", type=str, default=None)
parser.add_argument("--trainpth",type=str)
parser.add_argument("--testpth",type=str)

args = parser.parse_args()




def run(args):
    basepth = args.testpth
    print(basepth)
    flood_filenames = {
        "test":"dataset/malicious_traffic/Flood_data.json",
    }
    cridex_filenames = {
        'test':"dataset/malicious_traffic/Cridex_data.json",
    }
    geodo_filenames = {
        'test':"dataset/malicious_traffic/Geodo_data.json",
    }
    htbot_filenames = {
        'test':"dataset/malicious_traffic/Htbot_data.json",
    }
    neris_filenames = {
        'test':"dataset/malicious_traffic/Neris_data.json",
    }
    virut_filenames = {
        'test':"dataset/malicious_traffic/Virut_data.json",
    }

    flood_filenames[args.dataset] = args.testpth
    cridex_filenames[args.dataset] = args.testpth
    geodo_filenames[args.dataset] = args.testpth
    htbot_filenames[args.dataset] = args.testpth
    neris_filenames[args.dataset] = args.testpth
    virut_filenames[args.dataset] = args.testpth
    test_path = args.testpth





    flood_loader = build_unified_data_loader(args.len_vocab, args.ipd_vocab, args.window_size, flood_filenames, 256, shuffle=False)
    cridex_loader = build_unified_data_loader(args.len_vocab, args.ipd_vocab, args.window_size, cridex_filenames, 256, shuffle=False)
    geodo_loader = build_unified_data_loader(args.len_vocab, args.ipd_vocab, args.window_size, geodo_filenames, 256, shuffle=False)
    htbot_loader = build_unified_data_loader(args.len_vocab, args.ipd_vocab, args.window_size, htbot_filenames, 256, shuffle=False)
    neris_loader = build_unified_data_loader(args.len_vocab, args.ipd_vocab, args.window_size, neris_filenames, 256, shuffle=False)
    virut_loader = build_unified_data_loader(args.len_vocab, args.ipd_vocab, args.window_size, virut_filenames, 256, shuffle=False)
    test_loader= build_data_loader(args.len_vocab, args.ipd_vocab, args.window_size, test_path, 256, args, is_train=True, shuffle=True)
    device = torch.device(f"cuda:{args.device}")

    model = table_splitautoencoder_template(8, 
                args.len_vocab,args.ipd_vocab,
                args.len_embedding_bits,args.ipd_embedding_bits,
                args.ebdin,64,48,24,12,
                device)

    state = torch.load(args.ptpth,weights_only=True, map_location=device)

    model.load_state_dict(state)

    htbot_auc  = test(model,htbot_loader)
    flood_auc  = test(model,flood_loader)
    cridex_auc = test(model,cridex_loader)
    virut_auc  = test(model,virut_loader)
    neris_auc  = test(model,neris_loader)
    geodo_auc  = test(model,geodo_loader)
    max_auc = (htbot_auc + flood_auc + cridex_auc + virut_auc + neris_auc + geodo_auc)/6
    num = 1
    print("avgauc:",max_auc)
    txt = "htbot_auc:{}\nflood_auc:{}\ncridex_auc:{}\nvirut_auc:{}\nneris_auc:{}\ngeodo_auc:{}\n".format(htbot_auc,flood_auc,cridex_auc,virut_auc,neris_auc,geodo_auc)
    print(txt)

if __name__ == "__main__":
    run(args)
