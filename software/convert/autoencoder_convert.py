import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from util.reset import(
    reset_fc,
    reset_relu,
    adjust_norm_layer,
    reset_norm_to_T,
    reset_norm_to_LUT,
    reset_int,
    reset_int_double,
    reset_int_single,
    reset_embeding_LUT,
    reset_int_triple_l,
    reset_conv_LUT,
    kernel_to_toeplitz
)
from sklearn.metrics import f1_score
from model.splitAutoencoder import (
    MM_splitautoencoder_template,
    table_splitautoencoder_template
)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,confusion_matrix



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--len_vocab", default=1501)
parser.add_argument("--len_embedding_bits",default=10)
parser.add_argument("--ipd_vocab", type=int, default=2561)
parser.add_argument("--ipd_embedding_bits", type=int, default=8)
parser.add_argument("--ptpth", type=str,required=True)
parser.add_argument("--dataset", type=str,required=True)
parser.add_argument("--savepth", type=str)
parser.add_argument("--sn1", type=int, default=12)
parser.add_argument("--sn2", type=int, default=12)
parser.add_argument("--sn3", type=int, default=12)
args = parser.parse_args()


def get_seg_lut1(model,x):
    lengths = [6, 6, 6, 6, 6, 2]
    start_indices = [0, 6, 12, 18, 24, 30]
    result = []
    seg_length = 6 
    numsplit = 32 // seg_length
    newlut = []
    for n in range(5):
        group = x[n*3:n*3+3,:,:]
        group_combos = []
        for i in range(16):  
            for j in range(16):  
                for k in range(16):  
                    combo = torch.stack([group[0, i], group[1, j], group[2, k]])  # [3, 2]
                    combo = combo.view(-1)
                    o = torch.zeros(1,32)
                    o[:,seg_length * n: seg_length * (n + 1)] = combo
                    result = model.forward(o)
                    result = result.view(-1)
                    group_combos.append(result)  
        group_combos = torch.stack(group_combos)  # [4096, 2]
        newlut.append(group_combos)
    newlut = torch.stack(newlut)
    return newlut


def get_seg_lut2(model,x):
    lengths = [6, 6, 6, 6, 6, 2]
    start_indices = [0, 6, 12, 18, 24, 30]
    result = []
    seg_length = 6 
    numsplit = 32 // seg_length
    newlut = []
    group = x[15:16,:,:]
    group_combos = []
    for i in range(16):  
        combo = group[:,i,:]
        combo = combo.view(-1)
        #print(combo.shape)
        o = torch.zeros(1,32)
        o[:,30:32] = combo
        result = model.forward(o)
        result = result.view(-1)
        group_combos.append(result) 
    group_combos = torch.stack(group_combos)
    return group_combos


def main(args):
    device = torch.device('cpu')
    state = torch.load(args.ptpth,map_location=device)
    newstate = {}
    if args.dataset == "ISCXVPN":
        num_classes = 6
    else:
        num_classes = 3
    model = MM_splitautoencoder_template(8,1501,2561,10,8,4,64,48,24,12,device)
    model.load_state_dict(state,strict=False)
    lut1 = get_seg_lut1(model,state['MM.LUT'])
    lut2 = get_seg_lut2(model,state['MM.LUT'])
    newstate["MM_1.LUT"] = lut1
    newstate["MM_2.LUT"] = lut2.view(1,16,32)

    newstate["lenebdLUT"],newstate["ipdebdLUT"] = reset_embeding_LUT(state["len_embedding.weight"],state["ipd_embedding.weight"],state["fc1.weight"],state["fc1.bias"],device)
    sn1=args.sn1
    newstate["lenebdLUT"], newstate["ipdebdLUT"],newT = reset_int_triple_l(newstate["lenebdLUT"], newstate["ipdebdLUT"],state["MM.T"],device,sn1)
    newstate['MM_1.S'] = state['MM.S'][0:15,:,:].view(-1,2,15)
    newstate['MM_1.H'] = torch.load("convert/cnnl_new_H.pt",weights_only=True) 
    newstate['MM_1.T'] = newT[0:15,:].view(-1,15)
    newstate['MM_2.S'] = state['MM.S'][-1,:,:].view(1,2,15)
    newstate['MM_2.H'] = state['MM.H']
    newstate['MM_2.T'] = newT[-1,:].view(1,15)

    maxx = max(torch.max(torch.abs(newstate['MM_1.LUT'])),torch.max(torch.abs(newstate['MM_1.LUT'])))
    sn2=args.sn2
    boundary = (2 ** sn2) - 1
    scalar = 1
    while maxx * scalar * 2 < boundary:
        scalar = scalar * 2
    newstate['MM_1.LUT'] = reset_int(newstate['MM_1.LUT'], scalar)
    newstate['MM_2.LUT'] = reset_int(newstate['MM_2.LUT'], scalar)

    model = table_splitautoencoder_template(8,1501,2561,10,8,4,64,48,24,12,device)
    model.load_state_dict(newstate)

    

    if  args.savepth is not None:
        torch.save(newstate, args.savepth)


if __name__ == "__main__":
    main(args)