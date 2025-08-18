import torch
from util.testbrnndataloader import(
    FlowDataset
)
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
    kernel_to_toeplitz,
    reset_rnn_threshold,
    reset_int_triple_t
)
from sklearn.metrics import f1_score
from model.rnnmodel import (
    rnn_lut
)

def build_data_loader(len_vocab, ipd_vocab, window_size, filename, batch_size, args, is_train=False, shuffle=True):
    dataset = FlowDataset(len_vocab, ipd_vocab, filename, window_size, args)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--len_vocab", default=1501)
parser.add_argument("--len_embedding_bits",default=10)
parser.add_argument("--ipd_vocab", type=int, default=2561)
parser.add_argument("--ipd_embedding_bits", type=int, default=8)
parser.add_argument("--ptpth", type=str,required=True)
parser.add_argument("--dataset", type=str,required=True)
parser.add_argument("--testpth", type=str,required=True)
parser.add_argument("--savepth", type=str)
parser.add_argument("--sn1", type=int, default=12)
parser.add_argument("--sn2", type=int, default=12)
parser.add_argument("--sn3", type=int, default=12)
args = parser.parse_args()

def main(args):
    device = torch.device('cpu')
    state = torch.load(args.ptpth,map_location=device)
    newstate = {}
    newstate["lenebdLUT"],newstate["ipdebdLUT"] = reset_embeding_LUT(state["len_embedding.weight"],state["ipd_embedding.weight"],state["fc1.weight"],state["fc1.bias"],device)
    LUT = state['MM1.LUT']
    LUT1 = LUT[:8,:,:]
    LUT2 = LUT[8:,:,:]
    x2hweight = state['rnn.x2h.weight']
    h2hweight = state['rnn.h2h.weight']
    x2hbias = state['rnn.x2h.bias']
    h2hbias = state['rnn.h2h.bias']
    LUT1_new = reset_fc(x2hweight,x2hbias,LUT1,device)
    LUT2_new = reset_fc(h2hweight,h2hbias,LUT2,device)
    LUT_new = torch.cat((LUT1_new,LUT2_new),dim=0).to(device)
    h_T = state['MM1.T'][8:,:]
    x_T = state['MM1.T'][:8,:]
    h_T_new = reset_rnn_threshold(h_T,device)
    T_new = torch.cat((x_T, h_T_new), dim=0)
    newstate['MM1.T'] = T_new
    newstate['MM1.LUT'] = LUT_new
    newstate['MM2.T'] = reset_rnn_threshold(state['MM2.T'],device)
    newstate['MM2.LUT'] = reset_fc(state['fc2.weight'], state['fc2.bias'], state['MM2.LUT'],device)

    newstate['MM1.S'] = state['MM1.S']
    newstate['MM1.H'] = state['MM1.H']
    newstate['MM2.S'] = state['MM2.S']
    newstate['MM2.H'] = state['MM2.H']


    sn1 = args.sn1
    sn2 = args.sn2
    sn3 = args.sn3
    newstate["lenebdLUT"],newstate["ipdebdLUT"],newstate['MM1.T'][:8,:]=reset_int_triple_l(newstate["lenebdLUT"],newstate["ipdebdLUT"],newstate['MM1.T'][:8,:],device,sn1)
    newstate['MM1.LUT'],newstate['MM1.T'][8:,:], newstate['MM2.T']= reset_int_triple_t(newstate['MM1.LUT'],newstate['MM1.T'][8:,:], newstate['MM2.T'],device,sn2)
    newstate['MM2.LUT'] = reset_int_single(newstate['MM2.LUT'],sn3)

    if args.dataset == "ISCXVPN":
        num_classes = 6
    else:
        num_classes = 3
    model =rnn_lut(8, 8,
                 num_classes,
                 args.len_vocab, args.ipd_vocab,
                 args.len_embedding_bits, args.ipd_embedding_bits,
                 device, 0,
                 1,1)
    
    model.load_state_dict(newstate)

    test_loader = build_data_loader(1501, 2561, 8, args.testpth, 1000, args, is_train=False, shuffle=False)
    test_loss = 0
    correct = 0
    model.eval()
    nsum = 0
    all_preds = []
    all_labels = []


    model.eval() 
    with torch.no_grad():
        for x_batch, label_batch in test_loader:
            x_batch = x_batch
            #print(label_batch)
            #torch.load("exe")
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

    if  args.savepth is not None:
        torch.save(newstate, args.savepth)
    return f1_macro


if __name__ == "__main__":
    f1_macro = main(args)
    print(f"F1 Macro: {f1_macro}")