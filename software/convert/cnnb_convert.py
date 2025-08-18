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
    kernel_to_toeplitz
)
from sklearn.metrics import f1_score
from model.cnnmodel import (
    cnn_lut
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
    #newstate = state
    newstate["lenebdLUT"],newstate["ipdebdLUT"] = reset_embeding_LUT(state["len_embedding.weight"],state["ipd_embedding.weight"],state["fc1.weight"],state["fc1.bias"],device)

    newstate["MM2.LUT"] = reset_fc(state['fc2.weight'], state['fc2.bias'], state['MM2.LUT'],device)
    newstate["MM2.T"] = reset_relu(state["MM2.T"])
    input_shape = [1,8,4]
    conv3lut = reset_conv_LUT(state['MM1.LUT'],state["conv3.weight"], input_shape, state["conv3.bias"], device)
    newstate["convMM3.T"] = state['MM1.T']
    newstate["convMM3.S"] = state['MM1.S']
    newstate["convMM3.H"] = state['MM1.H']
    newstate["convMM3.LUT.0"] = conv3lut[0]
    newstate["convMM3.LUT.1"] = conv3lut[1]

    conv4lut = reset_conv_LUT(state['MM1.LUT'],state["conv4.weight"], input_shape, state["conv4.bias"], device)
    newstate["convMM4.T"] = state['MM1.T']
    newstate["convMM4.S"] = state['MM1.S']
    newstate["convMM4.H"] = state['MM1.H']
    newstate["convMM4.LUT.0"] = conv4lut[0]
    newstate["convMM4.LUT.1"] = conv4lut[1]

    conv5lut = reset_conv_LUT(state['MM1.LUT'],state["conv5.weight"], input_shape, state["conv5.bias"], device)
    newstate["convMM5.T"] = state['MM1.T']
    newstate["convMM5.S"] = state['MM1.S']
    newstate["convMM5.H"] = state['MM1.H']
    newstate["convMM5.LUT.0"] = conv5lut[0]
    newstate["convMM5.LUT.1"] = conv5lut[1]

    newstate["MM2.S"] = state["MM2.S"]
    newstate["MM2.H"] = state["MM2.H"]




    sn1=args.sn1
    sn2=args.sn2
    sn3=args.sn3
    
    
    
    newstate["lenebdLUT"], newstate["ipdebdLUT"],newT1 = reset_int_triple_l(newstate["lenebdLUT"], newstate["ipdebdLUT"],state["MM1.T"],device,sn1)
    newstate["convMM3.T"] = newT1
    newstate["convMM4.T"] = newT1
    newstate["convMM5.T"] = newT1

    


    tensors = [
        newstate["convMM3.LUT.0"],
        newstate["convMM3.LUT.1"],
        newstate["convMM4.LUT.0"],
        newstate["convMM4.LUT.1"],
        newstate["convMM5.LUT.0"],
        newstate["convMM5.LUT.1"]
    ]
    max_values = [torch.max(torch.abs(tensor)) for tensor in tensors]
    max_values.append(torch.max(newstate["MM2.T"]))
    maxx = torch.max(torch.tensor(max_values))
    boundary = (2 ** sn2) - 1
    scalar = boundary / maxx
    
    newstate["convMM3.LUT.0"] = reset_int(newstate["convMM3.LUT.0"], scalar)
    newstate["convMM3.LUT.1"] = reset_int(newstate["convMM3.LUT.1"], scalar)
    newstate["convMM4.LUT.0"] = reset_int(newstate["convMM4.LUT.0"], scalar)
    newstate["convMM4.LUT.1"] = reset_int(newstate["convMM4.LUT.1"], scalar)
    newstate["convMM5.LUT.0"] = reset_int(newstate["convMM5.LUT.0"], scalar)
    newstate["convMM5.LUT.1"] = reset_int(newstate["convMM5.LUT.1"], scalar)
    newstate["MM2.T"] = reset_int(newstate["MM2.T"], scalar)

    newstate["MM2.LUT"] = reset_int_single(newstate["MM2.LUT"],sn3)



    if args.dataset == "ISCXVPN":
        num_classes = 6
    else:
        num_classes = 3
    model =cnn_lut(8, num_classes,
                 args.len_vocab,args.ipd_vocab,
                 args.len_embedding_bits,args.ipd_embedding_bits,
                 2, 4, device, 2, 2)
    
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