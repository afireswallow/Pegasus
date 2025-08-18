import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
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
from model.bigcnnmodel import(
    bigcnn_lut,
    bigcnn_template
)

from util.bigcnn_data_loader import (
    bigcnnFlowDataset
)


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
parser.add_argument("--len_vocab", default=1501)
parser.add_argument("--len_embedding_bits",default=10)
parser.add_argument("--ipd_vocab", type=int, default=2561)
parser.add_argument("--ipd_embedding_bits", type=int, default=8)
parser.add_argument("--ptpth1", type=str,required=True)
parser.add_argument("--ptpth2", type=str,required=True)
parser.add_argument("--dataset", type=str,required=True)
parser.add_argument("--testpth", type=str,required=True)
parser.add_argument("--savepth", type=str)
parser.add_argument("--model_name", type=str,default="bigcnn")
parser.add_argument("--sn1", type=int, default=12)
parser.add_argument("--sn2", type=int, default=12)
args = parser.parse_args()




def model_ouput(model,groups):
    groups = groups.view(-1,3,16,2)
    finalresult = []
    g = 0
    for group in groups:
        group_combos = []
        for i in range(16):  
            for j in range(16):  
                for k in range(16): 
                    combo = torch.stack([group[0, i], group[1, j], group[2, k]]) 
                    combo = combo.view(1,-1)
                    o = torch.zeros(1,60)
                    #print(o[:,seg_length * g : seg_length * (g + 1)].shape)
                    #print(combo.shape)
                    if len(groups) == 1:
                        o[:,54:] = combo
                    else:
                        o[:,6 * g : 6 * (g + 1)] = combo
                    #o[:,54:] = combo
                    result = model.forward(o)
                    result = result.view(-1)
                    group_combos.append(result)
        group_combos = torch.stack(group_combos)
        finalresult.append(group_combos)
        g+=1
    finalresult = torch.stack(finalresult)
    return finalresult


def get_seg_lut(weights1,weights2,model):
    #print(weights1.keys())
    x = weights1['MM1.LUT']

    del weights2['fc3.weight']
    del weights2['fc3.bias']
    del weights2['MM2.S']
    del weights2['MM2.H']
    del weights2['MM2.T']
    del weights2['MM2.LUT']

    weights1['fc5.weight']  =weights2['fc1.weight']
    weights1['fc5.bias']    =weights2['fc1.bias']
    weights1['fc6.weight']  =weights2['fc2.weight']
    weights1['fc6.bias']    =weights2['fc2.bias']

    del weights2['fc1.weight']
    del weights2['fc1.bias']
    del weights2['fc2.weight']
    del weights2['fc2.bias']

    weights1.update(weights2)
    del weights1['MM1.T']
    del weights1['MM1.LUT']
    del weights1['MM1.S']
    del weights1['MM1.H']
    
    groups = x.view(10,3,16,2)
    groups1 = groups[-1,:,:,:]
    model.load_state_dict(weights1)
    lut_1 = model_ouput(model,groups1)

    weights1['norm1.bias'] = torch.zeros_like(weights1['norm1.bias'])
    weights1['norm2.bias'] = torch.zeros_like(weights1['norm2.bias'])
    weights1['norm3.bias'] = torch.zeros_like(weights1['norm3.bias'])
    weights1['norm4.bias'] = torch.zeros_like(weights1['norm4.bias'])
    weights1['fc1.bias'] = torch.zeros_like(weights1['fc1.bias'])
    weights1['fc2.bias'] = torch.zeros_like(weights1['fc2.bias'])
    weights1['fc3.bias'] = torch.zeros_like(weights1['fc3.bias'])
    weights1['fc4.bias'] = torch.zeros_like(weights1['fc4.bias'])
    weights1['norm1.running_mean'] = torch.zeros_like(weights1['norm1.running_mean'])
    weights1['norm2.running_mean'] = torch.zeros_like(weights1['norm2.running_mean'])
    weights1['norm3.running_mean'] = torch.zeros_like(weights1['norm3.running_mean'])
    weights1['norm4.running_mean'] = torch.zeros_like(weights1['norm4.running_mean'])

    
    model.load_state_dict(weights1)
    lut_2 = model_ouput(model,groups)
    lut_2[-1,:,:] = lut_1
    return lut_2
    
    







def main(args):
    if args.dataset == "ISCXVPN":
        num_classes = 6
    else:
        num_classes = 3

    device = torch.device('cpu')
    state1 = torch.load(args.ptpth1,map_location=device)
    state2 = torch.load(args.ptpth2,map_location=device)
    tempstate1 = state1.copy()
    tempstate2 = state2.copy()
    newstate = {}
    model = bigcnn_template(num_classes,device,6,16)
    model.eval()
    newstate['MM1.LUT'] = get_seg_lut(tempstate1,tempstate2,model)
    newstate['MM1.H'] = torch.load("convert/cnnl_new_H.pt",weights_only=True)
    newstate['MM1.S'] = state1['MM1.S']
    newstate['MM1.T'] = state1['MM1.T']

    newstate['MM2.H']   = state2['MM2.H']
    newstate['MM2.T']   = state2['MM2.T'].expand(8, -1)           # [8, 15]
    newstate['MM2.LUT'] = state2['MM2.LUT'].expand(8, -1, -1)     # [8, 16, 3]
    newstate['MM2.S']   = state2['MM2.S'].expand(8, -1, -1)
    newstate["MM2.LUT"] = reset_fc(state2["fc3.weight"],state2["fc3.bias"],newstate["MM2.LUT"],device)



    sn1 = args.sn1
    sn2 = args.sn2
    newstate['MM1.T'] = reset_int(newstate['MM1.T'],1)
    newstate["MM1.LUT"], newstate["MM2.T"] = reset_int_double(newstate["MM1.LUT"], newstate["MM2.T"],sn1)
    maxx = newstate["MM2.LUT"].max()
    boundary = (2 ** sn2) - 1
    scalar = boundary/maxx
    newstate['MM2.LUT'] = reset_int(newstate['MM2.LUT'],scalar)


    model = bigcnn_lut(num_classes,device,6,16)
    model.load_state_dict(newstate)

    test_loss = 0
    correct = 0
    model.eval()
    nsum = 0
    all_preds = []
    all_labels = []


    model.eval() 
    test_loader = build_data_loader(1501, 2561, 8, args.testpth, 1000, args, is_train=False, shuffle=False)
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
