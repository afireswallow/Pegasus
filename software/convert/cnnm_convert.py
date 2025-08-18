import torch
from util.testbrnndataloader import(
    FlowDataset
)
import argparse
import numpy as np
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
from model.cnnmmodel import (
    enlargesegTextCNN_LUT,
    enlargesegTextCNN2_getLUT
)

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
args = parser.parse_args()




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

def get_seg_lut(weight,model):
    lutdata = weight['MM1.LUT']
    fc = weight['fc2.weight']
    bias = weight['fc2.bias']
    del weight['len_embedding.weight']
    del weight['ipd_embedding.weight']
    del weight['fc1.weight']
    del weight['fc1.bias']
    del weight['fc2.weight']
    del weight['fc2.bias']
    del weight['MM1.T']
    del weight['MM1.LUT']
    del weight['MM1.S']
    del weight['MM1.H']

    model.load_state_dict(weight)
    groups = lutdata.view(8,2,16,2)
    g=0
    newlut = []
    for group in groups:
        group_combos = []  # 用于存储这一组的所有组合
        # 2层循环遍历所有 16^2 组合
        for i in range(16):  
            for j in range(16):  
                combo = torch.stack([group[0, i], group[1, j]])
                combo = combo.view(1,-1)
                o = torch.zeros(1,32)
                o[:,4*g : 4*(g + 1)] = combo
                result1,result2,result3 = model.forward(o)
                result1 = result1.view(-1)
                result2 = result2.view(-1)
                result3 = result3.view(-1)

                #print(result1.shape)
                #print(fc[:, 0:6*32].T.shape)
                result1 = torch.einsum('d, do->o', result1, fc[:, 0:6*32].T)
                result2 = torch.einsum('d, do->o', result2, fc[:, 6*32:11*32].T)
                result3 = torch.einsum('d, do->o', result3, fc[:, 11*32:15*32].T)
                if g == 7:
                    result3 += bias
                result = result1+result2+result3
                #result = torch.stack(result, dim=0) #[3,480]
                group_combos.append(result) 
        g+=1
        group_combos = torch.stack(group_combos, dim=0)  # [256,3,480]
        newlut.append(group_combos)

    newlut = torch.stack(newlut, dim=0) 
    return newlut



def main(args):
    if args.dataset == "ISCXVPN":
        num_classes = 6
    else:
        num_classes = 3

    device = torch.device('cpu')
    state = torch.load(args.ptpth,map_location=device)
    tempstate = state.copy()
    newstate = {}

    model = enlargesegTextCNN2_getLUT(8, 64,num_classes,1501,2561,10,8,32,4,2,device)

    newstate['MM1.LUT'] = get_seg_lut(tempstate,model)
    newstate['MM1.H'] = torch.load("convert/cnnm_new_H.pt", map_location='cpu', weights_only=True)
    newstate['MM1.T'] = state['MM1.T']
    newstate['MM1.S'] = state['MM1.S']
    newstate["lenebdLUT"],newstate["ipdebdLUT"] = reset_embeding_LUT(state["len_embedding.weight"],state["ipd_embedding.weight"],state["fc1.weight"],state["fc1.bias"],device)
    

    sn1 = args.sn1
    sn2 = args.sn2
    newstate['lenebdLUT'] ,newstate['ipdebdLUT'], newstate['MM1.T'] = reset_int_triple_l(newstate['lenebdLUT'] ,newstate['ipdebdLUT'], newstate['MM1.T'],device,sn1)
    newstate['MM1.LUT'] = reset_int_single(newstate['MM1.LUT'], sn2)


    model = enlargesegTextCNN_LUT(8, 64, num_classes, 1501, 2561, 10, 8, 32, 4, 2, device)
    model.load_state_dict(newstate)

    test_loss = 0
    correct = 0
    model.eval()
    nsum = 0
    all_preds = []
    all_labels = []

    test_loader = build_data_loader(1501, 2561, 8, args.testpth, 1000, args, is_train=False, shuffle=False)
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