import torch
from util.newmlpdataloader import new_mlp_dataloader
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
    reset_int_triple_t,
    reset_norm_to_T_
)
from model.mlpmodel import (
    mlp_lut
)
from sklearn.metrics import f1_score


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

    newstate['MM1.S'] = state['MM1.S']
    newstate['MM1.H'] = state['MM1.H']
    newstate["MM1.LUT"] = state["MM1.LUT"]

    newstate['MM2.S'] = state['MM2.S']
    newstate['MM2.H'] = state['MM2.H']
    newstate["MM2.LUT"] = state["MM2.LUT"]

    newstate['MM3.S'] = state['MM3.S']
    newstate['MM3.H'] = state['MM3.H']

    norm0_weight, norm0_bias = adjust_norm_layer(state['norm0.weight'], state['norm0.bias'], state['norm0.running_var'], state['norm0.running_mean'])
    norm1_weight, norm1_bias = adjust_norm_layer(state['norm1.weight'], state['norm1.bias'], state['norm1.running_var'], state['norm1.running_mean'])
    norm2_weight, norm2_bias = adjust_norm_layer(state['norm2.weight'], state['norm2.bias'], state['norm2.running_var'], state['norm2.running_mean'])


    newstate["MM1.LUT"] = reset_fc(state['fc1.weight'], state['fc1.bias'], state['MM1.LUT'],device)
    newstate["MM2.LUT"] = reset_fc(state['fc2.weight'], state['fc2.bias'], state['MM2.LUT'],device)
    newstate["MM3.LUT"] = reset_fc(state['output.weight'], state['output.bias'], state['MM3.LUT'],device)

    newstate["MM1.LUT"] = reset_norm_to_LUT(norm1_weight, norm1_bias, newstate["MM1.LUT"],device)
    newstate["MM2.LUT"] = reset_norm_to_LUT(norm2_weight, norm2_bias, newstate["MM2.LUT"],device)
    newstate["MM2.T"] = reset_relu(state["MM2.T"])
    newstate["MM3.T"] = reset_relu(state["MM3.T"])

    newstate["MM1.T"] = reset_norm_to_T(state['MM1.T'], state['MM1.S'], norm0_weight, norm0_bias)


    sn1 = args.sn1
    sn2 = args.sn2
    sn3 = args.sn3
    newstate["MM1.T"] = reset_int(newstate["MM1.T"], 1)
    newstate["MM1.LUT"], newstate["MM2.T"] = reset_int_double(newstate["MM1.LUT"],newstate["MM2.T"],sn1)
    newstate["MM2.LUT"], newstate["MM3.T"] = reset_int_double(newstate["MM2.LUT"],newstate["MM3.T"],sn2)
    newstate["MM3.LUT"] = reset_int_single(newstate["MM3.LUT"],sn3)



    if args.dataset == "ISCXVPN":
        num_classes = 6
    else:
        num_classes = 3
    model = mlp_lut(10, 32, 16, num_classes, device,1,2,1)
    
    model.load_state_dict(newstate)

    test_loader = new_mlp_dataloader(args.testpth, "mlp", args, batch_size=10000, shuffle=False, istrained = False)
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

"""
python mlp_convert.py --dataset CICIOT2022 \
    --testpth /mnt/sdc_space/oldz/Pegasus_open/dataset/CICIOT2022/CICIOT2022_test.csv \
    --ptpth /mnt/sdc_space/oldz/zlx/testquantimodel/usedpt/MLP/CICIOT2022/mlp4_32_16_best_0.8416400020134578f_0.810935489112734cf.pt

"""
if __name__ == "__main__":
    f1_macro = main(args)
    print(f"F1 Macro: {f1_macro}")