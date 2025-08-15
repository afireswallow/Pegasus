import os
import time
import json
import argparse
import torch
from util.seed import set_seed
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from util.early_stopping import *
from util.quanti import model_quanti
from util.getdata import get_data
import os
from sklearn.metrics import f1_score
import datetime
import copy
from sklearn.metrics import precision_score, recall_score
EARLY_STOP = 1
BEST_SCORE_UPDATED = 2

def train(optimizer, model, data):
    model.train()
    n =0
    for x_batch, label_batch in data:
        x_batch = x_batch
        label_batch = label_batch.to(model.device)
        
        output, quantidata = model(x_batch)
        
        loss = F.nll_loss(output, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n+=1

def test(model, data):
    test_loss = 0
    correct = 0
    model.eval()
    nsum = 0
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x_batch, label_batch in data:
            x_batch = x_batch
            label_batch = label_batch.to(model.device)
            nsum += label_batch.shape[0]
            
            output, quantidata = model(x_batch)
            test_loss += F.nll_loss(output, label_batch, reduction="sum")
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct += pred.eq(label_batch.data).sum()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())

        test_loss = test_loss / nsum
        all_preds = np.concatenate(all_preds)  
        all_labels = np.concatenate(all_labels)  
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        return 100. * correct / nsum, test_loss, f1_macro, precision_macro, recall_macro


def quanti_sequence(
                    modelname,
                    modellist, 
                    train_loader,
                    val_loader,
                    known_loader,
                    test_loader,
                    device, patience, 
                    dlist, savepth, 
                    lr, step,args,
                    istrained=True,
                    taskname='NONE'):

    os.makedirs(str(savepth)+f'/weight', exist_ok=True)
    
    weights = []
    num = 1
    print(
        '模型量化阶段数量:{}|lr:{}|step:{}|patience:{}|D:{}'.format(
            len(modellist),lr,step,patience,''.join(map(str, dlist))
        ),
        flush=True
    )
    for model in modellist:
        early_stop = EarlyStopping(patience=patience, delta=0, verbose=False)
        acc = 0
        f1 = 0
        epoch = 0
        while True:
            #print(model.state_dict().keys())
            if istrained:
                break
            #optimizer = optim.AdamW(model.parameters(),lr=lr)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            
            if epoch != 0:
                train(optimizer,model,train_loader)
            epoch+=1
            tempacc = 0
            tempacc, loss, f1_macro, _, _ = test(model,val_loader)
            status = early_stop(loss)
            #print(status)
            print('modelnum:{}{}|epoch:{}|acc:{:.4f}|loss:{:.4f}|status:{}|f1:{:.4f}|flagacc{:.4f}|flagf1{:.4f}'.format(
                modelname,args.modelnum+num-1,epoch,tempacc,loss,status,f1_macro,acc,f1
            ),
            flush=True
            )

            if status == EARLY_STOP:
                break

            if f1_macro>f1: 
                weights.append(copy.deepcopy(model.state_dict()))
                acc = tempacc
                f1 = f1_macro
                torch.save(weights[-1], f"{savepth}/weight/{modelname}{num+args.modelnum-1}_temp.pt")
            
            
            if epoch%step==0 :
                lr /= 2
                optimizer = optim.AdamW(model.parameters(),lr=lr)
            if epoch%args.confirmterm == 0 :
                acc = tempacc
                confirm_acc, loss, confirm_f1_macro, _, _ = test(model,test_loader)
                print("f1",f1_macro,"confirm f1:",confirm_f1_macro)
                torch.save(model.state_dict(), f"{savepth}/weight/{modelname}{num+args.modelnum-1}_{epoch}epoch_{f1_macro}f_{confirm_f1_macro}cf.pt")




        if istrained:
            weights.append(model.state_dict())
            istrained = False
        else:
            model.load_state_dict(weights[-1])
            tempacc, loss, f1, _, _ = test(model,val_loader)
            tempacc, loss, confirm_f1_macro,confirm_pr_macro,confirm_re_macro = test(model,test_loader)
            print("best f1",f1,"confirm acc:",tempacc,"confirm f1:",confirm_f1_macro,"confirm pr:",confirm_pr_macro,"confirm re:",confirm_re_macro)
            torch.save(weights[-1], f"{savepth}/weight/{modelname}{num+args.modelnum-1}_best_{f1}f_{confirm_f1_macro}cf.pt")
        weight = weights[-1]
        if num != len(modellist):
            train_data = get_data(model, known_loader, weight)
            pth = savepth
            if len(train_data) == 1:
                if args.model!= "bigcnn" or args.modelnum==1:
                    S, T, LUT = model_quanti(train_data[0],dlist[args.modelnum+num-2],args.modelnum+num-1,pth)
                    weight[f'MM{args.modelnum+num-1}.S'] = S
                    weight[f'MM{args.modelnum+num-1}.LUT'] = LUT
                    weight[f'MM{args.modelnum+num-1}.T'] = T
                    weight[f'MM{args.modelnum+num-1}.H'] = torch.load('./util/H.pth', weights_only=True)
                else:
                    S, T, LUT = model_quanti(train_data[0],dlist[args.modelnum+num-3],args.modelnum+num-2,pth)
                    weight[f'MM{args.modelnum+num-2}.S'] = S
                    weight[f'MM{args.modelnum+num-2}.LUT'] = LUT
                    weight[f'MM{args.modelnum+num-2}.T'] = T
                    weight[f'MM{args.modelnum+num-2}.H'] = torch.load('./util/H.pth', weights_only=True)
                
            else:
                for i in range(len(train_data)):
                    S, T, LUT = model_quanti(train_data[i],dlist[args.modelnum+num-2],args.modelnum+num-1,pth)
                    weight[f'MM{num+args.modelnum-1}_{i+1}.S'] = S
                    weight[f'MM{num+args.modelnum-1}_{i+1}.LUT'] = LUT
                    weight[f'MM{num+args.modelnum-1}_{i+1}.T'] = T
                    weight[f'MM{num+args.modelnum-1}_{i+1}.H'] = torch.load('./util/H.pth')
            nextmodel = modellist[num]
            nextmodel.load_state_dict(weight)
            num+=1



        