
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
from torch.autograd import Variable
from util import bnn_modules


def dropout_layer(X, dropout, device):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X).to(device)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape).to(device) > dropout).float()
    return mask * X / (1.0 - dropout)






class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, device, bias=True):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, hidden_size, bias=bias).to(device)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))
        xh = self.x2h(x) + self.h2h(hidden)
        hy = torch.tanh(xh)
        return hy




class MM(nn.Module):
    def __init__(self, C, D, device):
        super(MM, self).__init__()
        self.register_buffer('S', torch.randn(C, D, 15).to(device))
        self.register_buffer('H', torch.randn(15, 16).to(device))
        # number of codeblocks that input is divided into,
        # for example, if input size is 14, and C is 7, then input vector is divided into 7 codeblocks, each of size 2.
        self.C = C
        # size of each codeblock
        self.D = D
        # thresholds table.
        self.T = nn.Parameter(torch.randn(C, 15).to(device))
        self.LUT = nn.Parameter(torch.randn(C, 16, D).to(device))

    def forward(self, x):
        x = x.view(-1, self.C, self.D)
        #print(x.shape)
        x = torch.einsum('bcd,cdk->bck', x, self.S)
        x = x - self.T - 0.0001
        tanh = torch.tanh(x)
        sign = torch.sign(x)
        x = (sign - tanh).detach() + tanh
        x = torch.einsum('bcd,dk->bck', x, self.H)
        one_hot = F.one_hot(torch.argmax(x, dim=-1), num_classes=16).float()
        temperature = 1
        softmax = torch.softmax(x / temperature, dim=-1)
        x_one_hot = torch.einsum('bck,ckd->bcd', one_hot, self.LUT)
        x_softmax = torch.einsum('bck,ckd->bcd', softmax, self.LUT)
        x = (x_one_hot - x_softmax).detach() + x_softmax

        x = x.reshape(-1, self.C * self.D)
        return x







class RNN1(nn.Module):
    def __init__(self, 
                 rnn_in, hidden_size,
                 labels_num,
                 len_vocab, ipd_vocab,
                 len_embedding_bits, ipd_embedding_bits,
                 device, droprate):
        
        super(RNN1, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        self.dropout = droprate
        self.rnn_in = rnn_in
        self.labels_num = labels_num
        self.len_vocab = len_vocab
        self.ipd_vocab = ipd_vocab
        self.len_embedding_bits = len_embedding_bits
        self.ipd_embedding_bits = ipd_embedding_bits
        
        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits).to(device)
        
        self.rnn = RNNCell(self.rnn_in, self.hidden_size, self.device)
        self.fc1 = nn.Linear(self.len_embedding_bits+self.ipd_embedding_bits, self.rnn_in).to(device)
        self.fc2 = nn.Linear(self.hidden_size, self.labels_num).to(device)

    def forward(self, x):
        len_x = x[:,:,0].to(self.device)
        ipd_x = x[:,:,1].to(self.device)
        len_x = len_x.to(self.device)
        ipd_x = ipd_x.to(self.device)

        len_x = self.len_embedding(len_x)
        ipd_x = self.ipd_embedding(ipd_x)
        x = torch.cat((len_x, ipd_x), dim=-1)
        x = self.fc1(x)
        batch_size = x.shape[0]
        x = x.permute(1, 0, 2)
        h_0 = self.init_hidden_state(x.size(1)) 
        outs = []
        res = []
        for i in range(x.size(0)):
            res.append(torch.cat((x[i,:,:], h_0), dim=1))
            h_0 = self.rnn(x[i,:,:], h_0)
            outs.append(h_0)
        res = torch.cat(res, dim=0)
        out = outs[-1].squeeze()    
        if self.training:
            out = dropout_layer(out,self.dropout, self.device)     
        out = self.fc2(out)
        return F.log_softmax(out, dim=-1),[res]
 
    def init_hidden_state(self, batch_size):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return h_0








class RNN2(nn.Module):
    def __init__(self, 
                 rnn_in, hidden_size,
                 labels_num,
                 len_vocab, ipd_vocab,
                 len_embedding_bits, ipd_embedding_bits,
                 device, droprate,
                 D1):
        
        super(RNN2, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        self.dropout = droprate
        self.rnn_in = rnn_in
        self.labels_num = labels_num
        self.len_vocab = len_vocab
        self.ipd_vocab = ipd_vocab
        self.len_embedding_bits = len_embedding_bits
        self.ipd_embedding_bits = ipd_embedding_bits
        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits).to(device)
        
        self.rnn = RNNCell(self.rnn_in, self.hidden_size, self.device)
        self.fc1 = nn.Linear(self.len_embedding_bits+self.ipd_embedding_bits, self.rnn_in).to(device)
        self.fc2 = nn.Linear(self.hidden_size, self.labels_num).to(device)
        self.MM1 = MM( (rnn_in+hidden_size)//D1, D1,device)

    def forward(self, x):
        len_x = x[:,:,0].to(self.device)
        ipd_x = x[:,:,1].to(self.device)
        len_x = len_x.to(self.device)
        ipd_x = ipd_x.to(self.device)

        len_x = self.len_embedding(len_x)
        ipd_x = self.ipd_embedding(ipd_x)
        x = torch.cat((len_x, ipd_x), dim=-1)
        x = self.fc1(x)
        batch_size = x.shape[0]
        x = x.permute(1, 0, 2)
        h_0 = self.init_hidden_state(x.size(1)) 
        outs = []
        for i in range(x.size(0)):
            tensor = torch.cat((x[i,:,:], h_0), dim=1)
            tensor = self.MM1(tensor)
            x_ = tensor[:,:self.rnn_in]
            h_ = tensor[:,self.rnn_in:]
            h_0 = self.rnn(x_, h_)
            outs.append(h_0)
        out = outs[-1].squeeze()    
        if self.training:
            out = dropout_layer(out,self.dropout, self.device)   
        res = [out]  
        out = self.fc2(out)
        return F.log_softmax(out, dim=-1),res
 
    def init_hidden_state(self, batch_size):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return h_0





class RNN3(nn.Module):
    def __init__(self, 
                 rnn_in, hidden_size,
                 labels_num,
                 len_vocab, ipd_vocab,
                 len_embedding_bits, ipd_embedding_bits,
                 device, droprate,
                 D1,D2):
        
        super(RNN3, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        self.dropout = droprate
        self.rnn_in = rnn_in
        self.labels_num = labels_num
        self.len_vocab = len_vocab
        self.ipd_vocab = ipd_vocab
        self.len_embedding_bits = len_embedding_bits
        self.ipd_embedding_bits = ipd_embedding_bits
        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits).to(device)
        
        self.rnn = RNNCell(self.rnn_in, self.hidden_size, self.device)
        self.fc1 = nn.Linear(self.len_embedding_bits+self.ipd_embedding_bits, self.rnn_in).to(device)
        self.fc2 = nn.Linear(self.hidden_size, self.labels_num).to(device)
        self.MM1 =MM( (rnn_in+hidden_size)//D1, D1,device)
        self.MM2 = MM(hidden_size//D2,D2,device)

    def forward(self, x):
        len_x = x[:,:,0].to(self.device)
        ipd_x = x[:,:,1].to(self.device)
        len_x = len_x.to(self.device)
        ipd_x = ipd_x.to(self.device)

        len_x = self.len_embedding(len_x)
        ipd_x = self.ipd_embedding(ipd_x)
        x = torch.cat((len_x, ipd_x), dim=-1)
        x = self.fc1(x)
        batch_size = x.shape[0]
        x = x.permute(1, 0, 2)
        h_0 = self.init_hidden_state(x.size(1)) 
        outs = []
        for i in range(x.size(0)):
            tensor = torch.cat((x[i,:,:], h_0), dim=1)
            tensor = self.MM1(tensor)
            x_ = tensor[:,:self.rnn_in]
            h_ = tensor[:,self.rnn_in:]
            h_0 = self.rnn(x_, h_)
            outs.append(h_0)
        out = outs[-1].squeeze()    
        if self.training:
            out = dropout_layer(out,self.dropout, self.device)   
        out = self.MM2(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=-1),0
 
    def init_hidden_state(self, batch_size):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return h_0
    



class MM_final(nn.Module):
    def __init__(self, C, D, output_size, device):
        super(MM_final, self).__init__()
        self.register_buffer('S', torch.randn(C, D, 15).to(device))
        self.register_buffer('H', torch.randn(15, 16).to(device))
        # number of codeblocks that input is divided into,
        # for example, if input size is 14, and C is 7, then input vector is divided into 7 codeblocks, each of size 2.
        self.C = C
        # size of each codeblock
        self.D = D
        # thresholds table.
        self.T = nn.Parameter(torch.randn(C, 15).to(device))
        self.LUT = nn.Parameter(torch.randn(C, 16, output_size).to(device))

    def forward(self, x):
        x = x.view(-1, self.C, self.D)
        x = torch.einsum('bcd,cdk->bck', x, self.S)
        x = x - self.T
        x[x == 0] = -1
        x = torch.sign(x)

        x = torch.einsum('bcd,dk->bck', x, self.H)
        one_hot = F.one_hot(torch.argmax(x, dim=-1), num_classes=16).float()
        x = torch.einsum('bck,cko->bco', one_hot, self.LUT)

        x = torch.sum(x, dim=1)
        return x



class rnn_lut(nn.Module):
    def __init__(self, 
                 rnn_in, hidden_size,
                 labels_num,
                 len_vocab, ipd_vocab,
                 len_embedding_bits, ipd_embedding_bits,
                 device, droprate,
                 D1,D2):
        
        super(rnn_lut, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        self.dropout = droprate
        self.rnn_in = rnn_in
        self.labels_num = labels_num
        self.len_vocab = len_vocab
        self.ipd_vocab = ipd_vocab
        self.len_embedding_bits = len_embedding_bits
        self.ipd_embedding_bits = ipd_embedding_bits
        self.MM1 =MM_final((hidden_size+rnn_in)//D1,D1,hidden_size,device)
        self.MM2 = MM_final(hidden_size//D2,D2,labels_num,device)
        self.lenebdLUT = nn.Parameter(torch.randn(len_vocab, rnn_in).to(device))
        self.ipdebdLUT = nn.Parameter(torch.randn(ipd_vocab, rnn_in).to(device))

    def forward(self, x):
        len_x = x[:,:,0].to(self.device)
        ipd_x = x[:,:,1].to(self.device)
        len_x = len_x.to(self.device)
        ipd_x = ipd_x.to(self.device)
        
        len_x = self.lenebdLUT[len_x]
        ipd_x = self.ipdebdLUT[ipd_x]
        x = ipd_x+len_x
        batch_size = x.shape[0]
        x = x.permute(1, 0, 2)
        h_0 = self.init_hidden_state(x.size(1)) 
        outs = []
        for i in range(x.size(0)):
            x_0 = x[i,:,:]
            rnnin = torch.cat((x_0, h_0), dim=1)
            h_0 = self.MM1(rnnin)
            outs.append(h_0)
        out = outs[-1].squeeze() 
        out = self.MM2(out)
        return F.log_softmax(out, dim=-1)
 
    def init_hidden_state(self, batch_size):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        return h_0